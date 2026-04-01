import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tetrisEngine import TetrisEngine, PIECE_ID_TO_KIND
from fileParsing import (
    load_index,
    iter_game_frames,
    playfield_to_board,
    find_bfs_match_index,
)

"""
TetrisFormer v4 — Search-Distilled Candidate Q Learning
=======================================================

Core idea:
  - For each current state, sample K BFS candidates.
  - For each candidate, compute a short rollout return with beam search
    under the actual future piece sequence from the replay.
  - Train:
      1) rank head with SOFT targets over all K candidates
      2) Q head to regress per-candidate rollout return
      3) attack head to regress immediate attack for all K candidates
      4) weak expert imitation CE as auxiliary regularizer

This is the first version aligned with:
  - attack
  - T-spin setup
  - B2B preservation
  - delayed payoff moves

Assumes the following external symbols are available elsewhere, as before:
  - load_index
  - TetrisEngine
  - iter_game_frames
  - playfield_to_board
  - find_bfs_match_index
"""

import argparse
import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
PIECE_MAP = {"I": 1, "O": 2, "T": 3, "S": 4, "Z": 5, "J": 6, "L": 7}

BOARD_H = 40
BOARD_W = 10
VISIBLE_H = 20
HIDDEN_ROWS = BOARD_H - VISIBLE_H

NUM_BOARD_CHANNELS = 7
NUM_STATS = 7
QUEUE_SLOTS = 5
CACHE_VERSION = 3
DEFAULT_CACHE_DIR_NAME = "game_cache_v3"
DEFAULT_SPIN_MODE = "t_only"
CHECKPOINT_ARCH = "tetrisformer_v4r1"


# ===========================================================================
# Fork-safe global registry for game functions
# ===========================================================================
_GAME_FN_REGISTRY = {}


def _dataloader_worker_init(worker_id):
    """Re-register game functions in each DataLoader worker (needed on Windows/spawn)."""
    register_game_functions(
        TetrisEngine, iter_game_frames, playfield_to_board, find_bfs_match_index
    )


def register_game_functions(engine_factory, iter_game_frames_fn,
                            playfield_to_board_fn, find_bfs_match_index_fn):
    _GAME_FN_REGISTRY["engine_factory"] = engine_factory
    _GAME_FN_REGISTRY["iter_game_frames"] = iter_game_frames_fn
    _GAME_FN_REGISTRY["playfield_to_board"] = playfield_to_board_fn
    _GAME_FN_REGISTRY["find_bfs_match_index"] = find_bfs_match_index_fn


def _get_game_functions():
    if not _GAME_FN_REGISTRY:
        raise RuntimeError(
            "Game functions not registered. "
            "Call register_game_functions(...) before creating DataLoaders."
        )
    return (
        _GAME_FN_REGISTRY["engine_factory"],
        _GAME_FN_REGISTRY["iter_game_frames"],
        _GAME_FN_REGISTRY["playfield_to_board"],
        _GAME_FN_REGISTRY["find_bfs_match_index"],
    )


# ---------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def master_print(*args, **kwargs):
    print(*args, **kwargs)


def optimizer_step(optimizer):
    optimizer.step()


def save_checkpoint(obj, path):
    torch.save(obj, path)


# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------
def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def _safe_piece_str(p) -> str:
    if p is None:
        return ""
    s = str(p)
    if not s or s == "N" or s.lower() == "nan":
        return ""
    return s


def _softmax_np(x: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    z = x.astype(np.float32) / t
    z = z - z.max()
    e = np.exp(z)
    return e / max(float(e.sum()), 1e-8)


def _split_entries(entries, train_split):
    idx = int(len(entries) * train_split)
    return list(entries[:idx]), list(entries[idx:])


# ---------------------------------------------------------------------
# T-slot detection (shared by feature channels and reward shaping)
# ---------------------------------------------------------------------
# T-piece body offsets per rotation within a 3x3 bounding box.
_T_BODY_OFFSETS = [
    [(1, 0), (0, 1), (1, 1), (2, 1)],  # Rot 0 (up)
    [(1, 0), (1, 1), (2, 1), (1, 2)],  # Rot 1 (right)
    [(0, 1), (1, 1), (2, 1), (1, 2)],  # Rot 2 (down)
    [(1, 0), (0, 1), (1, 1), (1, 2)],  # Rot 3 (left)
]
# 3x3 bounding-box corner offsets (same for every rotation).
_T_CORNER_OFFSETS = [(0, 0), (2, 0), (0, 2), (2, 2)]


def detect_tslots(board: np.ndarray) -> np.ndarray:
    """Detect T-slot cavities on a board using the 3-corner rule.

    Returns a (H, W) float32 mask where 1.0 marks cells that are part of a
    valid T-slot cavity (the empty body cells where a T-piece could T-spin).
    Fully vectorized — no Python loops over board positions.

    Accessibility filter: the center column of the 3x3 bounding box must
    have no filled cells above the slot's top row, ensuring the cavity is
    reachable from above (not buried under existing blocks).
    """
    H, W = board.shape
    occ = board != 0  # bool (H, W)

    # Valid placement grid: 3x3 box must fit on board.
    ph = H - 2  # number of valid top-left y positions
    pw = W - 2  # number of valid top-left x positions

    # Corner occupancy sum — same for all rotations.
    corner_sum = (
        occ[:ph, :pw].astype(np.int8)
        + occ[:ph, 2:2 + pw].astype(np.int8)
        + occ[2:2 + ph, :pw].astype(np.int8)
        + occ[2:2 + ph, 2:2 + pw].astype(np.int8)
    )  # shape (ph, pw)

    # Accessibility check: center column (x+1) must have no filled cells
    # above the slot's top row y.  col_first_filled[c] = first filled row
    # index from top in column c (H if the column is entirely empty).
    col_first_filled = np.where(
        occ.any(axis=0), np.argmax(occ, axis=0), H
    ).astype(np.int32)  # shape (W,)
    center_col_first = col_first_filled[1:1 + pw]  # shape (pw,)
    y_rows = np.arange(ph, dtype=np.int32)          # shape (ph,)
    # accessible[y, x] = True iff no filled cell in col x+1 above row y
    accessible = center_col_first[np.newaxis, :] >= y_rows[:, np.newaxis]  # (ph, pw)

    out = np.zeros((H, W), dtype=np.float32)

    for body in _T_BODY_OFFSETS:
        # All 4 body cells must be empty at each candidate position.
        body_empty = np.ones((ph, pw), dtype=bool)
        for dx, dy in body:
            body_empty &= ~occ[dy:dy + ph, dx:dx + pw]

        tslot_pos = body_empty & (corner_sum >= 3) & accessible  # (ph, pw)

        # Paint body cells onto output mask for every detected T-slot.
        for dx, dy in body:
            out[dy:dy + ph, dx:dx + pw] += tslot_pos

    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
def compute_board_features(base_board: np.ndarray, result_board: np.ndarray) -> np.ndarray:
    """
    7-channel float32 tensor from (base_board, result_board).
    Vectorized for maximum speed.
    """
    H, W = base_board.shape
    base_occ = (base_board != 0).astype(np.float32)
    res_occ = (result_board != 0).astype(np.float32)
    diff = np.clip(res_occ - base_occ, 0.0, 1.0)

    # Vectorized column heights
    mask = res_occ > 0
    y_indices = np.argmax(mask, axis=0)
    col_heights = np.where(mask.any(axis=0), H - y_indices, 0).astype(np.float32)
    height_map = np.broadcast_to(col_heights / float(VISIBLE_H), (H, W)).copy()

    # Vectorized holes map
    blocks_above = np.cumsum(mask, axis=0)
    holes = ((res_occ == 0) & (blocks_above > 0)).astype(np.float32)

    # Vectorized row fill
    row_fill = res_occ.sum(axis=1, keepdims=True) / float(W)
    row_fill_map = np.broadcast_to(row_fill, (H, W)).copy()

    # T-slot cavity map on the result board
    tslot_map = detect_tslots(result_board)

    return np.stack([base_occ, res_occ, diff, height_map, holes, row_fill_map, tslot_map], axis=0)


def encode_stats(row: dict, base_board: np.ndarray, result_board: np.ndarray) -> np.ndarray:
    incoming = max(0.0, float(row.get("incoming_garbage", 0) or 0))
    combo = max(0.0, float(row.get("combo", 0) or 0))
    btb = max(0.0, float(row.get("btb", row.get("b2b_chain", 0)) or 0))

    res_occ = (result_board != 0).astype(np.float32)
    H, W = res_occ.shape

    # Vectorized heights
    mask = res_occ > 0
    y_indices = np.argmax(mask, axis=0)
    col_h = np.where(mask.any(axis=0), H - y_indices, 0).astype(np.float32)
    max_height = float(col_h.max())

    # Vectorized holes and bumpiness
    blocks_above = np.cumsum(mask, axis=0)
    n_holes = int(((res_occ == 0) & (blocks_above > 0)).sum())
    bumpiness = float(np.sum(np.abs(np.diff(col_h))))
    
    move_num = float(row.get("move_number", 0) or 0)

    return np.array([
        np.log1p(incoming) / 5.0,
        np.log1p(combo) / 5.0,
        np.log1p(btb) / 5.0,
        max_height / float(VISIBLE_H),
        min(n_holes, 20) / 20.0,
        bumpiness / float(VISIBLE_H * (W - 1)),
        min(move_num, 200) / 200.0,
    ], dtype=np.float32)

def _encode_queue(placed, hold, next_queue, slots=QUEUE_SLOTS):
    seq = []

    def to_id(p):
        return PIECE_MAP.get(_safe_piece_str(p), 0)

    seq.append(to_id(placed))
    seq.append(to_id(hold))
    nq_str = str(next_queue) if next_queue else ""
    for i in range(slots):
        seq.append(to_id(nq_str[i]) if i < len(nq_str) else 0)
    return np.array(seq, dtype=np.int64)


# ---------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------
def build_2d_sincos_pos_embed(h: int, w: int, embed_dim: int) -> torch.Tensor:
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    omega = 1.0 / (10000.0 ** (torch.arange(0, half, 2, dtype=torch.float32) / half))

    rows = torch.arange(h, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(w, dtype=torch.float32).unsqueeze(1)

    row_enc = torch.zeros(h, half)
    row_enc[:, 0::2] = torch.sin(rows * omega)
    row_enc[:, 1::2] = torch.cos(rows * omega)

    col_enc = torch.zeros(w, half)
    col_enc[:, 0::2] = torch.sin(cols * omega)
    col_enc[:, 1::2] = torch.cos(cols * omega)

    pos = torch.zeros(h, w, embed_dim)
    pos[:, :, :half] = row_enc.unsqueeze(1).expand(h, w, half)
    pos[:, :, half:] = col_enc.unsqueeze(0).expand(h, w, half)
    return pos.reshape(1, h * w, embed_dim)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class _ResidualConvBlock(nn.Module):
    """Conv + BN + ReLU with a residual shortcut (handles stride and channel mismatch)."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)) + self.shortcut(x))


class TetrisFormerV4(nn.Module):
    def __init__(self, embed_dim=192, num_heads=6, depth=4,
                 board_channels=NUM_BOARD_CHANNELS, num_stats=NUM_STATS):
        super().__init__()

        # CNN with residual connections (skip connections on stride-2 layers).
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(board_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.cnn_res1 = _ResidualConvBlock(32, 64, stride=2)
        self.cnn_res2 = _ResidualConvBlock(64, embed_dim, stride=2)

        self.piece_embedding = nn.Embedding(8, embed_dim)
        self.queue_pos_embed = nn.Parameter(torch.randn(1, 10, embed_dim))
        self.stats_proj = nn.Linear(num_stats, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True,
            dropout=0.1,
            norm_first=True,  # pre-LayerNorm: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.stats_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self._grid_pos_embed = None
        self._embed_dim = embed_dim

        self.rank_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.attack_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.q_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def _get_grid_pos_embed(self, h, w, device):
        if self._grid_pos_embed is not None and self._grid_pos_embed.shape[1] == h * w:
            return self._grid_pos_embed.to(device)
        self._grid_pos_embed = build_2d_sincos_pos_embed(h, w, self._embed_dim).to(device)
        return self._grid_pos_embed

    def forward(self, x_board, x_queue, x_stats):
        vis_feats = self.cnn_res2(self.cnn_res1(self.cnn_stem(x_board)))
        B, C, Hf, Wf = vis_feats.shape
        grid_tokens = vis_feats.view(B, C, Hf * Wf).permute(0, 2, 1)
        grid_tokens = grid_tokens + self._get_grid_pos_embed(Hf, Wf, grid_tokens.device)

        queue_tokens = self.piece_embedding(x_queue)
        queue_tokens = queue_tokens + self.queue_pos_embed[:, :queue_tokens.size(1), :]

        stats_embed = self.stats_proj(x_stats).unsqueeze(1)
        stats_tokens = stats_embed + self.stats_token

        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq = torch.cat((cls_tokens, stats_tokens, queue_tokens, grid_tokens), dim=1)
        attended = self.transformer(seq)
        cls_out = attended[:, 0, :]

        rank_score = self.rank_head(cls_out)
        pred_attack = self.attack_head(cls_out)
        pred_q = self.q_head(cls_out)
        return rank_score, pred_attack, pred_q



# ---------------------------------------------------------------------
# Engine state helpers
# ---------------------------------------------------------------------
def _clone_engine(engine):
    try:
        return copy.deepcopy(engine)
    except Exception:
        clone = engine.__class__(spin_mode=getattr(engine, "spin_mode", DEFAULT_SPIN_MODE))
        clone.board = engine.board.copy()

        for attr in [
            "bag", "bag_size", "hold",
            "b2b_chain", "surge_charge",
            "last_clear_stats", "combo", "combo_active",
            "game_over", "game_over_reason",
            "last_spawn_was_clutch", "last_end_phase",
            "incoming_garbage", "garbage_col"
        ]:
            if hasattr(engine, attr):
                setattr(clone, attr, copy.deepcopy(getattr(engine, attr)))

        if getattr(engine, "current_piece", None) is not None:
            p = engine.current_piece
            clone.current_piece = clone.spawn_piece(str(p.kind))
            clone.current_piece.rotation = int(p.rotation)
            clone.current_piece.position = tuple(p.position)
            clone.current_piece.last_action_was_rotation = bool(
                getattr(p, "last_action_was_rotation", False)
            )
            clone.current_piece.last_rotation_dir = getattr(p, "last_rotation_dir", None)
            clone.current_piece.last_kick_index = getattr(p, "last_kick_index", None)
        else:
            clone.current_piece = None
        return clone


def _set_engine_root_state(engine, step_info: dict):
    base_board = step_info["base_board"]
    pre_state = step_info["pre_state"]
    queue_state = step_info["queue_state"]
    placed_kind = _safe_piece_str(queue_state.get("current") or step_info.get("placed"))

    engine.board = base_board.copy()
    if hasattr(engine, "bag"):
        engine.bag = np.array([], dtype=int)
    if hasattr(engine, "bag_size"):
        engine.bag_size = 0

    if hasattr(engine, "combo"):
        engine.combo = _safe_int(pre_state.get("combo", 0))
    if hasattr(engine, "combo_active"):
        engine.combo_active = bool(pre_state.get("combo_active", engine.combo > 0))
    if hasattr(engine, "b2b_chain"):
        engine.b2b_chain = _safe_int(pre_state.get("b2b_chain", 0))
    if hasattr(engine, "surge_charge"):
        engine.surge_charge = _safe_int(pre_state.get("surge_charge", 0))
    if hasattr(engine, "hold"):
        engine.hold = _safe_piece_str(queue_state.get("hold")) or None
    if hasattr(engine, "incoming_garbage"):
        incoming_total = _safe_int(pre_state.get("incoming_garbage_total", 0))
        engine.incoming_garbage = (
            [{"lines": incoming_total, "timer": 0, "col": 0}]
            if incoming_total > 0
            else []
        )
    if hasattr(engine, "garbage_col"):
        engine.garbage_col = None

    engine.game_over = False
    engine.game_over_reason = None
    engine.last_clear_stats = None
    if hasattr(engine, "last_end_phase"):
        engine.last_end_phase = None

    engine.current_piece = engine.spawn_piece(placed_kind)
    if engine.current_piece is None:
        raise RuntimeError(f"Failed to spawn piece '{placed_kind}'")
    return engine


def _apply_placement_fields_to_current_piece(engine, placement: dict):
    p = engine.current_piece
    if p is None:
        return False

    x = int(placement.get("x", p.position[0]))
    y = int(placement.get("y", p.position[1]))
    rot = int(placement.get("rotation", 0))

    p.position = (x, y)
    p.rotation = rot
    p.last_action_was_rotation = bool(placement.get("last_was_rot", False))

    last_dir = placement.get("last_rot_dir")
    p.last_rotation_dir = None if last_dir in (None, 0) else int(last_dir)

    last_kick = placement.get("last_kick_idx")
    if last_kick is None:
        p.last_kick_index = None
    else:
        last_kick = int(last_kick)
        p.last_kick_index = None if last_kick < 0 else last_kick

    try:
        valid = engine.is_position_valid(p, p.position, p.rotation)
    except TypeError:
        valid = engine.is_position_valid(p, position=p.position)
    return bool(valid)


def _board_shape_metrics(board: np.ndarray, tslot_hole_discount: float = 0.5) -> Tuple[float, float, float]:
    occ = (board != 0).astype(np.float32)
    H, W = occ.shape

    mask = occ > 0
    y_indices = np.argmax(mask, axis=0)
    col_h = np.where(mask.any(axis=0), H - y_indices, 0).astype(np.float32)

    blocks_above = np.cumsum(mask, axis=0)
    all_holes = (occ == 0) & (blocks_above > 0)

    # Discount holes that are part of a complete, accessible T-slot cavity.
    tslot_mask = detect_tslots(board)
    has_tslot = float(tslot_mask.any())
    productive = all_holes & (tslot_mask > 0)
    unproductive = all_holes & (tslot_mask == 0)
    effective_holes = float(unproductive.sum()) + (1.0 - tslot_hole_discount) * float(productive.sum())

    return float(col_h.max()) / float(VISIBLE_H), min(effective_holes, 20) / 20.0, has_tslot


def _is_tspin_from_stats(clear_stats: dict) -> float:
    if clear_stats.get("lines_cleared", 0) <= 0:
        return 0.0
    spin = clear_stats.get("spin")
    if isinstance(spin, dict) and spin.get("spin_type") == "t-spin":
        return 1.0
    return 0.0


def _compute_rollout_reward(board_after: np.ndarray,
                            clear_stats: dict,
                            prev_b2b: int,
                            game_over: bool,
                            attack_w: float,
                            tspin_bonus: float,
                            b2b_bonus: float,
                            height_penalty: float,
                            holes_penalty: float,
                            topout_penalty: float,
                            tslot_ready_bonus: float = 0.0) -> Tuple[float, float]:
    attack = float(_safe_float(clear_stats.get("attack", 0.0)))
    tspin = _is_tspin_from_stats(clear_stats)

    b2b_cont = 0.0
    if _safe_float(clear_stats.get("b2b_bonus", 0.0)) > 0:
        b2b_cont = 1.0
    elif prev_b2b > 0 and attack > 0:
        b2b_cont = 1.0

    height_norm, holes_norm, has_tslot = _board_shape_metrics(board_after)

    reward = 0.0
    reward += attack_w * attack
    reward += tspin_bonus * tspin
    reward += b2b_bonus * b2b_cont
    reward += tslot_ready_bonus * has_tslot
    reward -= height_penalty * height_norm
    reward -= holes_penalty * holes_norm
    reward -= topout_penalty * float(game_over)

    return float(reward), float(attack)


def _simulate_current_placement(engine,
                                placement: dict,
                                next_piece_kind: Optional[str],
                                attack_w: float,
                                tspin_bonus: float,
                                b2b_bonus: float,
                                height_penalty: float,
                                holes_penalty: float,
                                topout_penalty: float,
                                tslot_ready_bonus: float = 0.0):
    sim = _clone_engine(engine)

    if sim.current_piece is None:
        return None, -topout_penalty, 0.0

    valid = _apply_placement_fields_to_current_piece(sim, placement)
    if not valid:
        sim.game_over = True
        sim.game_over_reason = "invalid_placement"
        reward, attack = _compute_rollout_reward(
            sim.board, {}, int(getattr(sim, "b2b_chain", 0)), True,
            attack_w, tspin_bonus, b2b_bonus,
            height_penalty, holes_penalty, topout_penalty, tslot_ready_bonus
        )
        return sim, reward, attack

    prev_b2b = int(getattr(sim, "b2b_chain", 0))

    sim.lock_piece(run_end_phase=False)
    clear_stats = sim.last_clear_stats or {}

    # Patch next spawn to the actual replay piece, not a random bag spawn.
    if not sim.game_over and next_piece_kind:
        actual_next = sim.spawn_piece(next_piece_kind)
        can_spawn = False
        if actual_next is not None:
            try:
                can_spawn = sim.is_position_valid(
                    actual_next, actual_next.position, actual_next.rotation
                )
            except TypeError:
                can_spawn = sim.is_position_valid(actual_next, position=actual_next.position)

        if can_spawn:
            sim.current_piece = actual_next
            sim.game_over = False
            sim.game_over_reason = None
        else:
            sim.current_piece = None
            sim.game_over = True
            sim.game_over_reason = "block_out"
    else:
        sim.current_piece = None

    reward, attack = _compute_rollout_reward(
        sim.board, clear_stats, prev_b2b, bool(sim.game_over),
        attack_w, tspin_bonus, b2b_bonus,
        height_penalty, holes_penalty, topout_penalty, tslot_ready_bonus
    )
    return sim, reward, attack


def _beam_rollout_future(engine_after_root,
                         future_piece_kinds: List[str],
                         next_index: int,
                         depth_remaining: int,
                         beam_width: int,
                         expand_width: int,
                         gamma: float,
                         attack_w: float,
                         tspin_bonus: float,
                         b2b_bonus: float,
                         height_penalty: float,
                         holes_penalty: float,
                         topout_penalty: float,
                         tslot_ready_bonus: float = 0.0) -> float:
    if depth_remaining <= 0:
        return 0.0

    root = {
        "engine": _clone_engine(engine_after_root),
        "next_index": int(next_index),
        "score": 0.0,
    }
    beam = [root]
    best = 0.0

    for d in range(depth_remaining):
        expanded = []

        for node in beam:
            eng = node["engine"]
            cur_piece = getattr(eng, "current_piece", None)

            if eng.game_over or cur_piece is None:
                best = max(best, float(node["score"]))
                continue

            try:
                results = eng.bfs_all_placements(include_no_place=False)
            except Exception:
                results = []

            if not results:
                best = max(best, float(node["score"]))
                continue

            local = []
            next_piece_kind = (
                future_piece_kinds[node["next_index"]]
                if node["next_index"] < len(future_piece_kinds)
                else None
            )

            for res in results:
                placement = res.get("placement")
                if not placement:
                    continue

                sim_eng, step_reward, _step_attack = _simulate_current_placement(
                    eng, placement, next_piece_kind,
                    attack_w, tspin_bonus, b2b_bonus,
                    height_penalty, holes_penalty, topout_penalty, tslot_ready_bonus,
                )

                if sim_eng is None:
                    continue

                local.append({
                    "engine": sim_eng,
                    "next_index": node["next_index"] + (1 if next_piece_kind else 0),
                    "score": float(node["score"]) + (gamma ** d) * float(step_reward),
                })

            if not local:
                best = max(best, float(node["score"]))
                continue

            local.sort(key=lambda x: x["score"], reverse=True)
            if expand_width > 0:
                local = local[:expand_width]
            expanded.extend(local)

        if not expanded:
            break

        expanded.sort(key=lambda x: x["score"], reverse=True)
        beam = expanded[:beam_width]
        best = max(best, float(beam[0]["score"]))

    if beam:
        best = max(best, max(float(n["score"]) for n in beam))
    return float(best)


def _compute_candidate_rollout_q(root_engine,
                                 placement: dict,
                                 future_piece_kinds: List[str],
                                 rollout_depth: int,
                                 rollout_beam: int,
                                 rollout_expand: int,
                                 gamma: float,
                                 attack_w: float,
                                 tspin_bonus: float,
                                 b2b_bonus: float,
                                 height_penalty: float,
                                 holes_penalty: float,
                                 topout_penalty: float,
                                 tslot_ready_bonus: float = 0.0) -> Tuple[float, float]:
    next_piece_kind = future_piece_kinds[0] if len(future_piece_kinds) > 0 else None

    after_engine, r0, immediate_attack = _simulate_current_placement(
        root_engine,
        placement,
        next_piece_kind,
        attack_w, tspin_bonus, b2b_bonus,
        height_penalty, holes_penalty, topout_penalty, tslot_ready_bonus,
    )
    if after_engine is None:
        return -topout_penalty, 0.0

    if rollout_depth <= 1 or next_piece_kind is None or after_engine.game_over:
        return float(r0), float(immediate_attack)

    future_return = _beam_rollout_future(
        after_engine,
        future_piece_kinds=future_piece_kinds,
        next_index=1,
        depth_remaining=rollout_depth - 1,
        beam_width=rollout_beam,
        expand_width=rollout_expand,
        gamma=gamma,
        attack_w=attack_w,
        tspin_bonus=tspin_bonus,
        b2b_bonus=b2b_bonus,
        height_penalty=height_penalty,
        holes_penalty=holes_penalty,
        topout_penalty=topout_penalty,
        tslot_ready_bonus=tslot_ready_bonus,
    )

    q_total = float(r0) + float(gamma) * float(future_return)
    return q_total, float(immediate_attack)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class SmartRolloutRankDataset(IterableDataset):
    """
    v4 dataset:
      - sample K candidates for each move
      - compute rollout Q target for every candidate
      - build soft target distribution from candidate rollout returns
      - include per-candidate stats
    """

    def __init__(self,
                 entries,
                 *,
                 data_path,
                 k_candidates=32,
                 max_garbage=8,
                 max_games=None,
                 max_samples=None,
                 important_weight=4.0,
                 important_q_threshold=1.5,
                 gamma=0.9,
                 rollout_depth=4,
                 rollout_beam=4,
                 rollout_expand=8,
                 soft_target_temp=0.35,
                 q_target_scale=20.0,
                 attack_target_scale=10.0,
                 reward_attack_weight=1.0,
                 reward_tspin_bonus=0.75,
                 reward_b2b_bonus=0.35,
                 reward_height_penalty=0.05,
                 reward_holes_penalty=0.08,
                 reward_topout_penalty=2.5,
                 reward_tslot_ready_bonus=0.0,
                 seed=1234,
                 cache_dir=None):
        self.entries = entries
        self.data_path = data_path
        self.cache_dir = cache_dir or os.path.join(_HERE, DEFAULT_CACHE_DIR_NAME)
        self.k_candidates = int(k_candidates)
        self.max_garbage = int(max_garbage)
        self.max_games = max_games
        self.max_samples = max_samples
        self.important_weight = float(important_weight)
        self.important_q_threshold = float(important_q_threshold)
        self.gamma = float(gamma)

        self.rollout_depth = int(rollout_depth)
        self.rollout_beam = int(rollout_beam)
        self.rollout_expand = int(rollout_expand)
        self.soft_target_temp = float(soft_target_temp)
        self.q_target_scale = float(q_target_scale)
        self.attack_target_scale = float(attack_target_scale)

        self.reward_attack_weight = float(reward_attack_weight)
        self.reward_tspin_bonus = float(reward_tspin_bonus)
        self.reward_b2b_bonus = float(reward_b2b_bonus)
        self.reward_height_penalty = float(reward_height_penalty)
        self.reward_holes_penalty = float(reward_holes_penalty)
        self.reward_topout_penalty = float(reward_topout_penalty)
        self.reward_tslot_ready_bonus = float(reward_tslot_ready_bonus)

        self.seed = int(seed)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            my_entries, worker_id = self.entries, 0
        else:
            per_worker = int(math.ceil(len(self.entries) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.entries))
            my_entries = self.entries[start:end]
            worker_id = worker_info.id

        import time
        rng = np.random.default_rng(self.seed + 1000 * worker_id)

        print(f"[Worker {worker_id}] STARTED | Entries: {len(my_entries)} | PID: {os.getpid()}")

        games_seen = 0
        samples_seen = 0
        total_start_time = time.time()
        last_log_time = total_start_time

        for entry in my_entries:
            if self.max_games and games_seen >= self.max_games:
                break

            cache_path = os.path.join(self.cache_dir, f"{entry['game_id']}.pt")
            if not os.path.exists(cache_path):
                continue

            if games_seen % 50 == 0:
                print(f"[Worker {worker_id}] Processing game {games_seen}... (game_id: {entry['game_id']})")


            game_data = torch.load(cache_path, weights_only=False)
            cache_version = int(game_data.get("cache_version", 0) or 0)
            if cache_version != CACHE_VERSION:
                raise RuntimeError(
                    f"Unsupported cache version in {cache_path}: expected v{CACHE_VERSION}, "
                    f"found v{cache_version or 'missing'}. Rerun `uv run python preparse_games.py`."
                )
            steps = game_data.get("steps", [])
            games_seen += 1
            EngineClass = _get_game_functions()[0]

            for step_info in steps:
                if self.max_samples and samples_seen >= self.max_samples:
                    return

                step_start = time.time()
                placed = step_info["placed"]
                base_board = step_info["base_board"]
                match_idx = step_info["expert_match_index"]
                valid_indices = step_info["valid_indices"]
                bfs_boards = step_info["bfs_boards"]
                bfs_placements = step_info["bfs_placements"]
                queue_state = step_info["queue_state"]
                pre_state = step_info["pre_state"]
                expert_replay = step_info["expert_replay"]
                
                # We cap rollout depth to what we asked for in args
                future_piece_kinds = step_info["future_pieces"][:self.rollout_depth]

                # Select candidates (Expert + random K-1 others)
                pool = [i for i in valid_indices if i != match_idx]
                if len(pool) >= (self.k_candidates - 1):
                    others = rng.choice(pool, size=(self.k_candidates - 1), replace=False).tolist()
                elif len(pool) > 0:
                    others = rng.choice(pool, size=(self.k_candidates - 1), replace=True).tolist()
                else:
                    others = [match_idx] * (self.k_candidates - 1)

                selected = [match_idx] + others
                rng.shuffle(selected)
                expert_label = int(selected.index(match_idx))

                # Prepare the Root Engine for Q-Rollout
                root_engine = EngineClass(spin_mode=DEFAULT_SPIN_MODE)
                try:
                    _set_engine_root_state(root_engine, step_info)
                except Exception:
                    continue

                queue_seq = _encode_queue(
                    queue_state.get("current") or placed,
                    queue_state.get("hold"),
                    queue_state.get("next_queue"),
                )
                row_for_stats = {
                    "incoming_garbage": pre_state.get("incoming_garbage_total", 0),
                    "combo": pre_state.get("combo", 0),
                    "btb": pre_state.get("b2b_chain", 0),
                    "b2b_chain": pre_state.get("b2b_chain", 0),
                    "move_number": step_info["move_number"],
                }

                boards_list, queues_list, stats_list = [], [], []
                q_raw_list, attack_raw_list = [], []

                for cand_idx in selected:
                    result_board = bfs_boards[cand_idx]
                    placement = bfs_placements[cand_idx]

                    boards_list.append(compute_board_features(base_board, result_board))
                    queues_list.append(queue_seq)
                    stats_list.append(encode_stats(row_for_stats, base_board, result_board))

                    # The rollout engine will still simulate the future tree,
                    # but the root node's candidates are instantly loaded!
                    q_raw, atk_raw = _compute_candidate_rollout_q(
                        root_engine=root_engine,
                        placement=placement,
                        future_piece_kinds=future_piece_kinds,
                        rollout_depth=self.rollout_depth,
                        rollout_beam=self.rollout_beam,
                        rollout_expand=self.rollout_expand,
                        gamma=self.gamma,
                        attack_w=self.reward_attack_weight,
                        tspin_bonus=self.reward_tspin_bonus,
                        b2b_bonus=self.reward_b2b_bonus,
                        height_penalty=self.reward_height_penalty,
                        holes_penalty=self.reward_holes_penalty,
                        topout_penalty=self.reward_topout_penalty,
                        tslot_ready_bonus=self.reward_tslot_ready_bonus,
                    )
                    q_raw_list.append(q_raw)
                    attack_raw_list.append(atk_raw)

                q_raw_arr = np.asarray(q_raw_list, dtype=np.float32)
                attack_raw_arr = np.asarray(attack_raw_list, dtype=np.float32)

                soft_targets = _softmax_np(q_raw_arr, self.soft_target_temp)
                q_targets = np.clip(q_raw_arr / max(self.q_target_scale, 1e-6), -2.0, 2.0)
                attack_targets = np.clip(attack_raw_arr / max(self.attack_target_scale, 1e-6), 0.0, 2.0)

                expert_q = float(q_raw_arr[expert_label])
                actual_attack = float(_safe_float(expert_replay.get("attack", 0)))
                actual_cleared = float(_safe_float(expert_replay.get("cleared", 0)))

                sample_weight = self.important_weight if (
                    actual_attack > 0.0 or actual_cleared > 0.0 or expert_q >= self.important_q_threshold
                ) else 1.0

                yield (
                    torch.from_numpy(np.stack(boards_list, axis=0).astype(np.float32)),
                    torch.from_numpy(np.stack(queues_list, axis=0).astype(np.int64)),
                    torch.from_numpy(np.stack(stats_list, axis=0).astype(np.float32)),
                    torch.tensor(expert_label, dtype=torch.long),
                    torch.tensor(sample_weight, dtype=torch.float32),
                    torch.from_numpy(soft_targets.astype(np.float32)),
                    torch.from_numpy(q_targets.astype(np.float32)),
                    torch.from_numpy(attack_targets.astype(np.float32)),
                )

                samples_seen += 1
                step_time = time.time() - step_start
                
                # Log every 50 samples to show progress without spamming
                if samples_seen % 50 == 0:
                    elapsed = time.time() - total_start_time
                    samples_per_sec = samples_seen / elapsed if elapsed > 0 else 0
                    print(f"[Worker {worker_id}] Sample {samples_seen} | "
                          f"Last step: {step_time:.3f}s | "
                          f"Avg: {samples_per_sec:.2f} samples/sec | "
                          f"Game {entry['game_id']}")
                
                if self.max_samples and samples_seen >= self.max_samples:
                    return


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    from fileParsing import DATA_PATH as _DEFAULT_DATA, INDEX_PATH as _DEFAULT_INDEX
    parser.add_argument("--data-path",  default=_DEFAULT_DATA)
    parser.add_argument("--index-path", default=_DEFAULT_INDEX)
    parser.add_argument("--cache-dir",  default=os.path.join(_HERE, DEFAULT_CACHE_DIR_NAME))
    parser.add_argument("--save-dir",   default=os.path.join(_HERE, "checkpoints"))

    parser.add_argument("--epochs", type=int, default=25)            # was 15; RL needs more time to converge
    parser.add_argument("--lr", type=float, default=2e-4)             # was 2.3e-4; slightly more conservative
    parser.add_argument("--batch-size", type=int, default=128)        # was 64; larger batch → more stable gradients
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--k-candidates", type=int, default=10)       # was 32; covers max placements per piece
    parser.add_argument("--samples-per-epoch", type=int, default=250_000)  # was 100k; more diverse board states
    parser.add_argument("--batches-per-epoch", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=500)

    parser.add_argument("--max-garbage", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.97)          # was 0.90; longer horizon for board cleanliness
    parser.add_argument("--seed", type=int, default=6767)

    # Search target generation
    parser.add_argument("--rollout-depth", type=int, default=6)       # was 3; captures current+hold+next+next logic
    parser.add_argument("--rollout-beam", type=int, default=2)        # was 3; wider beam finds T-spin setups
    parser.add_argument("--rollout-expand", type=int, default=6)      # was 6; match wider beam
    parser.add_argument("--soft-target-temp", type=float, default=0.5)     # was 0.35; softer targets reduce overfit to noisy search
    parser.add_argument("--q-target-scale", type=float, default=20.0)
    parser.add_argument("--attack-target-scale", type=float, default=12.0) # was 10.0; attack is spiky, higher scale prevents dead neurons

    # Reward shaping
    parser.add_argument("--reward-attack-weight", type=float, default=1.0)
    parser.add_argument("--reward-tspin-bonus", type=float, default=1.5)   # was 0.75; incentivise setup effort
    parser.add_argument("--reward-b2b-bonus", type=float, default=0.5)     # was 0.35; B2B chains are high-value
    parser.add_argument("--reward-height-penalty", type=float, default=0.1)    # was 0.05; height kills
    parser.add_argument("--reward-holes-penalty", type=float, default=0.5)     # was 0.08; holes are catastrophic
    parser.add_argument("--reward-topout-penalty", type=float, default=10.0)   # was 2.5; game-over should dominate
    parser.add_argument("--reward-tslot-ready-bonus", type=float, default=0.3)  # per-step bonus for having a complete accessible T-slot on the board

    # Loss weights
    parser.add_argument("--soft-rank-loss-weight", type=float, default=1.0)
    parser.add_argument("--q-loss-weight", type=float, default=1.0)
    parser.add_argument("--imit-loss-weight", type=float, default=0.5)     # was 0.2; human replays are high quality
    parser.add_argument("--attack-loss-weight", type=float, default=0.1)   # was 0.3; keep auxiliary from dominating

    parser.add_argument("--important-weight", type=float, default=6.0)     # was 4.0; emphasise critical states harder
    parser.add_argument("--important-q-threshold", type=float, default=1.5)

    parser.add_argument("--rank-q-alpha", type=float, default=0.3)     # was 0.25; slight favour toward rank signal
    parser.add_argument("--log-every", type=int, default=500)

    # Training stability / efficiency
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm (0 to disable).")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Linear LR warmup steps before cosine decay.")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Minimum LR at end of cosine schedule.")

    args, _ = parser.parse_known_args()

    device = get_device()
    master_print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    master_print(f"Loading game index from {args.index_path} ...")
    entries = load_index(args.index_path)
    train_entries, val_entries = _split_entries(entries, 0.9)
    master_print(
        f"Loaded {len(entries)} games | train={len(train_entries)} | val={len(val_entries)}"
    )

    register_game_functions(
        TetrisEngine, iter_game_frames, playfield_to_board, find_bfs_match_index
    )

    train_set = SmartRolloutRankDataset(
        train_entries,
        data_path=args.data_path,
        k_candidates=args.k_candidates,
        max_garbage=args.max_garbage,
        important_weight=args.important_weight,
        important_q_threshold=args.important_q_threshold,
        gamma=args.gamma,
        rollout_depth=args.rollout_depth,
        rollout_beam=args.rollout_beam,
        rollout_expand=args.rollout_expand,
        soft_target_temp=args.soft_target_temp,
        q_target_scale=args.q_target_scale,
        attack_target_scale=args.attack_target_scale,
        reward_attack_weight=args.reward_attack_weight,
        reward_tspin_bonus=args.reward_tspin_bonus,
        reward_b2b_bonus=args.reward_b2b_bonus,
        reward_height_penalty=args.reward_height_penalty,
        reward_holes_penalty=args.reward_holes_penalty,
        reward_topout_penalty=args.reward_topout_penalty,
        reward_tslot_ready_bonus=args.reward_tslot_ready_bonus,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    val_set = SmartRolloutRankDataset(
        val_entries,
        data_path=args.data_path,
        k_candidates=args.k_candidates,
        max_garbage=args.max_garbage,
        important_weight=1.0,
        important_q_threshold=args.important_q_threshold,
        gamma=args.gamma,
        rollout_depth=args.rollout_depth,
        rollout_beam=args.rollout_beam,
        rollout_expand=args.rollout_expand,
        soft_target_temp=args.soft_target_temp,
        q_target_scale=args.q_target_scale,
        attack_target_scale=args.attack_target_scale,
        reward_attack_weight=args.reward_attack_weight,
        reward_tspin_bonus=args.reward_tspin_bonus,
        reward_b2b_bonus=args.reward_b2b_bonus,
        reward_height_penalty=args.reward_height_penalty,
        reward_holes_penalty=args.reward_holes_penalty,
        reward_topout_penalty=args.reward_topout_penalty,
        reward_tslot_ready_bonus=args.reward_tslot_ready_bonus,
        seed=args.seed + 999,
        cache_dir=args.cache_dir,
    )

    _wif = _dataloader_worker_init if args.num_workers > 0 else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
        worker_init_fn=_wif,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
        worker_init_fn=_wif,
    )

    model = TetrisFormerV4(
        board_channels=NUM_BOARD_CHANNELS,
        num_stats=NUM_STATS,
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Cosine LR schedule with linear warmup.
    total_steps_estimate = args.epochs * (args.samples_per_epoch // max(args.batch_size, 1))

    def _lr_lambda(step):
        if step < args.warmup_steps:
            return max(float(step) / max(args.warmup_steps, 1), 1e-2)
        progress = (step - args.warmup_steps) / max(total_steps_estimate - args.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(cosine, args.min_lr / args.lr)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # Mixed precision (only on CUDA).
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ce_criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.05)
    smooth_l1 = nn.SmoothL1Loss(reduction="none")

    os.makedirs(args.save_dir, exist_ok=True)

    def next_batch(it, loader, name="loader"):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            try:
                return next(it), it
            except StopIteration:
                raise RuntimeError(
                    f"{name} produced no batches. "
                    f"Dataset is likely empty or filtered to zero samples."
                )

    train_iter = iter(train_loader)

    master_print("Starting v4 training ...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        master_print(f"--- Epoch {epoch}/{args.epochs} ---")

        total_loss = 0.0
        total_soft = 0.0
        total_q = 0.0
        total_imit = 0.0
        total_atk = 0.0

        total_samples = 0
        total_batches = 0

        total_expert_acc = 0
        total_qbest_acc = 0
        total_final_acc = 0

        cap_batches = args.batches_per_epoch if args.batches_per_epoch > 0 else None
        cap_samples = None if cap_batches is not None else args.samples_per_epoch

        while True:
            if cap_batches is not None and total_batches >= cap_batches:
                break
            if cap_samples is not None and total_samples >= cap_samples:
                break

            master_print(f"  [DEBUG] Fetching batch {total_batches + 1}...")
            batch, train_iter = next_batch(train_iter, train_loader)
            master_print(f"  [DEBUG] Batch {total_batches + 1} fetched. Running forward/backward pass...")

            (
                boards, queues, stats,
                expert_labels, weights,
                soft_targets, q_targets, attack_targets
            ) = batch

            boards = boards.to(device)
            queues = queues.to(device)
            stats = stats.to(device)
            expert_labels = expert_labels.to(device)
            weights = weights.to(device)
            soft_targets = soft_targets.to(device)
            q_targets = q_targets.to(device)
            attack_targets = attack_targets.to(device)

            B, K, C, H, W = boards.shape
            _, _, L = queues.shape
            _, _, S = stats.shape

            boards_flat = boards.reshape(B * K, C, H, W)
            queues_flat = queues.reshape(B * K, L)
            stats_flat = stats.reshape(B * K, S)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                rank_scores, pred_attack, pred_q = model(boards_flat, queues_flat, stats_flat)

                rank_logits = rank_scores.squeeze(-1).reshape(B, K)
                pred_attack = pred_attack.squeeze(-1).reshape(B, K)
                pred_q = pred_q.squeeze(-1).reshape(B, K)

                # 1) Soft rank loss from rollout-return distribution.
                log_probs = torch.log_softmax(rank_logits, dim=1)
                soft_rank_loss = -(soft_targets * log_probs).sum(dim=1)
                soft_rank_loss = (soft_rank_loss * weights).mean()

                # 2) Q regression on all K candidates.
                q_loss_per = smooth_l1(pred_q, q_targets).mean(dim=1)
                q_loss = (q_loss_per * weights).mean()

                # 3) Immediate attack regression on all K candidates.
                attack_loss_per = smooth_l1(pred_attack, attack_targets).mean(dim=1)
                attack_loss = (attack_loss_per * weights).mean()

                # 4) Weak expert imitation.
                imit_loss = ce_criterion(rank_logits, expert_labels)
                imit_loss = (imit_loss * weights).mean()

                loss = (
                    args.soft_rank_loss_weight * soft_rank_loss
                    + args.q_loss_weight * q_loss
                    + args.attack_loss_weight * attack_loss
                    + args.imit_loss_weight * imit_loss
                )

            if not torch.isfinite(loss):
                master_print(f"[WARN] Non-finite loss at epoch {epoch}; skipping batch.")
                continue

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                target_best = torch.argmax(q_targets, dim=1)
                pred_expert = torch.argmax(rank_logits, dim=1)
                pred_qbest = torch.argmax(pred_q, dim=1)
                pred_final = torch.argmax(rank_logits + args.rank_q_alpha * pred_q, dim=1)

                total_expert_acc += (pred_expert == expert_labels).sum().item()
                total_qbest_acc += (pred_qbest == target_best).sum().item()
                total_final_acc += (pred_final == target_best).sum().item()

            total_batches += 1
            total_samples += B

            total_loss += loss.item() * B
            total_soft += soft_rank_loss.item() * B
            total_q += q_loss.item() * B
            total_atk += attack_loss.item() * B
            total_imit += imit_loss.item() * B

            if args.log_every > 0 and (total_samples % args.log_every) < B:
                n = max(1, total_samples)
                master_print(
                    f"Ep {epoch} | Samples {total_samples} | Batches {total_batches} | "
                    f"Loss {total_loss/n:.4f} "
                    f"(SoftRank {total_soft/n:.4f}, Q {total_q/n:.4f}, "
                    f"Atk {total_atk/n:.4f}, Imit {total_imit/n:.4f}) | "
                    f"ExpertAcc {100.0*total_expert_acc/n:.2f}% | "
                    f"QBestAcc {100.0*total_qbest_acc/n:.2f}% | "
                    f"FinalAcc {100.0*total_final_acc/n:.2f}%"
                )

        # ---------------- Validation ----------------
        model.eval()
        val_iter = iter(val_loader)

        v_expert = 0
        v_qbest = 0
        v_final = 0
        v_samples = 0
        v_batches = 0

        with torch.no_grad():
            while True:
                if args.val_batches > 0 and v_batches >= args.val_batches:
                    break
                
                master_print(f"  [DEBUG] Fetching val batch {v_batches + 1}...")
                batch, val_iter = next_batch(val_iter, val_loader)
                master_print(f"  [DEBUG] Val batch {v_batches + 1} fetched. Evaluating...")

                (
                    boards, queues, stats,
                    expert_labels, _weights,
                    _soft_targets, q_targets, _attack_targets
                ) = batch

                boards = boards.to(device)
                queues = queues.to(device)
                stats = stats.to(device)
                expert_labels = expert_labels.to(device)
                q_targets = q_targets.to(device)

                B, K, C, H, W = boards.shape
                _, _, L = queues.shape
                _, _, S = stats.shape

                boards_flat = boards.reshape(B * K, C, H, W)
                queues_flat = queues.reshape(B * K, L)
                stats_flat = stats.reshape(B * K, S)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    rank_scores, _pred_attack, pred_q = model(boards_flat, queues_flat, stats_flat)
                rank_logits = rank_scores.squeeze(-1).reshape(B, K)
                pred_q = pred_q.squeeze(-1).reshape(B, K)

                target_best = torch.argmax(q_targets, dim=1)
                pred_expert = torch.argmax(rank_logits, dim=1)
                pred_qbest = torch.argmax(pred_q, dim=1)
                pred_final = torch.argmax(rank_logits + args.rank_q_alpha * pred_q, dim=1)

                v_expert += (pred_expert == expert_labels).sum().item()
                v_qbest += (pred_qbest == target_best).sum().item()
                v_final += (pred_final == target_best).sum().item()
                v_samples += B
                v_batches += 1

                if args.val_batches == 0:
                    raise ValueError("Set --val-batches > 0 for IterableDataset.")

        n = max(1, total_samples)
        master_print(
            f"Train | Loss {total_loss/n:.4f} "
            f"(SoftRank {total_soft/n:.4f}, Q {total_q/n:.4f}, "
            f"Atk {total_atk/n:.4f}, Imit {total_imit/n:.4f}) | "
            f"ExpertAcc {100.0*total_expert_acc/n:.2f}% | "
            f"QBestAcc {100.0*total_qbest_acc/n:.2f}% | "
            f"FinalAcc {100.0*total_final_acc/n:.2f}% | "
            f"Samples {total_samples}"
        )
        master_print(
            f"Val   | ExpertAcc {100.0*v_expert/max(1,v_samples):.2f}% | "
            f"QBestAcc {100.0*v_qbest/max(1,v_samples):.2f}% | "
            f"FinalAcc {100.0*v_final/max(1,v_samples):.2f}% | "
            f"Samples {v_samples}"
        )

        ckpt_path = os.path.join(args.save_dir, f"{CHECKPOINT_ARCH}_ep{epoch}.pt")
        ckpt = {
            "model_state_dict": model.state_dict(),
            "arch": "tetrisformer_v4",
            "checkpoint_arch": CHECKPOINT_ARCH,
            "cache_version": CACHE_VERSION,
            "spin_mode": DEFAULT_SPIN_MODE,
            "bootstrap_enabled": False,
            "config": vars(args),
        }
        save_checkpoint(ckpt, ckpt_path)
        master_print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
