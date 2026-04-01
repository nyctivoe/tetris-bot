from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch

from model import (
    _apply_placement_fields_to_current_piece,
    _compute_rollout_reward,
    _encode_queue,
    compute_board_features,
    encode_stats,
)
from rl.model_loader import LoadedModelBundle
from tetrisEngine import KIND_TO_PIECE_ID, PIECE_ID_TO_KIND, Piece, TetrisEngine


SELFPLAY_LOOKAHEAD_DEPTH = 5
SELFPLAY_GAMMA = 0.99
SELFPLAY_REWARD_CONFIG = {
    "attack_w": 1.0,
    "tspin_bonus": 0.75,
    "b2b_bonus": 0.35,
    "height_penalty": 0.05,
    "holes_penalty": 0.08,
    "topout_penalty": 2.5,
    "tslot_ready_bonus": 0.0,
}


def _kind_str(piece_or_kind) -> str:
    if piece_or_kind is None:
        return ""
    if isinstance(piece_or_kind, int):
        return str(PIECE_ID_TO_KIND.get(int(piece_or_kind), ""))
    return str(getattr(piece_or_kind, "kind", piece_or_kind))


def build_root_stats_row(engine: TetrisEngine, move_number: int) -> dict:
    return {
        "incoming_garbage": sum(int(g.get("lines", 0)) for g in (engine.incoming_garbage or [])),
        "combo": int(getattr(engine, "combo", 0)),
        "btb": int(getattr(engine, "b2b_chain", 0)),
        "b2b_chain": int(getattr(engine, "b2b_chain", 0)),
        "move_number": int(move_number),
    }


def build_queue_state(engine: TetrisEngine) -> dict:
    current_piece = getattr(engine, "current_piece", None)
    hold = getattr(engine, "hold", None)
    bag = getattr(engine, "bag", None)
    next_queue = ""
    if bag is not None:
        next_queue = "".join(_kind_str(int(pid)) for pid in bag[:5])
    return {
        "current": _kind_str(current_piece),
        "hold": _kind_str(hold) or None,
        "next_queue": next_queue,
    }


def build_pre_state(engine: TetrisEngine) -> dict:
    return {
        "combo": int(getattr(engine, "combo", 0)),
        "combo_active": bool(getattr(engine, "combo_active", False)),
        "b2b_chain": int(getattr(engine, "b2b_chain", 0)),
        "surge_charge": int(getattr(engine, "surge_charge", 0)),
        "incoming_garbage_total": sum(int(g.get("lines", 0)) for g in (engine.incoming_garbage or [])),
    }


def _encode_queue_state(queue_state: dict) -> np.ndarray:
    return _encode_queue(
        queue_state.get("current"),
        queue_state.get("hold"),
        queue_state.get("next_queue", ""),
    )


def _score_result_boards_for_state(
    model_bundle: LoadedModelBundle,
    *,
    base_board: np.ndarray,
    result_boards: list[np.ndarray],
    queue_seqs: list[np.ndarray],
    stats_rows: list[dict],
    rank_q_alpha: float,
) -> np.ndarray:
    boards = [compute_board_features(base_board, result_board) for result_board in result_boards]
    stats = [encode_stats(stats_row, base_board, result_board) for stats_row, result_board in zip(stats_rows, result_boards)]

    boards_t = torch.from_numpy(np.stack(boards, axis=0).astype(np.float32)).to(model_bundle.device)
    queues_t = torch.from_numpy(np.stack(queue_seqs, axis=0).astype(np.int64)).to(model_bundle.device)
    stats_t = torch.from_numpy(np.stack(stats, axis=0).astype(np.float32)).to(model_bundle.device)

    with torch.no_grad():
        rank_scores, _pred_attack, pred_q = model_bundle.model(boards_t, queues_t, stats_t)
        rank_scores = rank_scores.squeeze(-1)
        pred_q = pred_q.squeeze(-1)
        combined = rank_scores + rank_q_alpha * pred_q

    scores = combined.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Non-finite model scores encountered while scoring candidates.")
    return scores


def score_bfs_candidates(
    model_bundle: LoadedModelBundle,
    engine: TetrisEngine,
    bfs_results: List[dict],
    move_number: int,
    rank_q_alpha: float | None = None,
) -> np.ndarray:
    if not bfs_results:
        return np.zeros((0,), dtype=np.float32)

    base_board = engine.board.astype(np.uint8, copy=False)
    queue_state = build_queue_state(engine)
    stats_row = build_root_stats_row(engine, move_number=move_number)
    queue_seq = _encode_queue_state(queue_state)

    alpha = float(model_bundle.rank_q_alpha if rank_q_alpha is None else rank_q_alpha)
    result_boards = []
    for result in bfs_results:
        result_board = result.get("board")
        if result_board is None:
            raise RuntimeError("BFS result is missing board.")
        result_boards.append(result_board)

    return _score_result_boards_for_state(
        model_bundle,
        base_board=base_board,
        result_boards=result_boards,
        queue_seqs=[queue_seq for _ in result_boards],
        stats_rows=[stats_row for _ in result_boards],
        rank_q_alpha=alpha,
    )


def select_top_k_candidates(
    bfs_results: List[dict],
    scores: np.ndarray,
    top_k: int,
) -> tuple[list[int], np.ndarray]:
    if len(bfs_results) != int(scores.shape[0]):
        raise ValueError("bfs_results and scores length mismatch.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if len(bfs_results) == 0:
        return [], np.zeros((0,), dtype=np.float32)

    order = np.argsort(scores)[::-1]
    keep = order[: min(int(top_k), len(order))]
    indices = [int(i) for i in keep.tolist()]
    retained_scores = scores[keep].astype(np.float32, copy=True)
    return indices, retained_scores


def soft_policy_from_scores(scores: np.ndarray, temperature: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array.")
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float32)

    temp = float(temperature)
    if temp <= 1e-6:
        out = np.zeros_like(scores, dtype=np.float32)
        out[int(np.argmax(scores))] = 1.0
        return out

    z = scores / temp
    z = z - np.max(z)
    exp_z = np.exp(z, dtype=np.float32)
    denom = float(exp_z.sum())
    if not np.isfinite(denom) or denom <= 0.0:
        raise RuntimeError("Failed to normalize policy target from candidate scores.")
    return (exp_z / denom).astype(np.float32, copy=False)


def _spawn_kind(engine: TetrisEngine, kind: str, allow_clutch: bool = True) -> bool:
    spawn_pos = engine._spawn_position_for(kind)
    piece = Piece(kind, rotation=0, position=spawn_pos)
    engine.last_spawn_was_clutch = False
    if engine.is_position_valid(piece, spawn_pos, rotation=0):
        engine.current_piece = piece
        return True
    if allow_clutch:
        clutch_pos = engine._find_clutch_spawn(piece, spawn_pos)
        if clutch_pos is not None:
            piece.position = clutch_pos
            engine.current_piece = piece
            engine.last_spawn_was_clutch = True
            return True
    engine.current_piece = None
    engine.game_over = True
    engine.game_over_reason = "block_out"
    return False


def _perform_hold(engine: TetrisEngine) -> bool:
    if engine.current_piece is None or engine.game_over:
        return False
    cur_kind = _kind_str(engine.current_piece)
    if not cur_kind:
        return False

    hold_kind = _kind_str(getattr(engine, "hold", None))
    engine.hold = KIND_TO_PIECE_ID.get(cur_kind, KIND_TO_PIECE_ID.get(str(cur_kind), 1))
    engine.current_piece = None

    if hold_kind:
        return _spawn_kind(engine, hold_kind, allow_clutch=True)
    return bool(engine.spawn_next(allow_clutch=True))


def _simulate_candidate(
    action_engine: TetrisEngine,
    placement: dict,
    *,
    used_hold: bool,
) -> dict | None:
    sim = copy.deepcopy(action_engine)
    if sim.current_piece is None:
        return None

    prev_b2b = int(getattr(sim, "b2b_chain", 0))
    queue_state = build_queue_state(sim)
    pre_state = build_pre_state(sim)
    placed_kind = queue_state.get("current") or _kind_str(sim.current_piece)

    if not _apply_placement_fields_to_current_piece(sim, placement):
        return None

    sim.lock_and_spawn()
    clear_stats = sim.last_clear_stats or {}
    reward, attack = _compute_rollout_reward(
        sim.board,
        clear_stats,
        prev_b2b,
        bool(sim.game_over),
        SELFPLAY_REWARD_CONFIG["attack_w"],
        SELFPLAY_REWARD_CONFIG["tspin_bonus"],
        SELFPLAY_REWARD_CONFIG["b2b_bonus"],
        SELFPLAY_REWARD_CONFIG["height_penalty"],
        SELFPLAY_REWARD_CONFIG["holes_penalty"],
        SELFPLAY_REWARD_CONFIG["topout_penalty"],
        SELFPLAY_REWARD_CONFIG["tslot_ready_bonus"],
    )

    placement_payload = dict(placement)
    placement_payload["used_hold"] = bool(used_hold)
    placement_payload["placed_kind"] = placed_kind

    return {
        "board": sim.board.astype(np.uint8, copy=True),
        "stats": clear_stats,
        "placement": placement_payload,
        "queue_state": queue_state,
        "pre_state": pre_state,
        "engine_after": sim,
        "immediate_reward": float(reward),
        "immediate_attack": float(attack),
        "used_hold": bool(used_hold),
    }


def _advance_pending_garbage(engine: TetrisEngine) -> None:
    if not getattr(engine, "incoming_garbage", None):
        return
    engine.tick_garbage()
    piece = getattr(engine, "current_piece", None)
    if piece is None:
        return
    try:
        valid = bool(engine.is_position_valid(piece, piece.position, piece.rotation))
    except TypeError:
        valid = bool(engine.is_position_valid(piece, position=piece.position))
    if not valid:
        engine.current_piece = None
        engine.game_over = True
        engine.game_over_reason = "garbage_top_out"


def generate_action_candidates(engine: TetrisEngine, include_hold: bool = True) -> list[dict]:
    candidates: list[dict] = []
    if engine.current_piece is None or engine.game_over:
        return candidates

    variants: list[tuple[TetrisEngine, bool]] = [(copy.deepcopy(engine), False)]
    if include_hold:
        hold_engine = copy.deepcopy(engine)
        if _perform_hold(hold_engine):
            variants.append((hold_engine, True))

    for action_engine, used_hold in variants:
        if action_engine.current_piece is None or action_engine.game_over:
            continue
        try:
            bfs_results = action_engine.bfs_all_placements(include_no_place=False, dedupe_final=True)
        except Exception:
            bfs_results = []
        for result in bfs_results:
            placement = result.get("placement")
            if not placement:
                continue
            candidate = _simulate_candidate(action_engine, placement, used_hold=used_hold)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def score_action_candidates(
    model_bundle: LoadedModelBundle,
    engine: TetrisEngine,
    candidates: list[dict],
    move_number: int,
    rank_q_alpha: float | None = None,
) -> np.ndarray:
    if not candidates:
        return np.zeros((0,), dtype=np.float32)

    alpha = float(model_bundle.rank_q_alpha if rank_q_alpha is None else rank_q_alpha)
    base_board = engine.board.astype(np.uint8, copy=False)
    stats_row = build_root_stats_row(engine, move_number=move_number)
    result_boards = [c["board"] for c in candidates]
    queue_seqs = [_encode_queue_state(c["queue_state"]) for c in candidates]
    stats_rows = [stats_row for _ in candidates]

    return _score_result_boards_for_state(
        model_bundle,
        base_board=base_board,
        result_boards=result_boards,
        queue_seqs=queue_seqs,
        stats_rows=stats_rows,
        rank_q_alpha=alpha,
    )


def _future_beam_score(
    model_bundle: LoadedModelBundle,
    engine_after: TetrisEngine,
    move_number: int,
    beam_width: int,
    rank_q_alpha: float,
    depth_remaining: int,
    gamma: float,
) -> float:
    if depth_remaining <= 0 or engine_after.game_over or engine_after.current_piece is None:
        return 0.0

    beam = [(0.0, copy.deepcopy(engine_after), int(move_number))]
    best = 0.0

    for depth in range(int(depth_remaining)):
        expanded: list[tuple[float, TetrisEngine, int]] = []
        for acc_score, node_engine, node_move in beam:
            _advance_pending_garbage(node_engine)
            if node_engine.game_over or node_engine.current_piece is None:
                best = max(best, float(acc_score))
                continue
            candidates = generate_action_candidates(node_engine, include_hold=True)
            if not candidates:
                best = max(best, float(acc_score))
                continue

            model_scores = score_action_candidates(
                model_bundle,
                node_engine,
                candidates,
                move_number=node_move,
                rank_q_alpha=rank_q_alpha,
            )
            step_scores = model_scores + np.asarray(
                [float(c["immediate_reward"]) for c in candidates],
                dtype=np.float32,
            )
            order = np.argsort(step_scores)[::-1][: max(1, int(beam_width))]
            for idx in order.tolist():
                candidate = candidates[int(idx)]
                child_score = float(acc_score) + (float(gamma) ** depth) * float(step_scores[int(idx)])
                expanded.append((child_score, copy.deepcopy(candidate["engine_after"]), node_move + 1))

        if not expanded:
            break

        expanded.sort(key=lambda item: item[0], reverse=True)
        beam = expanded[: max(1, int(beam_width))]
        best = max(best, float(beam[0][0]))

    if beam:
        best = max(best, max(float(node[0]) for node in beam))
    return float(best)


def score_action_candidates_with_lookahead(
    model_bundle: LoadedModelBundle,
    engine: TetrisEngine,
    candidates: list[dict],
    move_number: int,
    beam_width: int,
    rank_q_alpha: float | None = None,
    gamma: float = SELFPLAY_GAMMA,
    lookahead_depth: int = SELFPLAY_LOOKAHEAD_DEPTH,
) -> np.ndarray:
    if not candidates:
        return np.zeros((0,), dtype=np.float32)

    alpha = float(model_bundle.rank_q_alpha if rank_q_alpha is None else rank_q_alpha)
    root_model_scores = score_action_candidates(
        model_bundle,
        engine,
        candidates,
        move_number=move_number,
        rank_q_alpha=alpha,
    )
    immediate_rewards = np.asarray(
        [float(candidate["immediate_reward"]) for candidate in candidates],
        dtype=np.float32,
    )
    adjusted = root_model_scores + immediate_rewards
    if beam_width <= 1 or lookahead_depth <= 1:
        return adjusted.astype(np.float32, copy=False)

    for idx, candidate in enumerate(candidates):
        future_score = _future_beam_score(
            model_bundle,
            candidate["engine_after"],
            move_number=move_number + 1,
            beam_width=beam_width,
            rank_q_alpha=alpha,
            depth_remaining=lookahead_depth - 1,
            gamma=gamma,
        )
        adjusted[idx] = float(adjusted[idx]) + float(gamma) * float(future_score)

    return adjusted.astype(np.float32, copy=False)
