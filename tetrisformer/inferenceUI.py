#!/usr/bin/env python3
"""
TetrisFormer Inference UI
=========================
A pygame-based Tetris game with manual controls and AI model inference.

Controls:
  ← →       Move piece left/right
  ↓          Soft drop
  Space      Hard drop
  ↑ / X      Rotate CW
  Z          Rotate CCW
  A          Rotate 180
  C / LShift Hold piece
  M          Model makes one move (animated)
  S          Toggle top-3 model suggestions
  N          Toggle continuous auto-play
  R          Reset game
  Esc        Quit
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tetrisEngine import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    HIDDEN_ROWS,
    VISIBLE_HEIGHT,
    SPAWN_X,
    SPAWN_Y,
    PIECE_ID_TO_KIND,
    KIND_TO_PIECE_ID,
    PIECE_DEFS,
    PIECE_ROTATIONS,
    TetrisEngine,
    Piece,
)
from tetrisformer.model import (
    compute_board_features,
    encode_stats,
    _encode_queue,
    get_device,
)
from tetrisformer.rl.model_loader import load_tetrisformer_checkpoint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CELL = 32  # pixel size of one board cell
BOARD_PX_W = BOARD_WIDTH * CELL
BOARD_PX_H = VISIBLE_HEIGHT * CELL

SIDEBAR_W = 220
MARGIN = 16
TOP_BAR_H = 40

WINDOW_W = MARGIN + BOARD_PX_W + MARGIN + SIDEBAR_W + MARGIN
WINDOW_H = TOP_BAR_H + BOARD_PX_H + MARGIN

FPS = 60

# Colours
COL_BG = (18, 18, 30)
COL_GRID = (40, 40, 60)
COL_TEXT = (220, 220, 240)
COL_DIM_TEXT = (140, 140, 160)
COL_GHOST = (100, 100, 120)
COL_GHOST_BEST = (0, 255, 128)
COL_GHOST_2ND = (255, 200, 0)
COL_GHOST_3RD = (255, 100, 50)
COL_HOLD_BORDER = (80, 80, 120)
COL_GARBAGE = (90, 90, 90)

PIECE_COLORS = {
    1: (0, 240, 240),    # I - cyan
    2: (240, 240, 0),    # O - yellow
    3: (160, 0, 240),    # T - purple
    4: (0, 240, 0),      # S - green
    5: (240, 0, 0),      # Z - red
    6: (0, 0, 240),      # J - blue
    7: (240, 160, 0),    # L - orange
    8: COL_GARBAGE,      # garbage
}

GHOST_COLORS = [COL_GHOST_BEST, COL_GHOST_2ND, COL_GHOST_3RD]

# ---------------------------------------------------------------------------
# Model inference helpers
# ---------------------------------------------------------------------------

import torch


def _score_placements(
    model,
    device,
    base_board: np.ndarray,
    result_boards: List[np.ndarray],
    queue_seq: np.ndarray,
    stats_row: dict,
    rank_q_alpha: float,
) -> np.ndarray:
    """Score a batch of result boards and return combined scores as numpy array."""
    boards_list = []
    stats_list = []
    for rb in result_boards:
        boards_list.append(compute_board_features(base_board, rb))
        stats_list.append(encode_stats(stats_row, base_board, rb))

    boards_t = torch.from_numpy(np.stack(boards_list, axis=0).astype(np.float32)).to(device)
    n = len(result_boards)
    queues_t = torch.from_numpy(
        np.tile(queue_seq[np.newaxis, :], (n, 1)).astype(np.int64)
    ).to(device)
    stats_t = torch.from_numpy(np.stack(stats_list, axis=0).astype(np.float32)).to(device)

    with torch.no_grad():
        rank_scores, pred_attack, pred_q = model(boards_t, queues_t, stats_t)
        rank_scores = rank_scores.squeeze(-1)
        pred_q = pred_q.squeeze(-1)
        combined = rank_scores + rank_q_alpha * pred_q

    return combined.cpu().numpy()


def _build_queue_seq(engine: TetrisEngine, placed_kind: str, bag_offset: int = 0):
    """Build queue encoding for the model, optionally skipping bag_offset pieces."""
    hold_kind = engine.hold
    if isinstance(hold_kind, int):
        hold_kind = PIECE_ID_TO_KIND.get(hold_kind)
    bag = engine.bag if engine.bag is not None else []
    next_str = "".join(PIECE_ID_TO_KIND.get(int(x), "") for x in bag[bag_offset:bag_offset + 5])
    return _encode_queue(placed_kind, hold_kind, next_str)


def _bfs_on_board(engine: TetrisEngine, board: np.ndarray, piece_kind: str):
    """Run BFS placements for a given piece on an arbitrary board state.

    Temporarily swaps the engine's board and piece, then restores them.
    """
    saved_board = engine.board
    saved_piece = engine.current_piece
    engine.board = board
    engine.current_piece = Piece(piece_kind, rotation=0, position=(SPAWN_X, SPAWN_Y))
    results = engine.bfs_all_placements(include_no_place=False, dedupe_final=True)
    engine.board = saved_board
    engine.current_piece = saved_piece
    return results


def beam_search_suggestions(
    engine: TetrisEngine,
    model,
    device,
    rank_q_alpha: float = 0.3,
    beam_width: int = 3,
    top_k: int = 3,
) -> List[Tuple[dict, float, np.ndarray]]:
    """Beam search over the full visible bag depth.

    Returns top_k first-move suggestions ranked by best accumulated score
    across all lookahead depths.
    """
    if engine.current_piece is None:
        return []

    model.eval()

    bag = engine.bag if engine.bag is not None else []
    # Pieces to search: current + all visible bag pieces
    piece_kinds = [str(engine.current_piece.kind)]
    for pid in bag[:5]:
        kind = PIECE_ID_TO_KIND.get(int(pid))
        if kind:
            piece_kinds.append(kind)

    depth = len(piece_kinds)
    if depth == 0:
        return []

    stats_row = {
        "b2b_chain": engine.b2b_chain,
        "btb": engine.b2b_chain,
        "combo": engine.combo,
        "incoming_garbage": sum(g["lines"] for g in (engine.incoming_garbage or [])),
        "move_number": 0,
    }

    # Each beam entry: (accumulated_score, board, first_placement, first_board)
    # Depth 0: enumerate placements of current piece
    base_board = engine.board.copy()
    results_d0 = engine.bfs_all_placements(include_no_place=False, dedupe_final=True)
    if not results_d0:
        return []

    queue_seq = _build_queue_seq(engine, piece_kinds[0], bag_offset=0)
    result_boards = [r["board"] for r in results_d0]
    scores = _score_placements(
        model, device, base_board, result_boards, queue_seq, stats_row, rank_q_alpha,
    )

    # Build initial beam from depth-0 results
    top_idx = np.argsort(scores)[::-1][:beam_width]
    beam = []
    for idx in top_idx:
        beam.append((
            float(scores[idx]),
            results_d0[idx]["board"],
            results_d0[idx]["placement"],
            results_d0[idx]["board"],
        ))

    # Expand beam for remaining depths
    for d in range(1, depth):
        kind = piece_kinds[d]
        candidates = []

        for acc_score, board, first_placement, first_board in beam:
            bfs_results = _bfs_on_board(engine, board, kind)
            if not bfs_results:
                # Dead end — carry forward with a penalty
                candidates.append((acc_score - 10.0, board, first_placement, first_board))
                continue

            queue_seq = _build_queue_seq(engine, kind, bag_offset=d)
            child_boards = [r["board"] for r in bfs_results]
            child_scores = _score_placements(
                model, device, board, child_boards, queue_seq, stats_row, rank_q_alpha,
            )

            # Keep only the best child for each beam entry to limit explosion
            best_child_idx = int(np.argmax(child_scores))
            candidates.append((
                acc_score + float(child_scores[best_child_idx]),
                bfs_results[best_child_idx]["board"],
                first_placement,
                first_board,
            ))

        # Prune to beam_width
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]

    # Collect top_k unique first-move suggestions
    beam.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    suggestions = []
    for acc_score, _, first_placement, first_board in beam:
        key = (first_placement["x"], first_placement["y"], first_placement["rotation"])
        if key in seen:
            continue
        seen.add(key)
        suggestions.append((first_placement, acc_score, first_board))
        if len(suggestions) >= top_k:
            break

    return suggestions


def get_model_suggestions(
    engine: TetrisEngine,
    model,
    device,
    rank_q_alpha: float = 0.3,
    top_k: int = 3,
) -> List[Tuple[dict, float]]:
    """Run beam search over full bag depth and return the top-k first-move placements."""
    return beam_search_suggestions(
        engine, model, device,
        rank_q_alpha=rank_q_alpha,
        beam_width=max(top_k, 3),
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Piece drawing helper
# ---------------------------------------------------------------------------

def draw_piece_blocks(surface, kind, blocks, offset_x, offset_y, cell_size, color=None, alpha=255):
    """Draw piece blocks at given pixel offset."""
    pid = KIND_TO_PIECE_ID.get(kind, 0) if isinstance(kind, str) else kind
    col = color or PIECE_COLORS.get(pid, (200, 200, 200))
    for bx, by in blocks:
        rect = pygame.Rect(offset_x + bx * cell_size, offset_y + by * cell_size, cell_size - 1, cell_size - 1)
        if alpha < 255:
            s = pygame.Surface((cell_size - 1, cell_size - 1), pygame.SRCALPHA)
            s.fill((*col, alpha))
            surface.blit(s, rect.topleft)
        else:
            pygame.draw.rect(surface, col, rect)
            # Highlight edge
            highlight = tuple(min(255, c + 40) for c in col)
            pygame.draw.rect(surface, highlight, rect, 1)


def get_piece_blocks(kind, rotation=0):
    """Get relative block positions for a piece kind at given rotation."""
    return PIECE_ROTATIONS[kind][rotation]


def draw_mini_piece(surface, kind, cx, cy, cell_size=18):
    """Draw a small piece centered at (cx, cy)."""
    blocks = PIECE_DEFS[kind]["blocks"]
    size = PIECE_DEFS[kind]["size"]
    pid = KIND_TO_PIECE_ID.get(kind, 0)
    col = PIECE_COLORS.get(pid, (200, 200, 200))
    # Center the piece
    for bx, by in blocks:
        px = cx + (bx - size / 2 + 0.5) * cell_size
        py = cy + (by - size / 2 + 0.5) * cell_size
        rect = pygame.Rect(int(px), int(py), cell_size - 1, cell_size - 1)
        pygame.draw.rect(surface, col, rect)
        highlight = tuple(min(255, c + 40) for c in col)
        pygame.draw.rect(surface, highlight, rect, 1)


# ---------------------------------------------------------------------------
# Animation state
# ---------------------------------------------------------------------------

class AnimationState:
    """Tracks an ongoing piece animation from current position to target."""

    def __init__(self):
        self.active = False
        self.start_x = 0
        self.start_y = 0
        self.start_rot = 0
        self.target_x = 0
        self.target_y = 0
        self.target_rot = 0
        self.progress = 0.0  # 0..1
        self.speed = 4.0  # complete in ~0.25s
        self.on_complete = None  # callback

    def start(self, piece, target_placement, on_complete=None):
        self.active = True
        self.start_x = piece.position[0]
        self.start_y = piece.position[1]
        self.start_rot = piece.rotation
        self.target_x = int(target_placement["x"])
        self.target_y = int(target_placement["y"])
        self.target_rot = int(target_placement["rotation"])
        self.progress = 0.0
        self.on_complete = on_complete

    def update(self, dt):
        if not self.active:
            return
        self.progress += self.speed * dt
        if self.progress >= 1.0:
            self.progress = 1.0
            self.active = False
            if self.on_complete:
                self.on_complete()

    def get_current_state(self):
        """Returns interpolated (x, y, rotation)."""
        t = self.progress
        # Smooth ease
        t = t * t * (3 - 2 * t)
        x = int(round(self.start_x + (self.target_x - self.start_x) * t))
        y = int(round(self.start_y + (self.target_y - self.start_y) * t))
        # For rotation, snap to target once past halfway
        rot = self.start_rot if t < 0.5 else self.target_rot
        return x, y, rot


# ---------------------------------------------------------------------------
# Main Game class
# ---------------------------------------------------------------------------

class TetrisInferenceUI:
    def __init__(self, model_path: Optional[str] = None, rank_q_alpha: float = 0.3):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("TetrisFormer Inference UI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16)
        self.font_big = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 13)

        self.device = get_device()
        self.model = None
        self.model_path = model_path
        self.rank_q_alpha = rank_q_alpha
        self.auto_play = False
        self.auto_play_delay = 0.3  # seconds between auto-play moves
        self.auto_play_timer = 0.0

        # Suggestions overlay (toggled via S key)
        self.suggestions: List[Tuple[dict, float, np.ndarray]] = []
        self.show_suggestions = False

        # Animation
        self.animation = AnimationState()

        # Game state
        self.engine = TetrisEngine()
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.last_clear_text = ""
        self.last_clear_timer = 0.0

        # Board origin
        self.board_ox = MARGIN
        self.board_oy = TOP_BAR_H

        # Load model if provided
        if model_path:
            self._load_model(model_path)

        # Spawn first piece
        self._spawn_next()

    def _load_model(self, path: str):
        """Load a TetrisFormer checkpoint."""
        print(f"Loading model from {path} ...")
        bundle = load_tetrisformer_checkpoint(path, device=self.device, eval_mode=True)
        if bundle.spin_mode != "all_spin":
            print(
                f"[warn] checkpoint spin_mode={bundle.spin_mode!r}; modern all-spin analysis expects 'all_spin'."
            )
        self.model = bundle.model
        self.model_path = path
        print(f"Model loaded successfully (device: {self.device})")

    def _spawn_next(self):
        """Spawn the next piece after locking."""
        if self.engine.current_piece is None and not self.engine.game_over:
            self.engine.spawn_next(allow_clutch=True)
            if self.engine.game_over:
                self.game_over = True
            self.suggestions = []
            self.show_suggestions = False

    def _lock_current(self):
        """Lock the current piece and spawn next."""
        if self.engine.current_piece is None:
            return

        piece = self.engine.current_piece
        stats = self.engine.last_clear_stats

        cleared, end_phase = self.engine.lock_and_spawn()

        if stats and stats.get("attack", 0) > 0:
            atk = stats["attack"]
            self.last_clear_text = f"Attack: {atk}"
            if stats.get("lines_cleared", 0) > 0:
                self.last_clear_text = f"{stats['lines_cleared']} lines | Atk: {atk}"
            self.last_clear_timer = 2.0

        self.lines_cleared += cleared
        self.pieces_placed += 1
        self.score += cleared * 100

        if self.engine.game_over:
            self.game_over = True
        else:
            self._spawn_next()

    def _do_hold(self):
        """Hold the current piece."""
        if self.engine.current_piece is None:
            return
        cur = self.engine.current_piece
        cur_kind = cur.kind

        if self.engine.hold is not None:
            hold_kind = self.engine.hold
            if isinstance(hold_kind, int):
                hold_kind = PIECE_ID_TO_KIND.get(hold_kind)
            self.engine.hold = KIND_TO_PIECE_ID.get(cur_kind, KIND_TO_PIECE_ID.get(str(cur_kind), 1))
            self.engine.current_piece = None
            self.engine.spawn_piece(hold_kind)
        else:
            self.engine.hold = KIND_TO_PIECE_ID.get(cur_kind, KIND_TO_PIECE_ID.get(str(cur_kind), 1))
            self.engine.current_piece = None
            self._spawn_next()

        self.suggestions = []
        self.show_suggestions = False

    def _get_ghost_y(self, piece=None):
        """Get the Y position where the piece would land (hard drop)."""
        if piece is None:
            piece = self.engine.current_piece
        if piece is None:
            return 0
        x, y = piece.position
        while self.engine.is_position_valid(piece, (x, y + 1)):
            y += 1
        return y

    def _toggle_suggestions(self):
        """Toggle display of top-3 model suggestions for the current piece."""
        if self.model is None or self.engine.current_piece is None:
            return
        if self.show_suggestions:
            self.show_suggestions = False
            return
        # Compute fresh suggestions
        suggestions = get_model_suggestions(
            self.engine, self.model, self.device,
            rank_q_alpha=self.rank_q_alpha, top_k=3,
        )
        if suggestions:
            self.suggestions = suggestions
            self.show_suggestions = True

    def _model_move(self):
        """Ask the model for the best move and animate to it."""
        if self.model is None or self.engine.current_piece is None or self.animation.active:
            return

        suggestions = get_model_suggestions(
            self.engine, self.model, self.device,
            rank_q_alpha=self.rank_q_alpha, top_k=3,
        )

        if not suggestions:
            return

        self.suggestions = suggestions

        best_placement = suggestions[0][0]

        # Start animation to best placement
        def on_anim_complete():
            # Apply the placement
            piece = self.engine.current_piece
            if piece is None:
                return
            piece.position = (int(best_placement["x"]), int(best_placement["y"]))
            piece.rotation = int(best_placement["rotation"])
            piece.last_action_was_rotation = bool(best_placement.get("last_was_rot", False))
            last_dir = best_placement.get("last_rot_dir")
            piece.last_rotation_dir = None if last_dir in (None, 0) else int(last_dir)
            last_kick = best_placement.get("last_kick_idx")
            piece.last_kick_index = None if last_kick is None or int(last_kick) < 0 else int(last_kick)
            self._lock_current()

        self.animation.start(self.engine.current_piece, best_placement, on_anim_complete)

    def _reset_game(self):
        """Reset the game state."""
        self.engine.reset()
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.last_clear_text = ""
        self.last_clear_timer = 0.0
        self.suggestions = []
        self.show_suggestions = False
        self.animation.active = False
        self.auto_play = False
        self._spawn_next()

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------

    def _draw_board(self):
        """Draw the Tetris board."""
        board = self.engine.board
        ox, oy = self.board_ox, self.board_oy

        # Draw grid
        for y in range(VISIBLE_HEIGHT):
            for x in range(BOARD_WIDTH):
                by = y + HIDDEN_ROWS
                cell_val = board[by, x]
                rect = pygame.Rect(ox + x * CELL, oy + y * CELL, CELL - 1, CELL - 1)

                if cell_val != 0:
                    col = PIECE_COLORS.get(int(cell_val), (200, 200, 200))
                    pygame.draw.rect(self.screen, col, rect)
                    # Highlight
                    highlight = tuple(min(255, c + 50) for c in col)
                    pygame.draw.rect(self.screen, highlight, rect, 1)
                else:
                    pygame.draw.rect(self.screen, COL_GRID, rect, 1)

        # Draw ghost piece (hard drop preview)
        piece = self.engine.current_piece
        if piece and not self.animation.active:
            ghost_y = self._get_ghost_y(piece)
            if ghost_y != piece.position[1]:
                blocks = self.piece_blocks(piece)
                pid = KIND_TO_PIECE_ID.get(piece.kind, 0) if isinstance(piece.kind, str) else piece.kind
                col = PIECE_COLORS.get(pid, (200, 200, 200))
                for bx, by in blocks:
                    gy = ghost_y + by - piece.position[1]
                    vis_y = gy - HIDDEN_ROWS
                    if 0 <= vis_y < VISIBLE_HEIGHT:
                        rect = pygame.Rect(ox + bx * CELL, oy + vis_y * CELL, CELL - 1, CELL - 1)
                        pygame.draw.rect(self.screen, col, rect, 2)

        # Draw current piece (or animated piece)
        if piece:
            if self.animation.active:
                ax, ay, arot = self.animation.get_current_state()
                blocks = PIECE_ROTATIONS[piece.kind][arot]
                pid = KIND_TO_PIECE_ID.get(piece.kind, 0) if isinstance(piece.kind, str) else piece.kind
                col = PIECE_COLORS.get(pid, (200, 200, 200))
                for bx, by in blocks:
                    vis_y = (ay + by) - HIDDEN_ROWS
                    if 0 <= vis_y < VISIBLE_HEIGHT and 0 <= ax + bx < BOARD_WIDTH:
                        rect = pygame.Rect(ox + (ax + bx) * CELL, oy + vis_y * CELL, CELL - 1, CELL - 1)
                        pygame.draw.rect(self.screen, col, rect)
                        highlight = tuple(min(255, c + 50) for c in col)
                        pygame.draw.rect(self.screen, highlight, rect, 1)
            else:
                blocks = self.piece_blocks(piece)
                pid = KIND_TO_PIECE_ID.get(piece.kind, 0) if isinstance(piece.kind, str) else piece.kind
                col = PIECE_COLORS.get(pid, (200, 200, 200))
                for bx, by in blocks:
                    vis_y = by - HIDDEN_ROWS
                    if 0 <= vis_y < VISIBLE_HEIGHT and 0 <= bx < BOARD_WIDTH:
                        rect = pygame.Rect(
                            ox + bx * CELL,
                            oy + vis_y * CELL,
                            CELL - 1, CELL - 1,
                        )
                        pygame.draw.rect(self.screen, col, rect)
                        highlight = tuple(min(255, c + 50) for c in col)
                        pygame.draw.rect(self.screen, highlight, rect, 1)

        # Draw suggestion ghosts (only when toggled via S key)
        if self.show_suggestions and self.suggestions:
            for rank, (placement, score, result_board) in enumerate(self.suggestions):
                color = GHOST_COLORS[min(rank, len(GHOST_COLORS) - 1)]
                px = int(placement["x"])
                py = int(placement["y"])
                prot = int(placement["rotation"])
                kind = placement.get("kind", piece.kind if piece else "T")
                blocks = PIECE_ROTATIONS[kind][prot]
                alpha = max(40, 120 - rank * 30)
                for bx, by in blocks:
                    vis_y = (py + by) - HIDDEN_ROWS
                    if 0 <= vis_y < VISIBLE_HEIGHT and 0 <= px + bx < BOARD_WIDTH:
                        s = pygame.Surface((CELL - 1, CELL - 1), pygame.SRCALPHA)
                        s.fill((*color, alpha))
                        self.screen.blit(s, (ox + (px + bx) * CELL, oy + vis_y * CELL))

                # Score label
                label_x = ox + px * CELL
                label_y = oy + (py - HIDDEN_ROWS) * CELL - 16
                if 0 <= py - HIDDEN_ROWS < VISIBLE_HEIGHT:
                    label = self.font_small.render(f"#{rank+1} {score:.2f}", True, color)
                    self.screen.blit(label, (label_x, max(TOP_BAR_H, label_y)))

        # Board border
        pygame.draw.rect(self.screen, COL_HOLD_BORDER,
                         (ox - 2, oy - 2, BOARD_PX_W + 4, BOARD_PX_H + 4), 2)

    def piece_blocks(self, piece):
        """Get absolute block positions for a piece."""
        return self.engine.piece_blocks(piece)

    def _draw_sidebar(self):
        """Draw hold, next queue, stats, controls, and model info."""
        sx = MARGIN + BOARD_PX_W + MARGIN
        y = TOP_BAR_H

        # --- Hold piece ---
        label = self.font.render("HOLD", True, COL_TEXT)
        self.screen.blit(label, (sx, y))
        y += 22
        hold_rect = pygame.Rect(sx, y, 100, 52)
        pygame.draw.rect(self.screen, COL_HOLD_BORDER, hold_rect, 2)
        if self.engine.hold is not None:
            hold_kind = self.engine.hold
            if isinstance(hold_kind, int):
                hold_kind = PIECE_ID_TO_KIND.get(hold_kind)
            if hold_kind:
                draw_mini_piece(self.screen, hold_kind, sx + 50, y + 26, cell_size=16)
        y += 60

        # --- Next pieces ---
        label = self.font.render("NEXT", True, COL_TEXT)
        self.screen.blit(label, (sx, y))
        y += 22
        bag = self.engine.bag if self.engine.bag is not None else []
        for i in range(5):
            if i < len(bag):
                pid = int(bag[i])
                kind = PIECE_ID_TO_KIND.get(pid)
                if kind:
                    draw_mini_piece(self.screen, kind, sx + 50, y + 16, cell_size=14)
            y += 36
        y += 4

        # --- Stats ---
        stats = [
            f"Score:  {self.score}",
            f"Lines:  {self.lines_cleared}",
            f"Pieces: {self.pieces_placed}",
            f"Combo:  {self.engine.combo}",
            f"B2B:    {max(0, self.engine.b2b_chain - 1)}",
        ]
        for text in stats:
            surf = self.font_small.render(text, True, COL_DIM_TEXT)
            self.screen.blit(surf, (sx, y))
            y += 18
        y += 4

        # --- Last clear ---
        if self.last_clear_timer > 0 and self.last_clear_text:
            surf = self.font.render(self.last_clear_text, True, (255, 255, 100))
            self.screen.blit(surf, (sx, y))
        y += 22

        # --- Model info ---
        if self.model:
            name = os.path.basename(self.model_path) if self.model_path else "Unknown"
            surf = self.font_small.render(f"Model: {name[:25]}", True, COL_DIM_TEXT)
            self.screen.blit(surf, (sx, y))
            y += 16
            if self.auto_play:
                surf = self.font.render("AUTO-PLAY ON", True, (0, 255, 128))
                self.screen.blit(surf, (sx, y))
            y += 20
        else:
            surf = self.font_small.render("No model loaded", True, (180, 80, 80))
            self.screen.blit(surf, (sx, y))
            y += 16
            surf = self.font_small.render("Use --model path.pt", True, COL_DIM_TEXT)
            self.screen.blit(surf, (sx, y))
            y += 20

        # --- Controls (anchored to bottom) ---
        controls = [
            "Controls:",
            "Arrows  Move/Drop",
            "Space   Hard drop",
            "X/Z/A   CW/CCW/180",
            "C/LShft Hold",
            "M       Model move",
            "S       Suggestions",
            "N       Auto-play",
            "R       Reset",
            "Esc     Quit",
        ]
        cy = WINDOW_H - MARGIN - len(controls) * 16
        for i, text in enumerate(controls):
            col = COL_TEXT if i == 0 else COL_DIM_TEXT
            surf = self.font_small.render(text, True, col)
            self.screen.blit(surf, (sx, cy + i * 16))

    def _draw_top_bar(self):
        """Draw the top bar with title."""
        title = self.font_big.render("TetrisFormer", True, COL_TEXT)
        self.screen.blit(title, (MARGIN, 8))

        if self.game_over:
            go_text = self.font_big.render("GAME OVER", True, (255, 60, 60))
            self.screen.blit(go_text, (MARGIN + 200, 8))

    def _draw(self):
        """Main draw call."""
        self.screen.fill(COL_BG)
        self._draw_top_bar()
        self._draw_board()
        self._draw_sidebar()
        pygame.display.flip()

    # -----------------------------------------------------------------------
    # Input handling
    # -----------------------------------------------------------------------

    def _handle_input(self, event):
        """Handle a single key event."""
        if event.type != pygame.KEYDOWN:
            return

        key = event.key

        if key == pygame.K_ESCAPE:
            return "quit"

        if key == pygame.K_r:
            self._reset_game()
            return

        if self.game_over or self.animation.active:
            return

        piece = self.engine.current_piece
        if piece is None:
            return

        # Movement
        if key == pygame.K_LEFT:
            x, y = piece.position
            if self.engine.is_position_valid(piece, (x - 1, y)):
                piece.position = (x - 1, y)
                piece.last_action_was_rotation = False
        elif key == pygame.K_RIGHT:
            x, y = piece.position
            if self.engine.is_position_valid(piece, (x + 1, y)):
                piece.position = (x + 1, y)
                piece.last_action_was_rotation = False
        elif key == pygame.K_DOWN:
            x, y = piece.position
            if self.engine.is_position_valid(piece, (x, y + 1)):
                piece.position = (x, y + 1)
                self.score += 1
                piece.last_action_was_rotation = False
        elif key == pygame.K_SPACE:
            # Hard drop
            ghost_y = self._get_ghost_y(piece)
            drop_dist = ghost_y - piece.position[1]
            piece.position = (piece.position[0], ghost_y)
            self.score += drop_dist * 2
            self._lock_current()
        elif key in (pygame.K_UP, pygame.K_x):
            self.engine.rotate_current("CW")
        elif key == pygame.K_z:
            self.engine.rotate_current("CCW")
        elif key == pygame.K_a:
            self.engine.rotate_current("180")
        elif key in (pygame.K_c, pygame.K_LSHIFT):
            self._do_hold()
        elif key == pygame.K_m:
            self._model_move()
        elif key == pygame.K_n:
            self.auto_play = not self.auto_play
            self.auto_play_timer = 0.0
        elif key == pygame.K_s:
            self._toggle_suggestions()

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self):
        """Main game loop."""
        running = True

        # If no model, try to open file dialog
        if self.model is None:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                model_path = filedialog.askopenfilename(
                    title="Select Model Checkpoint",
                    filetypes=[("PyTorch Checkpoint", "*.pt"), ("All Files", "*.*")],
                    initialdir=os.path.join(_ROOT, "checkpoints"),
                )
                root.destroy()
                if model_path:
                    self._load_model(model_path)
            except Exception as e:
                print(f"File dialog unavailable: {e}")
                print("Use --model path.pt to load a model.")

        while running:
            dt = self.clock.tick(FPS) / 1000.0

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                result = self._handle_input(event)
                if result == "quit":
                    running = False

            # Update animation
            self.animation.update(dt)

            # Update timers
            if self.last_clear_timer > 0:
                self.last_clear_timer -= dt

            # Auto-play
            if self.auto_play and not self.game_over and not self.animation.active:
                self.auto_play_timer += dt
                if self.auto_play_timer >= self.auto_play_delay:
                    self.auto_play_timer = 0.0
                    if self.engine.current_piece is not None:
                        self._model_move()

            # Draw
            self._draw()

        pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TetrisFormer Inference UI")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a .pt checkpoint file")
    parser.add_argument("--rank-q-alpha", type=float, default=0.3,
                        help="Weight for Q-head in combined score (default: 0.3)")
    parser.add_argument("--auto-delay", type=float, default=0.3,
                        help="Delay in seconds between auto-play moves (default: 0.3)")
    args = parser.parse_args()

    ui = TetrisInferenceUI(
        model_path=args.model,
        rank_q_alpha=args.rank_q_alpha,
    )
    ui.auto_play_delay = args.auto_delay
    ui.run()


if __name__ == "__main__":
    main()
