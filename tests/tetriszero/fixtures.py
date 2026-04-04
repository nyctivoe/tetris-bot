from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from model_v2 import TetrisZeroConfig, TetrisZeroNet
from rl.training import make_fixture_samples
from tetrisEngine import BOARD_HEIGHT, BOARD_WIDTH, GARBAGE_ID, HIDDEN_ROWS, KIND_TO_PIECE_ID, TetrisEngine


def ascii_visible_board(rows: list[str]) -> np.ndarray:
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int16)
    visible = board[HIDDEN_ROWS:]
    offset = max(0, 20 - len(rows))
    for row_index, row in enumerate(rows[-20:]):
        target = visible[offset + row_index]
        for col_index, ch in enumerate(row[:BOARD_WIDTH]):
            if ch in {"#", "X"}:
                target[col_index] = 9
            elif ch == "G":
                target[col_index] = GARBAGE_ID
    return board


def make_engine(
    *,
    seed: int = 0,
    spin_mode: str = "all_spin",
    board_rows: list[str] | None = None,
    current: str = "T",
    hold: str | None = None,
    next_ids: list[int] | None = None,
    b2b_chain: int = 0,
    surge_charge: int = 0,
    combo: int = 0,
    combo_active: bool = False,
    pieces_placed: int = 0,
    total_lines_cleared: int = 0,
    total_attack_sent: int = 0,
    total_attack_canceled: int = 0,
    incoming_garbage: list[dict[str, int]] | None = None,
) -> TetrisEngine:
    engine = TetrisEngine(spin_mode=spin_mode, rng=np.random.default_rng(seed))
    if board_rows is not None:
        engine.board = ascii_visible_board(board_rows)
    engine.current_piece = engine.spawn_piece(current)
    engine.hold = None if hold is None else KIND_TO_PIECE_ID[hold]
    if next_ids is None:
        next_ids = [KIND_TO_PIECE_ID[kind] for kind in ("I", "O", "T", "S", "Z", "J", "L")]
    engine.bag = np.asarray(next_ids, dtype=np.int64)
    engine.bag_size = len(engine.bag)
    engine.b2b_chain = int(b2b_chain)
    engine.surge_charge = int(surge_charge)
    engine.combo = int(combo)
    engine.combo_active = bool(combo_active)
    engine.pieces_placed = int(pieces_placed)
    engine.total_lines_cleared = int(total_lines_cleared)
    engine.total_attack_sent = int(total_attack_sent)
    engine.total_attack_canceled = int(total_attack_canceled)
    engine.incoming_garbage = [dict(batch) for batch in list(incoming_garbage or [])]
    engine.garbage_col = None
    engine.game_over = False
    engine.game_over_reason = None
    return engine


def make_piece(
    engine: TetrisEngine,
    *,
    kind: str = "T",
    rotation: int = 0,
    position=(4, 4),
    last_rotation_dir=1,
    last_kick_index=None,
):
    piece = engine.spawn_piece(kind, position=position, rotation=rotation)
    piece.last_action_was_rotation = True
    piece.last_rotation_dir = last_rotation_dir
    piece.last_kick_index = last_kick_index
    return piece


def occupy_cells(engine: TetrisEngine, cells: list[tuple[int, int]]) -> None:
    engine.board = np.zeros_like(engine.board)
    for x, y in cells:
        engine.board[y, x] = 9


def make_non_t_all_spin_piece(engine: TetrisEngine):
    piece = make_piece(engine, kind="L", rotation=0, position=(4, 4), last_rotation_dir=1)
    current_blocks = {tuple(block) for block in engine.piece_blocks(piece)}
    occupied: set[tuple[int, int]] = set()
    for shifted_position in ((piece.position[0] - 1, piece.position[1]), (piece.position[0] + 1, piece.position[1]), (piece.position[0], piece.position[1] - 1)):
        shifted_blocks = [tuple(block) for block in engine.piece_blocks(piece, position=shifted_position)]
        for block in shifted_blocks:
            if block not in current_blocks:
                occupied.add(block)
                break
    occupy_cells(engine, list(occupied))
    return piece


def make_manual_candidate(
    engine: TetrisEngine,
    *,
    candidate_index: int,
    stats: dict[str, Any],
    used_hold: bool = False,
    board_after: np.ndarray | None = None,
) -> dict[str, Any]:
    if board_after is None:
        board_after = engine.board.copy()
        board_after[-1, candidate_index % BOARD_WIDTH] = 9
    engine_after = engine.clone()
    engine_after.board = np.asarray(board_after).copy()
    blocks = np.array([[candidate_index % BOARD_WIDTH, BOARD_HEIGHT - 1]], dtype=np.int16)
    return {
        "candidate_index": int(candidate_index),
        "board": np.asarray(board_after).copy(),
        "stats": dict(stats),
        "placement": {
            "x": int(candidate_index % BOARD_WIDTH),
            "y": BOARD_HEIGHT - 2,
            "rotation": 0,
            "kind": engine.current_piece.kind if engine.current_piece is not None else "T",
            "used_hold": bool(used_hold),
        },
        "engine_after": engine_after,
        "used_hold": bool(used_hold),
        "blocks": blocks,
    }


@pytest.fixture
def seeded_engine():
    return make_engine(seed=123, current="T")


@pytest.fixture
def preserve_break_fixture():
    engine = make_engine(seed=7, current="T", b2b_chain=6, surge_charge=5, incoming_garbage=[{"lines": 2, "timer": 2, "col": 0}])
    preserve = make_manual_candidate(
        engine,
        candidate_index=0,
        stats={
            "attack": 2,
            "base_attack": 1,
            "lines_cleared": 2,
            "is_difficult": True,
            "is_spin": True,
            "spin_type": 2,
            "qualifies_b2b": True,
            "breaks_b2b": False,
            "surge_send": 0,
            "surge_charge": 6,
        },
    )
    break_board = engine.board.copy()
    break_board[-1, 5] = 9
    break_move = make_manual_candidate(
        engine,
        candidate_index=1,
        stats={
            "attack": 7,
            "base_attack": 2,
            "lines_cleared": 1,
            "is_difficult": False,
            "is_spin": False,
            "spin_type": 0,
            "qualifies_b2b": False,
            "breaks_b2b": True,
            "surge_send": 5,
            "surge_charge": 0,
        },
        board_after=break_board,
    )
    return engine, preserve, break_move


@pytest.fixture
def opener_cancel_engine():
    return make_engine(
        seed=9,
        current="I",
        pieces_placed=5,
        incoming_garbage=[{"lines": 3, "timer": 2, "col": 0}, {"lines": 2, "timer": 4, "col": 4}],
    )


@pytest.fixture
def synthetic_replay_samples():
    return make_fixture_samples(count=6, seed=11)


@pytest.fixture
def tiny_model_batch():
    samples = make_fixture_samples(count=4, seed=13)
    return samples


@pytest.fixture
def tiny_model():
    model = TetrisZeroNet(TetrisZeroConfig())
    model.eval()
    return model
