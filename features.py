from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from tetrisEngine import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    GARBAGE_ID,
    HIDDEN_ROWS,
    KIND_TO_PIECE_ID,
    PIECE_ID_TO_KIND,
    VISIBLE_HEIGHT,
    TetrisEngine,
)


BOARD_TENSOR_SHAPE = (12, VISIBLE_HEIGHT, BOARD_WIDTH)
CONTEXT_VECTOR_SIZE = 28
CANDIDATE_FEATURE_SIZE = 32


def _visible_board(board: np.ndarray) -> np.ndarray:
    return np.asarray(board[HIDDEN_ROWS:BOARD_HEIGHT], dtype=np.int16)


def _piece_id(piece_or_kind: Any) -> int:
    if piece_or_kind is None:
        return 0
    if isinstance(piece_or_kind, int):
        return int(piece_or_kind)
    kind = getattr(piece_or_kind, "kind", piece_or_kind)
    return int(KIND_TO_PIECE_ID.get(str(kind), 0))


def _piece_kind(piece_or_kind: Any) -> str | None:
    if piece_or_kind is None:
        return None
    if isinstance(piece_or_kind, int):
        return PIECE_ID_TO_KIND.get(int(piece_or_kind))
    kind = getattr(piece_or_kind, "kind", piece_or_kind)
    if kind is None:
        return None
    return str(kind)


def _column_heights(occ: np.ndarray) -> np.ndarray:
    heights = np.zeros((occ.shape[1],), dtype=np.float32)
    for x in range(occ.shape[1]):
        filled = np.flatnonzero(occ[:, x])
        heights[x] = float(occ.shape[0] - filled[0]) if filled.size else 0.0
    return heights


def _hole_map(occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks_above = np.cumsum(occ.astype(np.int16), axis=0)
    holes = (~occ) & (blocks_above > 0)
    covered_depth = np.where(holes, blocks_above / 10.0, 0.0).astype(np.float32)
    return holes.astype(np.float32), covered_depth


def _row_fullness(occ: np.ndarray) -> np.ndarray:
    fullness = occ.sum(axis=1, keepdims=True).astype(np.float32) / float(BOARD_WIDTH)
    return np.broadcast_to(fullness, occ.shape).astype(np.float32, copy=True)


def _row_transitions(occ: np.ndarray) -> np.ndarray:
    values = np.zeros((occ.shape[0], 1), dtype=np.float32)
    for y in range(occ.shape[0]):
        transitions = float(np.count_nonzero(occ[y, :-1] != occ[y, 1:]))
        values[y, 0] = transitions / float(BOARD_WIDTH)
    return np.broadcast_to(values, occ.shape).astype(np.float32, copy=True)


def _column_transitions(occ: np.ndarray) -> np.ndarray:
    values = np.zeros((1, occ.shape[1]), dtype=np.float32)
    for x in range(occ.shape[1]):
        transitions = float(np.count_nonzero(occ[:-1, x] != occ[1:, x]))
        values[0, x] = transitions / float(VISIBLE_HEIGHT)
    return np.broadcast_to(values, occ.shape).astype(np.float32, copy=True)


def _well_map(heights: np.ndarray) -> np.ndarray:
    well_depths = np.zeros((BOARD_WIDTH,), dtype=np.float32)
    for x in range(BOARD_WIDTH):
        left = heights[x - 1] if x > 0 else VISIBLE_HEIGHT
        right = heights[x + 1] if x < BOARD_WIDTH - 1 else VISIBLE_HEIGHT
        depth = max(0.0, min(left, right) - heights[x])
        well_depths[x] = depth / 10.0
    return np.broadcast_to(well_depths.reshape(1, -1), (VISIBLE_HEIGHT, BOARD_WIDTH)).astype(np.float32, copy=True)


def _reachability_map(occ: np.ndarray) -> np.ndarray:
    h, w = occ.shape
    distances = np.full((h, w), fill_value=h, dtype=np.int16)
    queue: deque[tuple[int, int]] = deque()

    for x in range(w):
        if not occ[0, x]:
            distances[0, x] = 0
            queue.append((0, x))

    while queue:
        y, x = queue.popleft()
        base = distances[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or occ[ny, nx]:
                continue
            if distances[ny, nx] <= base + 1:
                continue
            distances[ny, nx] = base + 1
            queue.append((ny, nx))

    return np.clip(distances.astype(np.float32) / float(VISIBLE_HEIGHT), 0.0, 1.0)


def _detect_tslot_map(board: np.ndarray) -> np.ndarray:
    occ = board != 0
    out = np.zeros_like(board, dtype=np.float32)
    h, w = board.shape
    for y in range(h - 2):
        for x in range(w - 2):
            corners = int(occ[y, x]) + int(occ[y, x + 2]) + int(occ[y + 2, x]) + int(occ[y + 2, x + 2])
            if corners < 3:
                continue
            cells = [(x + 1, y), (x, y + 1), (x + 1, y + 1), (x + 2, y + 1)]
            if all(not occ[cy, cx] for cx, cy in cells):
                for cx, cy in cells:
                    out[cy, cx] = 1.0
    return out


def _detect_all_spin_slot_map(board: np.ndarray) -> np.ndarray:
    occ = board != 0
    out = np.zeros_like(board, dtype=np.float32)
    h, w = board.shape
    for y in range(h):
        for x in range(w):
            if occ[y, x]:
                continue
            blocked = 0
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w or occ[ny, nx]:
                    blocked += 1
            if blocked >= 3:
                out[y, x] = 1.0
    return out


def summarize_board(board: np.ndarray) -> dict[str, float]:
    visible = _visible_board(np.asarray(board))
    occ = visible != 0
    heights = _column_heights(occ)
    holes, covered = _hole_map(occ)
    transitions_r = _row_transitions(occ)
    transitions_c = _column_transitions(occ)
    return {
        "max_height": float(heights.max()) if heights.size else 0.0,
        "holes": float(holes.sum()),
        "covered_hole_burden": float(covered.sum()),
        "row_transitions": float(transitions_r[:, 0].sum()),
        "column_transitions": float(transitions_c[0, :].sum()),
        "t_slot_count": float(_detect_tslot_map(visible).sum()),
        "all_spin_slot_count": float(_detect_all_spin_slot_map(visible).sum()),
        "overflow": float(np.count_nonzero(np.asarray(board[:HIDDEN_ROWS]) != 0)),
    }


def encode_board(engine: TetrisEngine) -> np.ndarray:
    board = _visible_board(engine.board)
    occ = board != 0
    garbage = board == GARBAGE_ID
    heights = _column_heights(occ)
    height_map = np.broadcast_to((heights / float(VISIBLE_HEIGHT)).reshape(1, -1), board.shape).astype(np.float32, copy=True)
    holes, covered_depth = _hole_map(occ)
    row_fullness = _row_fullness(occ)
    tslot_map = _detect_tslot_map(board)
    all_spin_map = _detect_all_spin_slot_map(board)
    well_map = _well_map(heights)
    reachability = _reachability_map(occ)
    row_transitions = _row_transitions(occ)
    column_transitions = _column_transitions(occ)

    return np.stack(
        [
            occ.astype(np.float32),
            garbage.astype(np.float32),
            height_map,
            holes,
            covered_depth,
            row_fullness,
            tslot_map,
            all_spin_map,
            well_map,
            reachability,
            row_transitions,
            column_transitions,
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _hold_lock_preference_flag(engine: TetrisEngine) -> float:
    queue = engine.get_queue_snapshot(next_slots=5)
    bag_info = engine.get_bag_remainder_counts()
    visible = _visible_board(engine.board)
    has_tslot = bool(_detect_tslot_map(visible).any())
    heights = _column_heights(visible != 0)
    has_i_well = bool(np.any(heights <= np.maximum(np.roll(heights, 1), np.roll(heights, -1)) - 4.0))
    scarce_t = queue["hold"] != "T" and queue["next_kinds"][:2].count("T") == 0 and bag_info["counts"].get("T", 0) <= 0
    scarce_i = queue["hold"] != "I" and queue["next_kinds"][:2].count("I") == 0 and bag_info["counts"].get("I", 0) <= 0
    return float((has_tslot and scarce_t) or (has_i_well and scarce_i))


def encode_context(
    engine: TetrisEngine,
    opponent_engine: TetrisEngine | None = None,
    move_number: int = 0,
) -> np.ndarray:
    board_summary = summarize_board(engine.board)
    garbage = engine.get_pending_garbage_summary()
    bag_info = engine.get_bag_remainder_counts()
    context = np.zeros((CONTEXT_VECTOR_SIZE,), dtype=np.float32)

    context[0] = float(engine.combo) / 20.0
    context[1] = float(engine.b2b_chain) / 20.0
    context[2] = float(engine.b2b_chain > 0)
    context[3] = float(engine.surge_charge) / 20.0
    context[4] = float(engine.surge_charge) / 20.0
    context[5] = float(garbage["total_lines"]) / 20.0
    context[6] = float(garbage["min_timer"]) / 60.0
    context[7] = float(garbage["max_timer"]) / 60.0
    context[8] = float(garbage["batch_count"]) / 10.0
    context[9] = float(garbage["landing_within_one_ply"])
    context[10] = float(move_number) / 200.0
    context[11] = min(float(engine.pieces_placed), 14.0) / 14.0
    context[12] = float(engine.is_opener_phase())
    context[13] = board_summary["max_height"] / 20.0
    context[14] = board_summary["holes"] / 20.0
    context[15] = board_summary["covered_hole_burden"] / 40.0
    context[16] = float(engine.total_lines_cleared) / 100.0
    context[17] = float(engine.total_attack_sent) / 100.0
    context[18] = float(engine.total_attack_canceled) / 100.0
    context[19] = float(bag_info["counts"].get("T", 0)) / 7.0
    context[20] = float(bag_info["counts"].get("I", 0)) / 7.0
    context[21] = float(bag_info["remaining"]) / 7.0
    context[22] = float(bag_info["bag_position"]) / 7.0
    context[23] = _hold_lock_preference_flag(engine)

    if opponent_engine is not None:
        opp_summary = summarize_board(opponent_engine.board)
        opp_garbage = opponent_engine.get_pending_garbage_summary()
        context[24] = opp_summary["max_height"] / 20.0
        context[25] = opp_summary["holes"] / 20.0
        context[26] = float(opponent_engine.b2b_chain + opponent_engine.surge_charge) / 20.0
        context[27] = float(opp_garbage["total_lines"]) / 20.0

    return context.astype(np.float32, copy=False)


def _candidate_payload(
    engine: TetrisEngine,
    candidate: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, dict[str, Any]]:
    if "stats" in candidate and "board" in candidate:
        board_after = np.asarray(candidate["board"])
        stats = dict(candidate.get("stats") or {})
        placement = dict(candidate.get("placement") or {})
        blocks = np.asarray(candidate.get("blocks")) if candidate.get("blocks") is not None else None
        if blocks is None:
            probe = engine.clone()
            if probe.apply_placement(placement):
                blocks = probe.piece_blocks(probe.current_piece)
            else:
                blocks = np.zeros((0, 2), dtype=np.int16)
        return board_after, stats, np.asarray(blocks), placement

    placement = dict(candidate.get("placement") or candidate)
    probe = engine.clone()
    if not probe.apply_placement(placement):
        return (
            probe.board.copy(),
            {
                "lines_cleared": 0,
                "is_difficult": False,
                "is_spin": False,
                "spin_type": 0,
                "qualifies_b2b": False,
                "breaks_b2b": False,
                "surge_send": 0,
                "surge_charge": probe.surge_charge,
                "base_attack": 0,
                "attack": 0,
            },
            np.zeros((0, 2), dtype=np.int16),
            placement,
        )
    payload = probe.predict_post_lock_stats(probe.current_piece)
    return payload["board"], payload["stats"], payload["blocks"], placement


def extract_candidate_features(
    engine: TetrisEngine,
    candidate: dict[str, Any],
    opponent_engine: TetrisEngine | None = None,
    move_number: int = 0,
) -> np.ndarray:
    del opponent_engine, move_number
    before = summarize_board(engine.board)
    pending = engine.get_pending_garbage_summary()
    board_after, stats, blocks, placement = _candidate_payload(engine, candidate)
    after = summarize_board(board_after)
    features = np.zeros((CANDIDATE_FEATURE_SIZE,), dtype=np.float32)

    if blocks.size > 0:
        center_x = int(np.clip(round(float(blocks[:, 0].mean())), 0, BOARD_WIDTH - 1))
        features[center_x] = 1.0
        max_visible_y = int(np.clip(np.max(blocks[:, 1]) - HIDDEN_ROWS + 1, 0, VISIBLE_HEIGHT))
        features[30] = float(max_visible_y) / 20.0

    rotation = int(placement.get("rotation", placement.get("r", 0)))
    if isinstance(rotation, str):
        rotation = {"N": 0, "E": 1, "S": 2, "W": 3}.get(rotation, 0)
    features[10 + int(rotation) % 4] = 1.0

    features[14] = float(stats.get("is_difficult", False))
    features[15] = float(stats.get("is_spin", bool(stats.get("spin"))))
    features[16] = float(stats.get("spin_type", 0)) / 2.0
    features[17] = float(stats.get("lines_cleared", 0)) / 4.0
    features[18] = float(stats.get("qualifies_b2b", False))
    features[19] = float(stats.get("breaks_b2b", False))
    features[20] = float(stats.get("surge_send", 0)) / 20.0
    features[21] = float(stats.get("surge_charge", 0)) / 20.0
    pre_surge_attack = float(stats.get("attack", 0) or 0) - float(stats.get("surge_send", 0) or 0)
    features[22] = pre_surge_attack / 10.0
    features[23] = float(stats.get("attack", 0) or 0) / 20.0
    features[24] = min(float(stats.get("attack", 0) or 0), float(pending["total_lines"])) / 20.0
    features[25] = (after["max_height"] - before["max_height"]) / 20.0
    features[26] = (after["holes"] - before["holes"]) / 10.0
    features[27] = float(after["t_slot_count"] > before["t_slot_count"])
    features[28] = float(after["all_spin_slot_count"] > before["all_spin_slot_count"])
    features[29] = float(bool(candidate.get("used_hold") or placement.get("used_hold")))
    features[31] = float(engine.is_opener_phase())
    return features.astype(np.float32, copy=False)


def build_model_inputs(
    engine: TetrisEngine,
    candidates: list[dict[str, Any]],
    opponent_engine: TetrisEngine | None = None,
    move_number: int = 0,
) -> dict[str, np.ndarray]:
    queue = engine.get_queue_snapshot(next_slots=5)
    piece_ids = np.asarray(queue["piece_ids"][:7], dtype=np.int64)
    while piece_ids.shape[0] < 7:
        piece_ids = np.concatenate([piece_ids, np.zeros((1,), dtype=np.int64)])

    candidate_features = (
        np.stack(
            [extract_candidate_features(engine, candidate, opponent_engine, move_number) for candidate in candidates],
            axis=0,
        ).astype(np.float32)
        if candidates
        else np.zeros((0, CANDIDATE_FEATURE_SIZE), dtype=np.float32)
    )

    return {
        "board_tensor": encode_board(engine),
        "piece_ids": piece_ids,
        "context_scalars": encode_context(engine, opponent_engine, move_number),
        "candidate_features": candidate_features,
        "candidate_mask": np.ones((candidate_features.shape[0],), dtype=np.bool_),
    }
