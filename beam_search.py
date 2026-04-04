from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np

from features import extract_candidate_features, summarize_board
from tetrisEngine import KIND_TO_PIECE_ID, PIECE_ID_TO_KIND, TetrisEngine


class ActionCandidate(TypedDict, total=False):
    candidate_index: int
    board: np.ndarray
    stats: dict[str, Any]
    placement: dict[str, Any]
    engine_after: TetrisEngine
    used_hold: bool
    blocks: np.ndarray
    immediate_score: float
    sequence_score: float


@dataclass(frozen=True)
class BeamSearchConfig:
    depth: int = 2
    width: int = 64
    include_hold: bool = True
    height_weight: float = 1.2
    holes_weight: float = 0.9
    covered_holes_weight: float = 0.08
    reachability_weight: float = 0.4
    attack_weight: float = 1.2
    realized_surge_weight: float = 1.4
    banked_surge_weight: float = 0.25
    difficult_clear_weight: float = 0.55
    garbage_solve_weight: float = 0.4
    opener_cancel_weight: float = 0.5
    preserve_b2b_weight: float = 0.65
    break_b2b_penalty: float = 0.8
    t_access_weight: float = 0.15
    i_access_weight: float = 0.15
    tie_break_low_index: bool = True


def _kind_str(piece_or_kind) -> str | None:
    if piece_or_kind is None:
        return None
    if isinstance(piece_or_kind, int):
        return PIECE_ID_TO_KIND.get(int(piece_or_kind))
    return str(getattr(piece_or_kind, "kind", piece_or_kind))


def _spawn_kind(engine: TetrisEngine, kind: str, allow_clutch: bool = True) -> bool:
    piece = engine.spawn_piece(kind)
    if piece is None:
        return False
    try:
        valid = bool(engine.is_position_valid(piece, piece.position, piece.rotation))
    except TypeError:
        valid = bool(engine.is_position_valid(piece, position=piece.position))
    if valid:
        return True
    if allow_clutch:
        clutch_pos = engine._find_clutch_spawn(piece, piece.position)
        if clutch_pos is not None:
            piece.position = clutch_pos
            engine.last_spawn_was_clutch = True
            return True
    engine.current_piece = None
    engine.game_over = True
    engine.game_over_reason = "block_out"
    return False


def _perform_hold(engine: TetrisEngine) -> bool:
    if engine.current_piece is None or engine.game_over:
        return False
    current_kind = _kind_str(engine.current_piece)
    hold_kind = _kind_str(engine.hold)
    engine.hold = KIND_TO_PIECE_ID.get(current_kind, 0)
    engine.current_piece = None
    if hold_kind:
        return _spawn_kind(engine, hold_kind, allow_clutch=True)
    return bool(engine.spawn_next(allow_clutch=True))


def _candidate_key(candidate: ActionCandidate) -> tuple[float, int]:
    return (float(candidate.get("sequence_score", candidate.get("immediate_score", 0.0))), -int(candidate.get("candidate_index", 0)))


def generate_candidates(engine: TetrisEngine, include_hold: bool = True) -> list[ActionCandidate]:
    if engine.current_piece is None or engine.game_over:
        return []

    candidates: list[ActionCandidate] = []
    variants: list[tuple[TetrisEngine, bool]] = [(engine.clone(), False)]
    if include_hold:
        hold_engine = engine.clone()
        if _perform_hold(hold_engine):
            variants.append((hold_engine, True))

    candidate_index = 0
    for action_engine, used_hold in variants:
        if action_engine.current_piece is None or action_engine.game_over:
            continue
        bfs_results = action_engine.bfs_all_placements(include_no_place=False, dedupe_final=True)
        for result in bfs_results:
            placement = dict(result.get("placement") or {})
            placement["used_hold"] = bool(used_hold)
            engine_after = action_engine.clone()
            payload = engine_after.execute_placement(placement, run_end_phase=True)
            if not payload["ok"]:
                continue
            blocks = action_engine.piece_blocks(action_engine.current_piece, position=(placement["x"], placement["y"]), rotation=placement["rotation"])
            candidates.append(
                ActionCandidate(
                    candidate_index=candidate_index,
                    board=np.asarray(result["board"]).copy(),
                    stats=dict(payload["stats"] or {}),
                    placement=placement,
                    engine_after=engine_after,
                    used_hold=bool(used_hold),
                    blocks=np.asarray(blocks).copy(),
                )
            )
            candidate_index += 1
    return candidates


def _transposition_key(engine: TetrisEngine, depth_remaining: int) -> tuple[Any, ...]:
    queue = engine.get_queue_snapshot(next_slots=5)
    pending_batches = tuple(
        (int(batch.get("lines", 0)), int(batch.get("timer", 0)), int(batch.get("col", -1)))
        for batch in list(engine.incoming_garbage or [])
    )
    return (
        np.asarray(engine.board).tobytes(),
        queue["current"],
        queue["hold"],
        tuple(queue["next_ids"]),
        np.asarray(engine.bag, dtype=np.int64).tobytes(),
        int(engine.b2b_chain),
        int(engine.surge_charge),
        int(engine.combo),
        bool(engine.combo_active),
        int(engine.pieces_placed),
        pending_batches,
        int(depth_remaining),
    )


def score_position(
    engine: TetrisEngine,
    opponent_engine: TetrisEngine | None = None,
    move_number: int = 0,
    horizon_state: dict[str, Any] | None = None,
    cfg: BeamSearchConfig | None = None,
) -> float:
    cfg = cfg or BeamSearchConfig()
    if horizon_state is None:
        board = engine.board
        stats = {}
        used_hold = False
        state_engine = engine
        prior_b2b_chain = int(state_engine.b2b_chain)
    else:
        board = np.asarray(horizon_state.get("board", engine.board))
        stats = dict(horizon_state.get("stats") or {})
        used_hold = bool(horizon_state.get("used_hold") or horizon_state.get("placement", {}).get("used_hold"))
        state_engine = horizon_state.get("engine_after") or engine
        prior_b2b_chain = int(engine.b2b_chain)

    summary = summarize_board(board)
    pending = state_engine.get_pending_garbage_summary()
    bag_info = state_engine.get_bag_remainder_counts()
    queue = state_engine.get_queue_snapshot(next_slots=5)
    attack = float(stats.get("attack", 0) or 0)
    realized_surge = float(stats.get("surge_send", 0) or 0)
    banked_surge = float(stats.get("surge_charge", state_engine.surge_charge) or 0)
    difficult = float(stats.get("is_difficult", False))
    qualifies_b2b = bool(stats.get("qualifies_b2b", False))
    breaks_b2b = bool(stats.get("breaks_b2b", False))
    cancel_potential = min(float(pending["total_lines"]), attack)
    opener_leverage = 0.0
    if state_engine.is_opener_phase() and pending["total_lines"] > attack:
        opener_leverage = min(float(pending["total_lines"]), attack * 2.0)

    reachability_bonus = 1.0 / (1.0 + summary["row_transitions"] + summary["column_transitions"])
    t_ready = summary["t_slot_count"] > 0 and (queue["hold"] == "T" or "T" in queue["next_kinds"][:2] or bag_info["counts"].get("T", 0) > 0)
    i_ready = np.any(np.asarray(board[20:40]) == 0) and (queue["hold"] == "I" or "I" in queue["next_kinds"][:2] or bag_info["counts"].get("I", 0) > 0)
    opponent_pressure = 0.0
    if opponent_engine is not None:
        opp_pending = opponent_engine.get_pending_garbage_summary()
        opponent_pressure = float(opp_pending["total_lines"]) * 0.05

    score = 0.0
    score -= cfg.height_weight * (summary["max_height"] / 20.0)
    score -= cfg.holes_weight * (summary["holes"] / 20.0)
    score -= cfg.covered_holes_weight * (summary["covered_hole_burden"] / 40.0)
    score += cfg.reachability_weight * reachability_bonus
    score += cfg.attack_weight * attack
    score += cfg.realized_surge_weight * realized_surge
    score += cfg.banked_surge_weight * banked_surge
    score += cfg.difficult_clear_weight * difficult
    score += cfg.garbage_solve_weight * cancel_potential
    score += cfg.opener_cancel_weight * opener_leverage
    score += cfg.t_access_weight * float(t_ready)
    score += cfg.i_access_weight * float(i_ready)
    score += opponent_pressure
    if qualifies_b2b:
        score += cfg.preserve_b2b_weight
    if breaks_b2b and prior_b2b_chain > 0:
        score -= cfg.break_b2b_penalty
        if realized_surge >= 4.0 or cancel_potential >= 3.0:
            score += 0.9 * realized_surge + 0.4 * cancel_potential
    if used_hold:
        score += 0.05
    return float(score)


def evaluate_candidate_sequence(
    engine_after: TetrisEngine,
    opponent_engine: TetrisEngine | None,
    move_number: int,
    depth: int,
    width: int,
    cfg: BeamSearchConfig | None = None,
    cache: dict[tuple[Any, ...], float] | None = None,
) -> float:
    cfg = cfg or BeamSearchConfig()
    cache = cache if cache is not None else {}
    key = _transposition_key(engine_after, depth)
    if key in cache:
        return cache[key]
    if depth <= 0 or engine_after.game_over or engine_after.current_piece is None:
        cache[key] = 0.0
        return 0.0

    candidates = generate_candidates(engine_after, include_hold=cfg.include_hold)
    if not candidates:
        cache[key] = 0.0
        return 0.0

    for candidate in candidates:
        candidate["immediate_score"] = score_position(
            engine_after,
            opponent_engine=opponent_engine,
            move_number=move_number,
            horizon_state=candidate,
            cfg=cfg,
        )
    candidates.sort(key=_candidate_key, reverse=True)
    frontier = candidates[: max(1, min(int(width), len(candidates)))]

    best = max(
        float(candidate["immediate_score"])
        + evaluate_candidate_sequence(
            candidate["engine_after"],
            opponent_engine,
            move_number + 1,
            depth - 1,
            width,
            cfg=cfg,
            cache=cache,
        )
        for candidate in frontier
    )
    cache[key] = float(best)
    return float(best)


def beam_search_select(
    engine: TetrisEngine,
    opponent_engine: TetrisEngine | None = None,
    move_number: int = 0,
    depth: int | None = None,
    width: int | None = None,
    cfg: BeamSearchConfig | None = None,
) -> ActionCandidate:
    cfg = cfg or BeamSearchConfig()
    if depth is None:
        depth = cfg.depth
    if width is None:
        width = cfg.width

    candidates = generate_candidates(engine, include_hold=cfg.include_hold)
    if not candidates:
        raise RuntimeError("No legal candidates available for beam search.")

    cache: dict[tuple[Any, ...], float] = {}
    for candidate in candidates:
        immediate = score_position(engine, opponent_engine, move_number, horizon_state=candidate, cfg=cfg)
        future = 0.0
        if int(depth) > 1:
            future = evaluate_candidate_sequence(
                candidate["engine_after"],
                opponent_engine,
                move_number + 1,
                int(depth) - 1,
                int(width),
                cfg=cfg,
                cache=cache,
            )
        candidate["immediate_score"] = float(immediate)
        candidate["sequence_score"] = float(immediate + future)

    candidates.sort(key=_candidate_key, reverse=True)
    return candidates[0]
