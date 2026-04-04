from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from beam_search import BeamSearchConfig, beam_search_select, generate_candidates, score_position
from features import CANDIDATE_FEATURE_SIZE, build_model_inputs
from model_v2 import TetrisZeroNet, resolve_module_device
from pvp_game import PvpGameConfig, calculate_garbage_timer
from tetrisEngine import KIND_TO_PIECE_ID, OPENER_PHASE_PIECES, TetrisEngine


PIECE_ORDER = ("I", "O", "T", "S", "Z", "J", "L")
PVP_RULES = PvpGameConfig()


@dataclass(frozen=True)
class MctsConfig:
    enabled: bool = True
    simulations: int = 32
    c_puct: float = 1.5
    temperature: float = 0.0
    beam_cfg: BeamSearchConfig = BeamSearchConfig()
    max_depth: int = 3
    visible_queue_depth: int = 5
    value_scale: float = 8.0


@dataclass
class SearchState:
    active_engine: TetrisEngine
    passive_engine: TetrisEngine | None
    move_number: int
    root_to_move: bool
    active_known_next: int
    passive_known_next: int | None = None


@dataclass
class TreeNode:
    candidates: list[dict[str, Any]]
    priors: np.ndarray
    state_value: float
    visits: np.ndarray
    value_sums: np.ndarray


def _softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=False)
    shifted = values.astype(np.float32) - np.max(values.astype(np.float32))
    exp = np.exp(shifted)
    denom = float(exp.sum())
    if denom <= 0:
        return np.full((len(values),), 1.0 / max(1, len(values)), dtype=np.float32)
    return (exp / denom).astype(np.float32, copy=False)


def _tensorize_model_inputs(inputs: dict[str, np.ndarray], model: Any) -> tuple[torch.Tensor, ...]:
    device = resolve_module_device(model)
    board = torch.from_numpy(inputs["board_tensor"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    pieces = torch.from_numpy(inputs["piece_ids"]).unsqueeze(0).to(device=device, dtype=torch.long)
    context = torch.from_numpy(inputs["context_scalars"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    candidate_features = torch.from_numpy(inputs["candidate_features"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    candidate_mask = torch.from_numpy(inputs["candidate_mask"]).unsqueeze(0).to(device=device, dtype=torch.bool)
    return board, pieces, context, candidate_features, candidate_mask


def _model_priors_and_value(
    model: TetrisZeroNet,
    engine: TetrisEngine,
    opponent_engine: TetrisEngine | None,
    move_number: int,
    candidates: list[dict[str, Any]],
) -> tuple[np.ndarray, float]:
    inputs = build_model_inputs(engine, candidates, opponent_engine=opponent_engine, move_number=move_number)
    if not candidates:
        inputs["candidate_features"] = np.zeros((1, CANDIDATE_FEATURE_SIZE), dtype=np.float32)
        inputs["candidate_mask"] = np.zeros((1,), dtype=np.bool_)
    board, pieces, context, candidate_features, candidate_mask = _tensorize_model_inputs(inputs, model)
    with torch.no_grad():
        outputs = model(board, pieces, context, candidate_features, candidate_mask)
    priors = np.zeros((len(candidates),), dtype=np.float32)
    if candidates:
        priors = (
            torch.softmax(outputs["policy_logits"], dim=-1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    value = float(outputs["value"].squeeze(0).detach().cpu())
    return priors, value


def _heuristic_values(
    engine: TetrisEngine,
    opponent_engine: TetrisEngine | None,
    move_number: int,
    candidates: list[dict[str, Any]],
    beam_cfg: BeamSearchConfig,
) -> np.ndarray:
    return np.asarray(
        [
            score_position(engine, opponent_engine=opponent_engine, move_number=move_number, horizon_state=candidate, cfg=beam_cfg)
            for candidate in candidates
        ],
        dtype=np.float32,
    )


def _evaluate_state(
    state: SearchState,
    model: TetrisZeroNet | None,
    cfg: MctsConfig,
) -> float:
    if model is not None and hasattr(model, "encode_state") and hasattr(model, "value_head"):
        _, actor_value = _model_priors_and_value(
            model,
            state.active_engine,
            state.passive_engine,
            state.move_number,
            candidates=[],
        )
    else:
        raw_value = score_position(
            state.active_engine,
            opponent_engine=state.passive_engine,
            move_number=state.move_number,
            cfg=cfg.beam_cfg,
        )
        actor_value = float(np.tanh(raw_value / max(1.0e-6, float(cfg.value_scale))))
    return actor_value if state.root_to_move else -actor_value


def _pending_key(engine: TetrisEngine) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(batch.get("lines", 0)), int(batch.get("timer", 0)), int(batch.get("col", -1)))
        for batch in list(engine.incoming_garbage or [])
    )


def _engine_key(engine: TetrisEngine) -> tuple[Any, ...]:
    queue = engine.get_queue_snapshot(next_slots=5)
    bag_info = engine.get_bag_remainder_counts()
    return (
        np.asarray(engine.board).tobytes(),
        queue["current"],
        queue["hold"],
        tuple(queue["next_ids"]),
        tuple(int(bag_info["counts"].get(kind, 0)) for kind in PIECE_ORDER),
        int(bag_info["remaining"]),
        int(bag_info["bag_position"]),
        int(engine.b2b_chain),
        int(engine.surge_charge),
        int(engine.combo),
        bool(engine.combo_active),
        int(engine.pieces_placed),
        _pending_key(engine),
        bool(engine.game_over),
        engine.game_over_reason,
    )


def _state_key(state: SearchState) -> tuple[Any, ...]:
    return (
        _engine_key(state.active_engine),
        None if state.passive_engine is None else _engine_key(state.passive_engine),
        int(state.move_number),
        bool(state.root_to_move),
        int(state.active_known_next),
        None if state.passive_known_next is None else int(state.passive_known_next),
    )


def _current_piece_valid(engine: TetrisEngine) -> bool:
    piece = getattr(engine, "current_piece", None)
    if piece is None:
        return True
    try:
        return bool(engine.is_position_valid(piece, piece.position, piece.rotation))
    except TypeError:
        return bool(engine.is_position_valid(piece, position=piece.position))


def _select_child_index(node: TreeNode, root_to_move: bool, c_puct: float) -> int:
    total_visits = max(1.0, float(node.visits.sum()))
    q_values = node.value_sums / np.maximum(node.visits, 1)
    signed_q = q_values if root_to_move else -q_values
    u_values = float(c_puct) * node.priors * np.sqrt(total_visits) / (1.0 + node.visits)
    return int(np.argmax(signed_q + u_values))


def select_action_from_visits(
    visits: np.ndarray,
    temperature: float,
    rng: np.random.Generator | None = None,
) -> int:
    if visits.size == 0:
        raise RuntimeError("Cannot select from empty visit counts.")
    if temperature <= 1.0e-6:
        return int(np.argmax(visits))
    powered = np.power(visits.astype(np.float32), 1.0 / float(temperature))
    probs = powered / max(1.0e-8, float(powered.sum()))
    rng = rng if rng is not None else np.random.default_rng()
    return int(rng.choice(len(visits), p=probs))


def _terminal_value(state: SearchState) -> float | None:
    active = state.active_engine
    passive = state.passive_engine
    if active.game_over:
        return -1.0 if state.root_to_move else 1.0
    if passive is not None and passive.game_over:
        return 1.0 if state.root_to_move else -1.0
    return None


def _hidden_piece_distribution(engine: TetrisEngine) -> tuple[np.ndarray, np.ndarray, int]:
    bag_info = engine.get_bag_remainder_counts()
    remaining = int(bag_info["remaining"])
    if remaining > 0:
        piece_ids = np.asarray(
            [KIND_TO_PIECE_ID[kind] for kind in PIECE_ORDER if int(bag_info["counts"].get(kind, 0)) > 0],
            dtype=np.int64,
        )
        probs = np.asarray(
            [float(bag_info["counts"].get(kind, 0)) / float(remaining) for kind in PIECE_ORDER if int(bag_info["counts"].get(kind, 0)) > 0],
            dtype=np.float32,
        )
        return piece_ids, probs, remaining
    piece_ids = np.asarray([KIND_TO_PIECE_ID[kind] for kind in PIECE_ORDER], dtype=np.int64)
    probs = np.full((len(piece_ids),), 1.0 / float(len(piece_ids)), dtype=np.float32)
    return piece_ids, probs, 7


def _remove_hidden_piece_from_bag(engine: TetrisEngine, piece_id: int, search_limit: int) -> None:
    bag = list(np.asarray(engine.bag, dtype=np.int64))
    limit = min(len(bag), max(1, int(search_limit)))
    removed = False
    for index in range(limit):
        if int(bag[index]) == int(piece_id):
            del bag[index]
            removed = True
            break
    if not removed:
        for index, candidate in enumerate(bag):
            if int(candidate) == int(piece_id):
                del bag[index]
                removed = True
                break
    if removed:
        engine.bag = np.asarray(bag, dtype=np.int64)
        engine.bag_size = len(engine.bag)
        if engine.bag_size <= 14:
            engine.generate_bag()


def _spawn_specific_piece(engine: TetrisEngine, piece_id: int, allow_clutch: bool) -> bool:
    piece = engine.spawn_piece(int(piece_id))
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
            engine.current_piece = piece
            engine.last_spawn_was_clutch = True
            return True
    engine.current_piece = None
    engine.game_over = True
    engine.game_over_reason = "block_out"
    return False


def _spawn_next_for_search(
    engine: TetrisEngine,
    *,
    known_next_available: int,
    allow_clutch: bool,
    rng: np.random.Generator,
) -> int:
    if int(known_next_available) > 0:
        if not engine.spawn_next(allow_clutch=allow_clutch):
            return max(0, int(known_next_available) - 1)
        return max(0, int(known_next_available) - 1)

    piece_ids, probs, search_limit = _hidden_piece_distribution(engine)
    piece_id = int(rng.choice(piece_ids, p=probs))
    _remove_hidden_piece_from_bag(engine, piece_id, search_limit)
    _spawn_specific_piece(engine, piece_id, allow_clutch=allow_clutch)
    return 0


def _advance_singleplayer_state(
    state: SearchState,
    candidate: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[SearchState | None, float | None]:
    next_known = max(0, int(state.active_known_next) - 1)
    if int(state.active_known_next) > 0 and candidate.get("engine_after") is not None:
        engine_after = candidate["engine_after"].clone()
        if engine_after.game_over:
            return None, -1.0
        return SearchState(
            active_engine=engine_after,
            passive_engine=None,
            move_number=int(state.move_number) + 1,
            root_to_move=True,
            active_known_next=next_known,
        ), None

    engine_after = state.active_engine.clone()
    result = engine_after.execute_placement(dict(candidate.get("placement") or {}), run_end_phase=False)
    if not result["ok"] or engine_after.game_over:
        return None, -1.0
    next_known = _spawn_next_for_search(
        engine_after,
        known_next_available=int(state.active_known_next),
        allow_clutch=bool(result["lines_cleared"] > 0),
        rng=rng,
    )
    if engine_after.game_over:
        return None, -1.0
    return SearchState(
        active_engine=engine_after,
        passive_engine=None,
        move_number=int(state.move_number) + 1,
        root_to_move=True,
        active_known_next=int(next_known),
    ), None


def _advance_pvp_state(
    state: SearchState,
    candidate: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[SearchState | None, float | None]:
    active = state.active_engine.clone()
    passive = state.passive_engine.clone()

    landed = active.tick_garbage()
    if landed and not _current_piece_valid(active):
        active.current_piece = None
        active.game_over = True
        active.game_over_reason = "garbage_top_out"
    if active.game_over:
        return None, (-1.0 if state.root_to_move else 1.0)

    result = active.execute_placement(dict(candidate.get("placement") or {}), run_end_phase=False)
    if not result["ok"] or active.game_over:
        return None, (-1.0 if state.root_to_move else 1.0)

    resolve = active.resolve_outgoing_attack(
        int(result["attack"]),
        opener_phase=bool(active.pieces_placed <= OPENER_PHASE_PIECES),
    )
    sent = int(resolve["sent"])
    if sent > 0:
        passive.add_incoming_garbage(
            lines=sent,
            timer=calculate_garbage_timer(state.move_number, PVP_RULES),
            col=None,
        )

    updated_known_next = _spawn_next_for_search(
        active,
        known_next_available=int(state.active_known_next),
        allow_clutch=bool(result["lines_cleared"] > 0),
        rng=rng,
    )
    if active.game_over:
        return None, (-1.0 if state.root_to_move else 1.0)

    next_move_number = int(state.move_number) if state.root_to_move else int(state.move_number) + 1
    return SearchState(
        active_engine=passive,
        passive_engine=active,
        move_number=next_move_number,
        root_to_move=not state.root_to_move,
        active_known_next=int(state.passive_known_next or 0),
        passive_known_next=int(updated_known_next),
    ), None


def _advance_state(
    state: SearchState,
    candidate: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[SearchState | None, float | None]:
    if state.passive_engine is None:
        return _advance_singleplayer_state(state, candidate, rng)
    return _advance_pvp_state(state, candidate, rng)


def _create_node(
    state: SearchState,
    model: TetrisZeroNet | None,
    cfg: MctsConfig,
    *,
    root_candidates: list[dict[str, Any]] | None = None,
) -> TreeNode:
    candidates = root_candidates if root_candidates is not None else generate_candidates(
        state.active_engine,
        include_hold=cfg.beam_cfg.include_hold,
    )
    if model is not None:
        priors, state_value = _model_priors_and_value(
            model,
            state.active_engine,
            state.passive_engine,
            state.move_number,
            candidates,
        )
        state_value = state_value if state.root_to_move else -state_value
    else:
        heuristic_values = _heuristic_values(
            state.active_engine,
            state.passive_engine,
            state.move_number,
            candidates,
            cfg.beam_cfg,
        )
        priors = _softmax(heuristic_values)
        state_value = _evaluate_state(state, model=None, cfg=cfg)
    return TreeNode(
        candidates=candidates,
        priors=priors,
        state_value=float(state_value),
        visits=np.zeros((len(candidates),), dtype=np.int32),
        value_sums=np.zeros((len(candidates),), dtype=np.float32),
    )


def _simulate(
    state: SearchState,
    nodes: dict[tuple[Any, ...], TreeNode],
    model: TetrisZeroNet | None,
    cfg: MctsConfig,
    rng: np.random.Generator,
    *,
    depth: int,
    root_candidates: list[dict[str, Any]] | None = None,
) -> float:
    terminal_value = _terminal_value(state)
    if terminal_value is not None:
        return float(terminal_value)
    if int(depth) >= int(cfg.max_depth):
        return _evaluate_state(state, model=model, cfg=cfg)

    key = _state_key(state)
    node = nodes.get(key)
    if node is None:
        node = _create_node(state, model, cfg, root_candidates=root_candidates)
        nodes[key] = node
        return float(node.state_value)
    if not node.candidates:
        return float(node.state_value)

    index = _select_child_index(node, state.root_to_move, cfg.c_puct)
    next_state, terminal_value = _advance_state(state, node.candidates[index], rng)
    if terminal_value is None and next_state is not None:
        leaf_value = _simulate(next_state, nodes, model, cfg, rng, depth=depth + 1)
    else:
        leaf_value = float(terminal_value if terminal_value is not None else node.state_value)

    node.visits[index] += 1
    node.value_sums[index] += float(leaf_value)
    return float(leaf_value)


def run_mcts(
    engine: TetrisEngine,
    opponent_engine: TetrisEngine | None,
    move_number: int,
    model: TetrisZeroNet | None,
    candidates: list[dict[str, Any]] | None,
    cfg: MctsConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or MctsConfig()
    if not cfg.enabled:
        selected = beam_search_select(engine, opponent_engine=opponent_engine, move_number=move_number, cfg=cfg.beam_cfg)
        if candidates is None:
            candidates = generate_candidates(engine, include_hold=cfg.beam_cfg.include_hold)
        selected_index = next(idx for idx, candidate in enumerate(candidates) if int(candidate["candidate_index"]) == int(selected["candidate_index"]))
        visits = np.zeros((len(candidates),), dtype=np.int32)
        visits[selected_index] = 1
        return {
            "selected_index": selected_index,
            "selected_candidate": candidates[selected_index],
            "visits": visits,
            "priors": np.eye(1, len(candidates), selected_index, dtype=np.float32).reshape(-1),
            "q_values": np.zeros((len(candidates),), dtype=np.float32),
        }

    if candidates is None:
        candidates = generate_candidates(engine, include_hold=cfg.beam_cfg.include_hold)
    if not candidates:
        raise RuntimeError("MCTS requires at least one candidate.")

    root_state = SearchState(
        active_engine=engine.clone(),
        passive_engine=None if opponent_engine is None else opponent_engine.clone(),
        move_number=int(move_number),
        root_to_move=True,
        active_known_next=max(0, int(cfg.visible_queue_depth)),
        passive_known_next=None if opponent_engine is None else max(0, int(cfg.visible_queue_depth)),
    )

    rng = np.random.default_rng()
    nodes: dict[tuple[Any, ...], TreeNode] = {}
    for sim_index in range(max(1, int(cfg.simulations))):
        _simulate(
            root_state,
            nodes,
            model,
            cfg,
            rng,
            depth=0,
            root_candidates=candidates if sim_index == 0 else None,
        )

    root_node = nodes.get(_state_key(root_state))
    if root_node is None:
        root_node = _create_node(root_state, model, cfg, root_candidates=candidates)
        nodes[_state_key(root_state)] = root_node

    visits = root_node.visits
    q_values = root_node.value_sums / np.maximum(root_node.visits, 1)
    if int(visits.sum()) <= 0:
        selected_index = int(np.argmax(root_node.priors))
    else:
        selected_index = select_action_from_visits(visits, cfg.temperature, rng=rng)
    return {
        "selected_index": int(selected_index),
        "selected_candidate": root_node.candidates[selected_index],
        "visits": visits.copy(),
        "priors": root_node.priors.copy(),
        "q_values": q_values.astype(np.float32, copy=False),
    }
