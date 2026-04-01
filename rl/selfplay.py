from __future__ import annotations

import numpy as np

from rl.model_loader import LoadedModelBundle
from rl.search import (
    SELFPLAY_GAMMA,
    build_pre_state,
    build_queue_state,
    generate_action_candidates,
    score_action_candidates_with_lookahead,
    select_top_k_candidates,
    soft_policy_from_scores,
)
from tetrisEngine import TetrisEngine


SELFPLAY_N_STEP = 5


def create_selfplay_engine(spin_mode: str, seed: int | None = None) -> TetrisEngine:
    rng = np.random.default_rng(seed)
    engine = TetrisEngine(spin_mode=spin_mode, rng=rng)
    engine.spawn_next(allow_clutch=True)
    return engine


def _current_piece_valid(engine: TetrisEngine) -> bool:
    piece = getattr(engine, "current_piece", None)
    if piece is None:
        return True
    try:
        return bool(engine.is_position_valid(piece, piece.position, piece.rotation))
    except TypeError:
        return bool(engine.is_position_valid(piece, position=piece.position))


def _maybe_inject_synthetic_garbage(
    engine: TetrisEngine,
    rng: np.random.Generator,
    *,
    garbage_rate: float,
    garbage_min_lines: int,
    garbage_max_lines: int,
    garbage_timer: int,
) -> None:
    if garbage_rate <= 0.0:
        return
    if rng.random() >= float(garbage_rate):
        return
    lines = int(rng.integers(int(garbage_min_lines), int(garbage_max_lines) + 1))
    engine.add_incoming_garbage(lines=lines, timer=max(1, int(garbage_timer)))


def _advance_garbage_state(engine: TetrisEngine) -> None:
    if not getattr(engine, "incoming_garbage", None):
        return
    engine.tick_garbage()
    if not _current_piece_valid(engine):
        engine.current_piece = None
        engine.game_over = True
        engine.game_over_reason = "garbage_top_out"


def _populate_value_targets(steps: list[dict], gamma: float, n_step: int) -> None:
    running = 0.0
    for idx in range(len(steps) - 1, -1, -1):
        running = float(steps[idx]["immediate_reward"]) + float(gamma) * running
        steps[idx]["value_target"] = float(running)

    for idx in range(len(steps)):
        ret = 0.0
        discount = 1.0
        for j in range(idx, min(len(steps), idx + int(n_step))):
            ret += discount * float(steps[j]["immediate_reward"])
            discount *= float(gamma)
        steps[idx]["n_step_value_target"] = float(ret)


def run_single_selfplay_game(
    model_bundle: LoadedModelBundle,
    game_id: int,
    max_plies: int,
    beam_width: int,
    top_k: int,
    temperature: float,
    spin_mode: str = "all_spin",
    rank_q_alpha: float | None = None,
    seed: int | None = None,
    garbage_rate: float = 0.0,
    garbage_min_lines: int = 1,
    garbage_max_lines: int = 4,
    garbage_timer: int = 1,
) -> dict:
    if max_plies <= 0:
        raise ValueError("max_plies must be positive.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive.")
    if garbage_min_lines <= 0 or garbage_max_lines < garbage_min_lines:
        raise ValueError("Invalid synthetic garbage range.")

    engine = create_selfplay_engine(spin_mode=spin_mode, seed=seed)
    rng = np.random.default_rng(None if seed is None else int(seed) + 1)

    steps = []
    termination = "max_plies"
    total_attack = 0.0
    total_reward = 0.0

    for ply in range(int(max_plies)):
        _maybe_inject_synthetic_garbage(
            engine,
            rng,
            garbage_rate=garbage_rate,
            garbage_min_lines=garbage_min_lines,
            garbage_max_lines=garbage_max_lines,
            garbage_timer=garbage_timer,
        )
        _advance_garbage_state(engine)

        if engine.current_piece is None:
            termination = "invalid_no_moves"
            break
        if engine.game_over:
            termination = "top_out"
            break

        root_move_number = ply + 1
        base_board = engine.board.astype(np.uint8, copy=True)
        root_queue_state = build_queue_state(engine)
        root_pre_state = build_pre_state(engine)

        candidates = generate_action_candidates(engine, include_hold=True)
        if not candidates:
            termination = "invalid_no_moves"
            break

        raw_scores = score_action_candidates_with_lookahead(
            model_bundle,
            engine,
            candidates,
            move_number=root_move_number,
            beam_width=beam_width,
            rank_q_alpha=rank_q_alpha,
            gamma=SELFPLAY_GAMMA,
        )
        candidate_indices, retained_scores = select_top_k_candidates(
            candidates,
            raw_scores,
            top_k=top_k,
        )
        retained_results = [candidates[i] for i in candidate_indices]
        policy_target = soft_policy_from_scores(retained_scores, temperature=temperature)
        chosen_index = int(rng.choice(len(candidate_indices), p=policy_target))
        chosen_result = retained_results[chosen_index]

        total_attack += float(chosen_result["immediate_attack"])
        total_reward += float(chosen_result["immediate_reward"])

        retained_boards = np.stack([r["board"] for r in retained_results]).astype(np.uint8)
        retained_placements = [dict(r["placement"]) for r in retained_results]
        steps.append(
            {
                "ply": ply,
                "base_board": base_board,
                "queue_state": root_queue_state,
                "pre_state": root_pre_state,
                "root_stats": {
                    "move_number": root_move_number,
                    "spin_mode": spin_mode,
                },
                "candidate_indices": [int(i) for i in candidate_indices],
                "bfs_boards": retained_boards,
                "bfs_placements": retained_placements,
                "policy_target": policy_target.astype(np.float32, copy=False),
                "chosen_index": chosen_index,
                "value_target": 0.0,
                "n_step_value_target": 0.0,
                "immediate_reward": float(chosen_result["immediate_reward"]),
                "immediate_attack": float(chosen_result["immediate_attack"]),
                "search_scores": retained_scores.astype(np.float32, copy=False),
                "search_details": {
                    "beam_width": int(beam_width),
                    "top_k": int(top_k),
                    "temperature": float(temperature),
                    "gamma": float(SELFPLAY_GAMMA),
                    "include_hold": True,
                },
            }
        )

        engine = chosen_result["engine_after"]
        if engine.game_over:
            termination = "top_out"
            break
    else:
        termination = "max_plies"

    _populate_value_targets(steps, gamma=SELFPLAY_GAMMA, n_step=SELFPLAY_N_STEP)

    return {
        "game_id": int(game_id),
        "seed": None if seed is None else int(seed),
        "termination": termination,
        "winner": None,
        "steps": steps,
        "stats": {
            "plies": len(steps),
            "total_attack": float(total_attack),
            "total_reward": float(total_reward),
        },
    }
