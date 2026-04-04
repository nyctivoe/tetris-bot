from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Protocol

import numpy as np

from beam_search import BeamSearchConfig, beam_search_select
from features import build_model_inputs
from tetrisEngine import TetrisEngine


class PlayerAgent(Protocol):
    def select_action(self, engine: TetrisEngine, opponent_engine: TetrisEngine, move_number: int) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class PvpGameConfig:
    max_plies: int = 400
    opening_cancel_pieces: int = 14
    base_garbage_timer: int = 60
    late_game_min_timer: int = 20
    garbage_frames_per_turn: int = 20
    decision_temperature: float = 0.0
    include_hold: bool = True
    beam_cfg: BeamSearchConfig = BeamSearchConfig()


@dataclass
class BeamAgent:
    cfg: BeamSearchConfig = BeamSearchConfig()

    def select_action(self, engine: TetrisEngine, opponent_engine: TetrisEngine, move_number: int) -> dict[str, Any]:
        candidate = beam_search_select(
            engine,
            opponent_engine=opponent_engine,
            move_number=move_number,
            cfg=self.cfg,
        )
        return dict(candidate["placement"])


def calculate_garbage_timer(ply: int, cfg: PvpGameConfig) -> int:
    frames_per_turn = max(1, int(cfg.garbage_frames_per_turn))
    if ply < 28:
        timer_frames = int(cfg.base_garbage_timer)
    elif ply < 100:
        timer_frames = int(max(cfg.base_garbage_timer - ply // 5, cfg.late_game_min_timer))
    else:
        timer_frames = int(cfg.late_game_min_timer)
    return int(max(1, math.ceil(timer_frames / float(frames_per_turn))))


def _current_piece_valid(engine: TetrisEngine) -> bool:
    piece = getattr(engine, "current_piece", None)
    if piece is None:
        return True
    try:
        return bool(engine.is_position_valid(piece, piece.position, piece.rotation))
    except TypeError:
        return bool(engine.is_position_valid(piece, position=piece.position))


def _pick_action(player: PlayerAgent | Callable[..., dict[str, Any]], engine: TetrisEngine, opponent_engine: TetrisEngine, move_number: int) -> dict[str, Any]:
    if hasattr(player, "select_action"):
        return dict(player.select_action(engine, opponent_engine, move_number))
    return dict(player(engine, opponent_engine, move_number))


def advance_turn_start(engine: TetrisEngine) -> dict[str, Any]:
    landed = engine.tick_garbage()
    if landed and not _current_piece_valid(engine):
        engine.current_piece = None
        engine.game_over = True
        engine.game_over_reason = "garbage_top_out"
    return {
        "landed_garbage": int(landed),
        "terminated": bool(engine.game_over),
        "reason": engine.game_over_reason,
    }


def build_observation(engine: TetrisEngine, opponent_engine: TetrisEngine | None, move_number: int) -> dict[str, np.ndarray]:
    return build_model_inputs(engine, candidates=[], opponent_engine=opponent_engine, move_number=move_number)


def run_pvp_turn(
    active_engine: TetrisEngine,
    passive_engine: TetrisEngine,
    action: dict[str, Any],
    move_number: int,
    cfg: PvpGameConfig,
) -> dict[str, Any]:
    turn_start = advance_turn_start(active_engine)
    if turn_start["terminated"]:
        return {
            "move_number": int(move_number),
            "landed_garbage": int(turn_start["landed_garbage"]),
            "terminated": True,
            "reason": active_engine.game_over_reason,
            "stats": None,
            "resolve": None,
        }

    result = active_engine.execute_placement(action, run_end_phase=True)
    stats = dict(result["stats"] or {})
    resolve = active_engine.resolve_outgoing_attack(
        int(result["attack"]),
        opener_phase=active_engine.pieces_placed <= int(cfg.opening_cancel_pieces),
    )
    sent = int(resolve["sent"])
    if sent > 0:
        passive_engine.add_incoming_garbage(
            lines=sent,
            timer=calculate_garbage_timer(move_number, cfg),
            col=None,
        )

    return {
        "move_number": int(move_number),
        "landed_garbage": int(turn_start["landed_garbage"]),
        "terminated": bool(active_engine.game_over),
        "reason": active_engine.game_over_reason,
        "action": dict(action),
        "stats": stats,
        "resolve": resolve,
        "sent_to_opponent": sent,
        "active_pending_after": active_engine.get_pending_garbage_summary(),
        "passive_pending_after": passive_engine.get_pending_garbage_summary(),
    }


def run_pvp_game(
    player_a: PlayerAgent | Callable[..., dict[str, Any]],
    player_b: PlayerAgent | Callable[..., dict[str, Any]],
    seed: int,
    max_plies: int | None = None,
    cfg: PvpGameConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or PvpGameConfig()
    if max_plies is None:
        max_plies = cfg.max_plies

    master_rng = np.random.default_rng(seed)
    engine_a = TetrisEngine(spin_mode="all_spin", rng=np.random.default_rng(int(master_rng.integers(0, 2**31 - 1))))
    engine_b = TetrisEngine(spin_mode="all_spin", rng=np.random.default_rng(int(master_rng.integers(0, 2**31 - 1))))
    engine_a.spawn_next(allow_clutch=True)
    engine_b.spawn_next(allow_clutch=True)

    turns: list[dict[str, Any]] = []
    winner: str | None = None
    termination = "max_plies"

    for ply in range(int(max_plies)):
        for name, player, active, passive in (
            ("a", player_a, engine_a, engine_b),
            ("b", player_b, engine_b, engine_a),
        ):
            if active.game_over or passive.game_over:
                break
            planning_active = active.clone()
            planning_passive = passive.clone()
            preview = advance_turn_start(planning_active)
            if preview["terminated"]:
                action = {}
                latency = 0.0
            else:
                start = time.perf_counter()
                action = _pick_action(player, planning_active, planning_passive, ply + 1)
                latency = time.perf_counter() - start
            turn = run_pvp_turn(active, passive, action, ply + 1, cfg)
            turn["player"] = name
            turn["decision_latency"] = float(latency)
            turns.append(turn)

            if active.game_over:
                winner = "b" if name == "a" else "a"
                termination = active.game_over_reason or "top_out"
                break
        if winner is not None or engine_a.game_over or engine_b.game_over:
            break

    if winner is None and engine_a.game_over != engine_b.game_over:
        winner = "b" if engine_a.game_over else "a"
        termination = engine_a.game_over_reason if engine_a.game_over else engine_b.game_over_reason

    return {
        "seed": int(seed),
        "winner": winner,
        "termination": termination,
        "turns": turns,
        "engine_a": engine_a,
        "engine_b": engine_b,
        "observations": {
            "a": build_observation(engine_a, engine_b, len(turns) + 1),
            "b": build_observation(engine_b, engine_a, len(turns) + 1),
        },
    }
