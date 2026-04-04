from __future__ import annotations

import math

from beam_search import generate_candidates
from pvp_game import BeamAgent, PvpGameConfig, calculate_garbage_timer, run_pvp_game, run_pvp_turn
from tetrisEngine import TetrisEngine


def test_run_pvp_game_is_deterministic_for_same_seed():
    cfg = PvpGameConfig(max_plies=2)
    first = run_pvp_game(BeamAgent(), BeamAgent(), seed=123, max_plies=2, cfg=cfg)
    second = run_pvp_game(BeamAgent(), BeamAgent(), seed=123, max_plies=2, cfg=cfg)

    assert first["winner"] == second["winner"]
    assert len(first["turns"]) == len(second["turns"])
    assert [turn["action"] for turn in first["turns"]] == [turn["action"] for turn in second["turns"]]


def test_run_pvp_turn_reports_garbage_top_out(monkeypatch, seeded_engine):
    opponent = seeded_engine.clone()

    def forced_tick():
        seeded_engine.game_over = True
        seeded_engine.game_over_reason = "garbage_top_out"
        return 1

    monkeypatch.setattr(seeded_engine, "tick_garbage", forced_tick)
    turn = run_pvp_turn(seeded_engine, opponent, {"x": 0, "y": 0, "rotation": 0}, move_number=1, cfg=PvpGameConfig())

    assert turn["terminated"] is True
    assert turn["reason"] == "garbage_top_out"


def test_match_record_contains_expected_fields(seeded_engine):
    candidate = generate_candidates(seeded_engine, include_hold=False)[0]
    active = seeded_engine.clone()
    passive = seeded_engine.clone()

    turn = run_pvp_turn(active, passive, dict(candidate["placement"]), move_number=1, cfg=PvpGameConfig(max_plies=2))

    assert "stats" in turn
    assert "resolve" in turn
    assert "active_pending_after" in turn
    assert "passive_pending_after" in turn


def test_run_pvp_game_exposes_decision_latency():
    game = run_pvp_game(BeamAgent(), BeamAgent(), seed=4, max_plies=1, cfg=PvpGameConfig(max_plies=1))

    assert game["turns"]
    assert "decision_latency" in game["turns"][0]
    assert game["turns"][0]["decision_latency"] >= 0.0


def test_calculate_garbage_timer_converts_frame_schedule_to_turns():
    cfg = PvpGameConfig(base_garbage_timer=60, late_game_min_timer=20, garbage_frames_per_turn=20)

    assert calculate_garbage_timer(1, cfg) == 3
    assert calculate_garbage_timer(60, cfg) == math.ceil(max(60 - 60 // 5, 20) / 20)
    assert calculate_garbage_timer(120, cfg) == 1


def test_run_pvp_turn_enqueues_scaled_garbage_timer(monkeypatch, seeded_engine):
    active = seeded_engine.clone()
    passive = seeded_engine.clone()
    cfg = PvpGameConfig(garbage_frames_per_turn=20)

    monkeypatch.setattr(
        active,
        "execute_placement",
        lambda action, run_end_phase=True: {
            "ok": True,
            "lines_cleared": 0,
            "stats": {},
            "end_phase": None,
            "attack": 2,
        },
    )
    monkeypatch.setattr(
        active,
        "resolve_outgoing_attack",
        lambda outgoing_attack, opener_phase=None: {
            "incoming_before": active.get_pending_garbage_summary(),
            "incoming_after": active.get_pending_garbage_summary(),
            "outgoing_attack": int(outgoing_attack),
            "canceled": 0,
            "sent": int(outgoing_attack),
            "used_opener_multiplier": False,
            "opener_phase": bool(opener_phase),
        },
    )

    turn = run_pvp_turn(active, passive, {"x": 0, "y": 0, "rotation": 0}, move_number=1, cfg=cfg)

    assert turn["sent_to_opponent"] == 2
    assert passive.incoming_garbage
    assert passive.incoming_garbage[0]["timer"] == 3


def test_run_pvp_game_selects_action_after_turn_start_garbage(monkeypatch):
    class TurnStartGarbageEngine(TetrisEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_incoming_garbage(lines=1, timer=1, col=0)

    class RecordingAgent:
        def __init__(self):
            self.pending_seen: list[int] = []
            self.bottom_row_seen: list[int] = []

        def select_action(self, engine, opponent_engine, move_number):
            del opponent_engine, move_number
            self.pending_seen.append(int(engine.get_pending_garbage_summary()["total_lines"]))
            self.bottom_row_seen.append(int(engine.board[-1, 1]))
            return dict(generate_candidates(engine, include_hold=False)[0]["placement"])

    agent_a = RecordingAgent()
    monkeypatch.setattr("pvp_game.TetrisEngine", TurnStartGarbageEngine)

    run_pvp_game(agent_a, BeamAgent(), seed=7, max_plies=1, cfg=PvpGameConfig(max_plies=1))

    assert agent_a.pending_seen == [0]
    assert agent_a.bottom_row_seen[0] != 0
