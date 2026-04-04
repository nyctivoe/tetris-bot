from __future__ import annotations

from pvp_game import BeamAgent, PvpGameConfig
from rl.arena import compare_agents


def test_compare_agents_returns_stable_metrics():
    result = compare_agents(BeamAgent(), BeamAgent(), games=2, seed=0, cfg=PvpGameConfig(max_plies=2))
    metrics = result["metrics"]

    assert len(result["matches"]) == 2
    assert metrics["version"] == 1
    assert 0.0 <= metrics["win_rate"] <= 1.0
    assert metrics["mean_decision_latency"] >= 0.0
