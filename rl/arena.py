from __future__ import annotations

from typing import Any, Callable

from pvp_game import PvpGameConfig, run_pvp_game
from rl.schemas import EVAL_METRICS_VERSION, EvalMetricsV1, validate_eval_metrics


def compute_eval_metrics(match_records: list[dict[str, Any]]) -> EvalMetricsV1:
    total_games = max(1, len(match_records))
    wins = sum(1 for match in match_records if match.get("winner") == "a")
    surge_events = 0
    total_turns = 0
    bad_breaks = 0
    difficult_clears = 0
    canceled = 0.0
    outgoing = 0.0
    pending = 0.0
    latency = 0.0

    for match in match_records:
        for turn in list(match.get("turns") or []):
            total_turns += 1
            stats = dict(turn.get("stats") or {})
            resolve = dict(turn.get("resolve") or {})
            surge_events += int((stats.get("surge_send") or 0) > 0)
            difficult_clears += int(bool(stats.get("is_difficult")))
            bad_breaks += int(bool(stats.get("breaks_b2b")) and int(stats.get("surge_send") or 0) == 0)
            canceled += float(resolve.get("canceled", 0) or 0)
            outgoing += float(resolve.get("outgoing_attack", 0) or 0)
            pending += float(dict(turn.get("active_pending_after") or {}).get("total_lines", 0) or 0)
            latency += float(turn.get("decision_latency", 0.0) or 0.0)

    metrics = {
        "version": EVAL_METRICS_VERSION,
        "win_rate": wins / float(total_games),
        "realized_surge_rate": surge_events / float(max(1, total_turns)),
        "unintentional_b2b_break_rate": bad_breaks / float(max(1, total_turns)),
        "difficult_clear_fraction": difficult_clears / float(max(1, total_turns)),
        "cancel_efficiency": canceled / float(max(1.0, outgoing)),
        "average_pending_garbage": pending / float(max(1, total_turns)),
        "mean_decision_latency": latency / float(max(1, total_turns)),
    }
    return validate_eval_metrics(metrics)


def run_match_set(
    player_a: Callable[..., dict[str, Any]] | Any,
    player_b: Callable[..., dict[str, Any]] | Any,
    *,
    games: int = 4,
    seed: int = 0,
    cfg: PvpGameConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = cfg or PvpGameConfig()
    return [
        run_pvp_game(player_a, player_b, seed=seed + game_idx, max_plies=cfg.max_plies, cfg=cfg)
        for game_idx in range(int(games))
    ]


def compare_agents(
    player_a: Callable[..., dict[str, Any]] | Any,
    player_b: Callable[..., dict[str, Any]] | Any,
    *,
    games: int = 4,
    seed: int = 0,
    cfg: PvpGameConfig | None = None,
) -> dict[str, Any]:
    matches = run_match_set(player_a, player_b, games=games, seed=seed, cfg=cfg)
    return {
        "matches": matches,
        "metrics": compute_eval_metrics(matches),
    }
