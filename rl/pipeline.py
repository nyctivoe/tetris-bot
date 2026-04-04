from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from beam_search import BeamSearchConfig
from mcts import MctsConfig, run_mcts
from model_v2 import load_checkpoint
from pvp_game import BeamAgent, PvpGameConfig
from rl.arena import compare_agents
from rl.replay_buffer import init_buffer_dir, load_manifest, write_manifest, write_replay_shard
from rl.selfplay_v2 import _collect_episode_samples, _next_shard_index, generate_garbage_pressure_episode, generate_pvp_episode, generate_singleplayer_episode
from rl.training import TrainingConfig, train_model, load_samples_from_buffer
from beam_search import beam_search_select, generate_candidates


class PipelineSearchAgent:
    def __init__(self, model, *, beam_cfg: BeamSearchConfig, simulations: int, max_depth: int, visible_queue_depth: int, c_puct: float):
        self.model = model
        self.beam_cfg = beam_cfg
        self.mcts_cfg = MctsConfig(
            enabled=model is not None,
            simulations=int(simulations),
            c_puct=float(c_puct),
            temperature=0.0,
            beam_cfg=beam_cfg,
            max_depth=int(max_depth),
            visible_queue_depth=int(visible_queue_depth),
        )

    def select_action(self, engine, opponent_engine, move_number):
        if self.model is None:
            return dict(
                beam_search_select(
                    engine,
                    opponent_engine=opponent_engine,
                    move_number=move_number,
                    cfg=self.beam_cfg,
                )["placement"]
            )
        candidates = generate_candidates(engine, include_hold=self.beam_cfg.include_hold)
        if not candidates:
            raise RuntimeError("PipelineSearchAgent found no legal candidates.")
        result = run_mcts(engine, opponent_engine, move_number, self.model, candidates, cfg=self.mcts_cfg)
        return dict(result["selected_candidate"]["placement"])


def _append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_eval_model(checkpoint_path: str | None, device: str):
    if checkpoint_path is None:
        return None
    bundle = load_checkpoint(checkpoint_path, device=device)
    model = bundle["model"]
    model.eval()
    return model


def _run_eval_pair(
    *,
    current_checkpoint: str,
    opponent_checkpoint: str | None,
    games: int,
    seed: int,
    max_plies: int,
    beam_cfg: BeamSearchConfig,
    device: str,
    simulations: int,
    max_depth: int,
    visible_queue_depth: int,
    c_puct: float,
) -> dict[str, Any]:
    current_model = _load_eval_model(current_checkpoint, device)
    opponent_model = _load_eval_model(opponent_checkpoint, device)
    current_agent = PipelineSearchAgent(
        current_model,
        beam_cfg=beam_cfg,
        simulations=simulations,
        max_depth=max_depth,
        visible_queue_depth=visible_queue_depth,
        c_puct=c_puct,
    )
    if opponent_checkpoint is None:
        opponent_agent = BeamAgent(cfg=beam_cfg)
        label = "vs_beam"
    else:
        opponent_agent = PipelineSearchAgent(
            opponent_model,
            beam_cfg=beam_cfg,
            simulations=simulations,
            max_depth=max_depth,
            visible_queue_depth=visible_queue_depth,
            c_puct=c_puct,
        )
        label = "vs_previous"
    result = compare_agents(
        current_agent,
        opponent_agent,
        games=int(games),
        seed=int(seed),
        cfg=PvpGameConfig(max_plies=int(max_plies), beam_cfg=beam_cfg),
    )
    return {
        "label": label,
        "metrics": dict(result["metrics"]),
        "games": int(games),
        "seed": int(seed),
    }


def _generate_cycle_episode(
    mode: str,
    *,
    seed: int,
    max_plies: int,
    beam_cfg: BeamSearchConfig,
    model,
    garbage_lines: int,
    garbage_timer: int,
    garbage_rate: float,
) -> dict[str, Any]:
    if mode == "singleplayer":
        return generate_singleplayer_episode(max_plies=max_plies, seed=seed, beam_cfg=beam_cfg, model=model)
    if mode == "garbage_pressure":
        return generate_garbage_pressure_episode(
            max_plies=max_plies,
            seed=seed,
            beam_cfg=beam_cfg,
            model=model,
            garbage_lines=garbage_lines,
            garbage_timer=garbage_timer,
            garbage_rate=garbage_rate,
        )
    if mode == "pvp":
        return generate_pvp_episode(max_plies=max_plies, seed=seed, beam_cfg=beam_cfg, model=model)
    raise RuntimeError(f"Unsupported mode: {mode!r}")


def run_pipeline(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run repeated self-play generation and training in one command.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--mode", choices=("singleplayer", "garbage_pressure", "pvp"), default="pvp")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--episodes-per-cycle", type=int, default=50)
    parser.add_argument("--max-plies", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs-per-cycle", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--garbage-lines", type=int, default=2)
    parser.add_argument("--garbage-timer", type=int, default=1)
    parser.add_argument("--garbage-rate", type=float, default=1.0)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--selfplay-source", choices=("beam", "model"), default="beam")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-games", type=int, default=0)
    parser.add_argument("--eval-max-plies", type=int, default=32)
    parser.add_argument("--eval-simulations", type=int, default=16)
    parser.add_argument("--eval-max-depth", type=int, default=2)
    parser.add_argument("--log-csv-name", type=str, default="cycle_metrics.csv")
    parser.add_argument("--history-csv-name", type=str, default="train_history.csv")
    parser.add_argument("--events-jsonl-name", type=str, default="events.jsonl")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    replay_dir = run_dir / "replay_buffer"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    replay_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    init_buffer_dir(replay_dir, append=True)
    manifest = {"buffer_version": 1, "shards": []} if not (replay_dir / "manifest.json").exists() else load_manifest(replay_dir)
    cycle_csv_path = log_dir / str(args.log_csv_name)
    history_csv_path = log_dir / str(args.history_csv_name)
    events_jsonl_path = log_dir / str(args.events_jsonl_name)
    run_config_path = run_dir / "run_config.json"
    status_path = run_dir / "status.json"
    latest_checkpoint_path = checkpoint_dir / "latest.json"
    eval_summary_path = reports_dir / "latest_eval.json"
    cycle_fieldnames = [
        "cycle",
        "samples_generated",
        "cumulative_samples",
        "shard_name",
        "checkpoint_path",
        "val_total",
        "val_policy",
        "val_value",
        "val_attack",
        "val_survival",
        "val_surge",
        "eval_vs_previous_win_rate",
        "eval_vs_previous_cancel_efficiency",
        "eval_vs_previous_difficult_clear_fraction",
        "eval_vs_beam_win_rate",
        "eval_vs_beam_cancel_efficiency",
        "eval_vs_beam_difficult_clear_fraction",
    ]
    _write_json(
        run_config_path,
        {
            "mode": str(args.mode),
            "cycles": int(args.cycles),
            "episodes_per_cycle": int(args.episodes_per_cycle),
            "max_plies": int(args.max_plies),
            "epochs_per_cycle": int(args.epochs_per_cycle),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "device": str(args.device),
            "depth": int(args.depth),
            "width": int(args.width),
            "garbage_lines": int(args.garbage_lines),
            "garbage_timer": int(args.garbage_timer),
            "garbage_rate": float(args.garbage_rate),
            "init_checkpoint": args.init_checkpoint,
            "selfplay_source": str(args.selfplay_source),
            "eval_every": int(args.eval_every),
            "eval_games": int(args.eval_games),
            "eval_max_plies": int(args.eval_max_plies),
            "eval_simulations": int(args.eval_simulations),
            "eval_max_depth": int(args.eval_max_depth),
        },
    )
    _write_json(
        status_path,
        {
            "state": "initialized",
            "completed_cycles": 0,
            "current_checkpoint": args.init_checkpoint,
            "replay_manifest": str(replay_dir / "manifest.json"),
            "latest_cycle_metrics": None,
        },
    )

    current_model = None
    optimizer_state_dict = None
    current_checkpoint = args.init_checkpoint
    previous_checkpoint = args.init_checkpoint
    if current_checkpoint:
        bundle = load_checkpoint(current_checkpoint, device=args.device)
        current_model = bundle["model"]
        current_model.eval()
        optimizer_state_dict = bundle.get("optimizer_state_dict")

    beam_cfg = BeamSearchConfig(depth=int(args.depth), width=int(args.width))
    seed_cursor = int(args.seed)
    for cycle in range(1, int(args.cycles) + 1):
        use_model = current_model if args.selfplay_source == "model" else None
        print(
            f"[cycle {cycle}] selfplay mode={args.mode} source={'model' if use_model is not None else 'beam'} "
            f"episodes={int(args.episodes_per_cycle)} max_plies={int(args.max_plies)}",
            flush=True,
        )
        all_samples = []
        terminations: list[str] = []
        for episode_index in range(int(args.episodes_per_cycle)):
            episode = _generate_cycle_episode(
                str(args.mode),
                seed=seed_cursor,
                max_plies=int(args.max_plies),
                beam_cfg=beam_cfg,
                model=use_model,
                garbage_lines=int(args.garbage_lines),
                garbage_timer=int(args.garbage_timer),
                garbage_rate=float(args.garbage_rate),
            )
            seed_cursor += 1
            samples = _collect_episode_samples(str(args.mode), episode)
            all_samples.extend(samples)
            terminations.append(str(episode.get("termination", episode.get("winner"))))
            print(
                f"[cycle {cycle}] episode {episode_index + 1}/{int(args.episodes_per_cycle)} "
                f"samples={len(samples)} termination={terminations[-1]}",
                flush=True,
            )

        if not all_samples:
            raise RuntimeError(f"Cycle {cycle} produced no training samples.")

        shard_index = _next_shard_index(replay_dir)
        shard_name = write_replay_shard(
            replay_dir,
            shard_index,
            all_samples,
            {
                "cycle": cycle,
                "mode": str(args.mode),
                "episodes": int(args.episodes_per_cycle),
                "max_plies": int(args.max_plies),
                "seed_start": int(seed_cursor - int(args.episodes_per_cycle)),
                "seed_end": int(seed_cursor - 1),
                "selfplay_source": str(args.selfplay_source),
                "init_checkpoint": current_checkpoint,
            },
        )
        shards = list(manifest.get("shards") or [])
        shards.append(shard_name)
        manifest = {"buffer_version": 1, "shards": shards}
        write_manifest(replay_dir, manifest)
        print(f"[cycle {cycle}] wrote shard={shard_name} cumulative_shards={len(shards)} samples={len(all_samples)}", flush=True)

        samples = load_samples_from_buffer(replay_dir)
        checkpoint_name = f"tetriszero_cycle_{cycle:04d}.pt"
        result = train_model(
            samples,
            cfg=TrainingConfig(
                batch_size=int(args.batch_size),
                epochs=int(args.epochs_per_cycle),
                lr=float(args.lr),
                device=str(args.device),
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_name=checkpoint_name,
            ),
            model=current_model,
            optimizer_state_dict=optimizer_state_dict,
            checkpoint_metadata={
                "pipeline_cycle": cycle,
                "mode": str(args.mode),
                "episodes_per_cycle": int(args.episodes_per_cycle),
                "max_plies": int(args.max_plies),
                "buffer_dir": str(replay_dir),
                "selfplay_source": str(args.selfplay_source),
            },
        )
        current_model = result["model"]
        current_model.eval()
        optimizer_state_dict = result["optimizer"].state_dict()
        previous_checkpoint = current_checkpoint
        current_checkpoint = str(result["checkpoint_path"])
        train_history = list(result["history"])
        for step_index, step_metrics in enumerate(train_history, start=1):
            _append_csv_row(
                history_csv_path,
                ["cycle", "step", "total", "policy", "value", "attack", "survival", "surge"],
                {
                    "cycle": cycle,
                    "step": step_index,
                    **step_metrics,
                },
            )
        print(
            f"[cycle {cycle}] trained_steps={len(result['history'])} "
            f"val_total={result['validation']['val_total']:.4f} "
            f"checkpoint={result['checkpoint_path']}",
            flush=True,
        )
        cycle_row = {
            "cycle": cycle,
            "samples_generated": len(all_samples),
            "cumulative_samples": len(samples),
            "shard_name": shard_name,
            "checkpoint_path": current_checkpoint,
            "val_total": float(result["validation"]["val_total"]),
            "val_policy": float(result["validation"]["val_policy"]),
            "val_value": float(result["validation"]["val_value"]),
            "val_attack": float(result["validation"]["val_attack"]),
            "val_survival": float(result["validation"]["val_survival"]),
            "val_surge": float(result["validation"]["val_surge"]),
        }
        _append_jsonl(
            events_jsonl_path,
            {
                "event": "train_cycle",
                "cycle": cycle,
                "samples_generated": len(all_samples),
                "cumulative_samples": len(samples),
                "checkpoint_path": current_checkpoint,
                "validation": dict(result["validation"]),
                "history_steps": len(train_history),
            },
        )
        _write_json(
            latest_checkpoint_path,
            {
                "cycle": cycle,
                "checkpoint_path": current_checkpoint,
            },
        )

        should_eval = int(args.eval_games) > 0 and int(args.eval_every) > 0 and cycle % int(args.eval_every) == 0
        if should_eval:
            eval_seed = int(args.seed) + 100000 + cycle * 100
            eval_results: list[dict[str, Any]] = []
            if previous_checkpoint is not None and previous_checkpoint != current_checkpoint:
                eval_results.append(
                    _run_eval_pair(
                        current_checkpoint=current_checkpoint,
                        opponent_checkpoint=previous_checkpoint,
                        games=int(args.eval_games),
                        seed=eval_seed,
                        max_plies=int(args.eval_max_plies),
                        beam_cfg=beam_cfg,
                        device=str(args.device),
                        simulations=int(args.eval_simulations),
                        max_depth=int(args.eval_max_depth),
                        visible_queue_depth=5,
                        c_puct=1.5,
                    )
                )
            eval_results.append(
                _run_eval_pair(
                    current_checkpoint=current_checkpoint,
                    opponent_checkpoint=None,
                    games=int(args.eval_games),
                    seed=eval_seed + 1,
                    max_plies=int(args.eval_max_plies),
                    beam_cfg=beam_cfg,
                    device=str(args.device),
                    simulations=int(args.eval_simulations),
                    max_depth=int(args.eval_max_depth),
                    visible_queue_depth=5,
                    c_puct=1.5,
                )
            )
            eval_updates: dict[str, Any] = {}
            for item in eval_results:
                label = str(item["label"])
                metrics = dict(item["metrics"])
                _append_jsonl(
                    events_jsonl_path,
                    {
                        "event": "arena_eval",
                        "cycle": cycle,
                        "label": label,
                        "metrics": metrics,
                        "games": int(item["games"]),
                        "seed": int(item["seed"]),
                        "checkpoint_path": current_checkpoint,
                        "opponent_checkpoint": previous_checkpoint if label == "vs_previous" else None,
                    },
                )
                print(
                    f"[cycle {cycle}] eval {label} win_rate={metrics['win_rate']:.3f} "
                    f"cancel_efficiency={metrics['cancel_efficiency']:.3f}",
                    flush=True,
                )
                if label == "vs_previous":
                    eval_updates["eval_vs_previous_win_rate"] = float(metrics["win_rate"])
                    eval_updates["eval_vs_previous_cancel_efficiency"] = float(metrics["cancel_efficiency"])
                    eval_updates["eval_vs_previous_difficult_clear_fraction"] = float(metrics["difficult_clear_fraction"])
                elif label == "vs_beam":
                    eval_updates["eval_vs_beam_win_rate"] = float(metrics["win_rate"])
                    eval_updates["eval_vs_beam_cancel_efficiency"] = float(metrics["cancel_efficiency"])
                    eval_updates["eval_vs_beam_difficult_clear_fraction"] = float(metrics["difficult_clear_fraction"])
            if eval_updates:
                cycle_row.update(eval_updates)
            _write_json(
                eval_summary_path,
                {
                    "cycle": cycle,
                    "checkpoint_path": current_checkpoint,
                    "results": eval_results,
                },
            )
        _append_csv_row(cycle_csv_path, cycle_fieldnames, cycle_row)
        _write_json(
            status_path,
            {
                "state": "running" if cycle < int(args.cycles) else "completed",
                "completed_cycles": cycle,
                "current_checkpoint": current_checkpoint,
                "replay_manifest": str(replay_dir / "manifest.json"),
                "latest_cycle_metrics": cycle_row,
            },
        )

    print(f"pipeline_complete final_checkpoint={current_checkpoint}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_pipeline(argv)


if __name__ == "__main__":
    raise SystemExit(main())
