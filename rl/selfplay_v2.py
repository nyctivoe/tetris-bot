from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from beam_search import BeamSearchConfig, beam_search_select, generate_candidates
from features import build_model_inputs
from model_v2 import TetrisZeroNet, load_checkpoint, resolve_module_device
from pvp_game import BeamAgent, PvpGameConfig, advance_turn_start, run_pvp_turn
from rl.replay_buffer import init_buffer_dir, load_manifest, write_manifest, write_replay_shard
from rl.schemas import TRAINING_SAMPLE_VERSION, TrainingSampleV1, validate_training_sample
from tetrisEngine import TetrisEngine


@dataclass(frozen=True)
class EpisodeConfig:
    max_plies: int = 64
    beam_cfg: BeamSearchConfig = BeamSearchConfig()
    spin_mode: str = "all_spin"
    garbage_lines: int = 2
    garbage_timer: int = 1
    garbage_rate: float = 1.0


def _make_engine(seed: int | None, spin_mode: str = "all_spin") -> TetrisEngine:
    engine = TetrisEngine(spin_mode=spin_mode, rng=np.random.default_rng(seed))
    engine.spawn_next(allow_clutch=True)
    return engine


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    if temperature <= 1.0e-6:
        out = np.zeros_like(values, dtype=np.float32)
        out[int(np.argmax(values))] = 1.0
        return out
    shifted = values.astype(np.float32) / float(temperature)
    shifted = shifted - np.max(shifted)
    exp = np.exp(shifted)
    return (exp / exp.sum()).astype(np.float32, copy=False)


def serialize_training_sample(
    *,
    board_tensor: np.ndarray,
    piece_ids: np.ndarray,
    context_scalars: np.ndarray,
    candidate_features: np.ndarray,
    policy_target: np.ndarray,
    value_target: float,
    attack_target: float,
    survival_target: float,
    surge_target: float,
    metadata: dict[str, Any],
) -> TrainingSampleV1:
    return validate_training_sample(
        {
            "version": TRAINING_SAMPLE_VERSION,
            "board_tensor": np.asarray(board_tensor, dtype=np.float32),
            "piece_ids": np.asarray(piece_ids, dtype=np.int64),
            "context_scalars": np.asarray(context_scalars, dtype=np.float32),
            "candidate_features": np.asarray(candidate_features, dtype=np.float32),
            "candidate_count": int(candidate_features.shape[0]),
            "policy_target": np.asarray(policy_target, dtype=np.float32),
            "value_target": float(value_target),
            "attack_target": float(attack_target),
            "survival_target": float(survival_target),
            "surge_target": float(surge_target),
            "metadata": dict(metadata),
        }
    )


def _model_policy_logits(
    model: TetrisZeroNet,
    engine: TetrisEngine,
    candidates: list[dict[str, Any]],
    opponent_engine: TetrisEngine | None,
    move_number: int,
) -> np.ndarray:
    inputs = build_model_inputs(engine, candidates, opponent_engine=opponent_engine, move_number=move_number)
    device = resolve_module_device(model)
    board = torch.from_numpy(inputs["board_tensor"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    pieces = torch.from_numpy(inputs["piece_ids"]).unsqueeze(0).to(device=device, dtype=torch.long)
    context = torch.from_numpy(inputs["context_scalars"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    candidate_features = torch.from_numpy(inputs["candidate_features"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    candidate_mask = torch.from_numpy(inputs["candidate_mask"]).unsqueeze(0).to(device=device, dtype=torch.bool)
    with torch.no_grad():
        outputs = model(board, pieces, context, candidate_features, candidate_mask)
    return outputs["policy_logits"].squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _select_candidate(
    engine: TetrisEngine,
    opponent_engine: TetrisEngine | None,
    move_number: int,
    beam_cfg: BeamSearchConfig,
    *,
    model: TetrisZeroNet | None = None,
) -> tuple[list[dict[str, Any]], int, np.ndarray]:
    candidates = generate_candidates(engine, include_hold=beam_cfg.include_hold)
    if not candidates:
        return [], -1, np.zeros((0,), dtype=np.float32)
    if model is None:
        policy_target = np.zeros((len(candidates),), dtype=np.float32)
        selected = beam_search_select(engine, opponent_engine=opponent_engine, move_number=move_number, cfg=beam_cfg)
        chosen_index = next(
            idx for idx, candidate in enumerate(candidates)
            if int(candidate["candidate_index"]) == int(selected["candidate_index"])
        )
        policy_target[chosen_index] = 1.0
        return candidates, chosen_index, policy_target
    logits = _model_policy_logits(model, engine, candidates, opponent_engine, move_number)
    policy_target = _softmax(logits, temperature=1.0)
    return candidates, int(np.argmax(logits)), policy_target


def _record_step(
    engine: TetrisEngine,
    candidates: list[dict[str, Any]],
    chosen_index: int,
    policy_target: np.ndarray,
    move_number: int,
    *,
    opponent_engine: TetrisEngine | None = None,
) -> TrainingSampleV1:
    inputs = build_model_inputs(engine, candidates, opponent_engine=opponent_engine, move_number=move_number)
    chosen = candidates[chosen_index]
    stats = dict(chosen.get("stats") or {})
    return serialize_training_sample(
        board_tensor=inputs["board_tensor"],
        piece_ids=inputs["piece_ids"],
        context_scalars=inputs["context_scalars"],
        candidate_features=inputs["candidate_features"],
        policy_target=policy_target,
        value_target=0.0,
        attack_target=float(stats.get("attack", 0) or 0),
        survival_target=1.0,
        surge_target=float(stats.get("surge_send", 0) or 0),
        metadata={
            "move_number": int(move_number),
            "chosen_index": int(chosen_index),
            "used_hold": bool(chosen.get("used_hold")),
            "b2b_chain": int(engine.b2b_chain),
            "pieces_placed": int(engine.pieces_placed),
        },
    )


def _assign_outcome(samples: list[TrainingSampleV1], value_target: float, survived: bool) -> list[TrainingSampleV1]:
    output: list[TrainingSampleV1] = []
    for sample in samples:
        payload = dict(sample)
        payload["value_target"] = float(value_target)
        payload["survival_target"] = 1.0 if survived else 0.0
        output.append(validate_training_sample(payload))
    return output


def generate_singleplayer_episode(
    *,
    max_plies: int = 32,
    seed: int = 0,
    beam_cfg: BeamSearchConfig | None = None,
    model: TetrisZeroNet | None = None,
) -> dict[str, Any]:
    beam_cfg = beam_cfg or BeamSearchConfig()
    engine = _make_engine(seed, spin_mode="all_spin")
    samples: list[TrainingSampleV1] = []

    for move_number in range(1, int(max_plies) + 1):
        if engine.game_over or engine.current_piece is None:
            break
        candidates, chosen_index, policy_target = _select_candidate(engine, None, move_number, beam_cfg, model=model)
        if not candidates:
            break
        samples.append(_record_step(engine, candidates, chosen_index, policy_target, move_number))
        chosen = candidates[chosen_index]
        engine = chosen["engine_after"]

    survived = not engine.game_over and len(samples) >= int(max_plies)
    value_target = 1.0 if survived else -1.0
    return {
        "mode": "singleplayer",
        "seed": int(seed),
        "samples": _assign_outcome(samples, value_target, survived),
        "termination": "max_plies" if survived else (engine.game_over_reason or "no_moves"),
    }


def generate_garbage_pressure_episode(
    *,
    max_plies: int = 32,
    seed: int = 0,
    beam_cfg: BeamSearchConfig | None = None,
    model: TetrisZeroNet | None = None,
    garbage_lines: int = 2,
    garbage_timer: int = 1,
    garbage_rate: float = 1.0,
) -> dict[str, Any]:
    beam_cfg = beam_cfg or BeamSearchConfig()
    rng = np.random.default_rng(seed + 1)
    engine = _make_engine(seed, spin_mode="all_spin")
    samples: list[TrainingSampleV1] = []

    for move_number in range(1, int(max_plies) + 1):
        if engine.game_over or engine.current_piece is None:
            break
        if rng.random() <= float(garbage_rate):
            engine.add_incoming_garbage(lines=int(garbage_lines), timer=int(garbage_timer))
        landed = engine.tick_garbage()
        if landed and engine.current_piece is not None:
            try:
                valid = bool(engine.is_position_valid(engine.current_piece, engine.current_piece.position, engine.current_piece.rotation))
            except TypeError:
                valid = bool(engine.is_position_valid(engine.current_piece, position=engine.current_piece.position))
            if not valid:
                engine.current_piece = None
                engine.game_over = True
                engine.game_over_reason = "garbage_top_out"
                break
        candidates, chosen_index, policy_target = _select_candidate(engine, None, move_number, beam_cfg, model=model)
        if not candidates:
            break
        samples.append(_record_step(engine, candidates, chosen_index, policy_target, move_number))
        engine = candidates[chosen_index]["engine_after"]

    survived = not engine.game_over and len(samples) >= int(max_plies)
    value_target = 1.0 if survived else -1.0
    return {
        "mode": "garbage_pressure",
        "seed": int(seed),
        "samples": _assign_outcome(samples, value_target, survived),
        "termination": "max_plies" if survived else (engine.game_over_reason or "no_moves"),
    }


def generate_pvp_episode(
    *,
    max_plies: int = 32,
    seed: int = 0,
    beam_cfg: BeamSearchConfig | None = None,
    model: TetrisZeroNet | None = None,
) -> dict[str, Any]:
    beam_cfg = beam_cfg or BeamSearchConfig()
    cfg = PvpGameConfig(max_plies=max_plies, beam_cfg=beam_cfg)
    rng = np.random.default_rng(seed)
    engine_a = _make_engine(int(rng.integers(0, 2**31 - 1)), spin_mode="all_spin")
    engine_b = _make_engine(int(rng.integers(0, 2**31 - 1)), spin_mode="all_spin")
    samples_a: list[TrainingSampleV1] = []
    samples_b: list[TrainingSampleV1] = []
    turns: list[dict[str, Any]] = []
    agents = {
        "a": BeamAgent(cfg=beam_cfg),
        "b": BeamAgent(cfg=beam_cfg),
    }

    for ply in range(1, int(max_plies) + 1):
        for name, active, passive in (("a", engine_a, engine_b), ("b", engine_b, engine_a)):
            if active.game_over or passive.game_over:
                break
            planning_active = active.clone()
            planning_passive = passive.clone()
            preview = advance_turn_start(planning_active)
            if preview["terminated"]:
                turns.append(run_pvp_turn(active, passive, {}, ply, cfg))
                break
            candidates, chosen_index, policy_target = _select_candidate(planning_active, planning_passive, ply, beam_cfg, model=model)
            if not candidates:
                active.game_over = True
                active.game_over_reason = "no_moves"
                break
            sample = _record_step(planning_active, candidates, chosen_index, policy_target, ply, opponent_engine=planning_passive)
            if name == "a":
                samples_a.append(sample)
            else:
                samples_b.append(sample)
            action = dict(candidates[chosen_index]["placement"])
            turns.append(run_pvp_turn(active, passive, action, ply, cfg))
            if active.game_over:
                break
        if engine_a.game_over or engine_b.game_over:
            break

    if engine_a.game_over == engine_b.game_over:
        winner = None
    else:
        winner = "b" if engine_a.game_over else "a"
    value_a = 0.0 if winner is None else (1.0 if winner == "a" else -1.0)
    value_b = 0.0 if winner is None else (1.0 if winner == "b" else -1.0)
    out_a = _assign_outcome(samples_a, value_a, not engine_a.game_over)
    out_b = _assign_outcome(samples_b, value_b, not engine_b.game_over)
    return {
        "mode": "pvp",
        "seed": int(seed),
        "winner": winner,
        "player_a_samples": out_a,
        "player_b_samples": out_b,
        "turns": turns,
    }


def _next_shard_index(output_dir: str | Path) -> int:
    out = Path(output_dir)
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return 0
    manifest = load_manifest(out)
    shard_indices: list[int] = []
    for shard_name in list(manifest.get("shards") or []):
        stem = Path(str(shard_name)).stem
        if not stem.startswith("shard_"):
            continue
        try:
            shard_indices.append(int(stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (max(shard_indices) + 1) if shard_indices else 0


def _collect_episode_samples(mode: str, episode: dict[str, Any]) -> list[TrainingSampleV1]:
    if mode == "pvp":
        return list(episode.get("player_a_samples") or []) + list(episode.get("player_b_samples") or [])
    return list(episode.get("samples") or [])


def _generate_episode(mode: str, seed: int, max_plies: int, beam_cfg: BeamSearchConfig, args: argparse.Namespace) -> dict[str, Any]:
    if mode == "singleplayer":
        return generate_singleplayer_episode(max_plies=max_plies, seed=seed, beam_cfg=beam_cfg, model=getattr(args, "_loaded_model", None))
    if mode == "garbage_pressure":
        return generate_garbage_pressure_episode(
            max_plies=max_plies,
            seed=seed,
            beam_cfg=beam_cfg,
            model=getattr(args, "_loaded_model", None),
            garbage_lines=int(args.garbage_lines),
            garbage_timer=int(args.garbage_timer),
            garbage_rate=float(args.garbage_rate),
        )
    if mode == "pvp":
        return generate_pvp_episode(max_plies=max_plies, seed=seed, beam_cfg=beam_cfg, model=getattr(args, "_loaded_model", None))
    raise RuntimeError(f"Unsupported self-play mode: {mode!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate TetrisZero self-play replay shards.")
    parser.add_argument("--mode", choices=("singleplayer", "garbage_pressure", "pvp"), default="pvp")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--garbage-lines", type=int, default=2)
    parser.add_argument("--garbage-timer", type=int, default=1)
    parser.add_argument("--garbage-rate", type=float, default=1.0)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    beam_cfg = BeamSearchConfig(depth=int(args.depth), width=int(args.width))
    output_dir = Path(args.output_dir)
    init_buffer_dir(output_dir, append=bool(args.append))
    manifest = {"buffer_version": 1, "shards": []} if not (output_dir / "manifest.json").exists() else load_manifest(output_dir)
    args._loaded_model = None
    if args.model_path:
        args._loaded_model = load_checkpoint(args.model_path, device=args.device)["model"]

    all_samples: list[TrainingSampleV1] = []
    episode_summaries: list[str] = []
    for episode_index in range(int(args.episodes)):
        episode_seed = int(args.seed) + episode_index
        episode = _generate_episode(str(args.mode), episode_seed, int(args.max_plies), beam_cfg, args)
        samples = _collect_episode_samples(str(args.mode), episode)
        all_samples.extend(samples)
        termination = episode.get("termination", episode.get("winner"))
        episode_summaries.append(f"{episode_index}:{termination}:{len(samples)}")

    if not all_samples:
        raise RuntimeError("Self-play generation produced no samples.")

    shard_index = _next_shard_index(output_dir)
    shard_name = write_replay_shard(
        output_dir,
        shard_index,
        all_samples,
        {
            "mode": str(args.mode),
            "episodes": int(args.episodes),
            "max_plies": int(args.max_plies),
            "seed": int(args.seed),
            "beam_depth": int(args.depth),
            "beam_width": int(args.width),
            "model_path": args.model_path,
        },
    )
    shards = list(manifest.get("shards") or [])
    shards.append(shard_name)
    write_manifest(output_dir, {"buffer_version": 1, "shards": shards})
    print(
        f"mode={args.mode} episodes={int(args.episodes)} samples={len(all_samples)} "
        f"shard={shard_name} output_dir={output_dir} episode_summaries={'|'.join(episode_summaries)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
