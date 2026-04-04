#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tetrisformer.rl.model_loader import load_tetrisformer_checkpoint
from tetrisformer.rl.selfplay import run_single_selfplay_game
from tetrisformer.rl.selfplay_buffer import (
    SELFPLAY_BUFFER_VERSION,
    SelfPlayManifest,
    init_buffer_dir,
    load_manifest,
    write_game_shard,
    write_manifest,
)


def _validate_existing_manifest(manifest: dict, *, spin_mode: str, checkpoint_path: str) -> None:
    if int(manifest.get("buffer_version", 0)) != SELFPLAY_BUFFER_VERSION:
        raise RuntimeError(
            f"Incompatible self-play buffer version: expected {SELFPLAY_BUFFER_VERSION}, "
            f"found {manifest.get('buffer_version')}."
        )
    if str(manifest.get("spin_mode")) != str(spin_mode):
        raise RuntimeError(
            f"Cannot append with spin_mode={spin_mode!r}; existing manifest uses {manifest.get('spin_mode')!r}."
        )
    existing_checkpoint = str(manifest.get("checkpoint_path", ""))
    if existing_checkpoint and existing_checkpoint != checkpoint_path:
        raise RuntimeError(
            "Cannot append self-play data generated from a different checkpoint path."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--spin-mode", default="all_spin")
    parser.add_argument("--rank-q-alpha", type=float, default=None)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--dirichlet-eps", type=float, default=0.0)
    parser.add_argument("--garbage-rate", type=float, default=0.0)
    parser.add_argument("--garbage-min-lines", type=int, default=1)
    parser.add_argument("--garbage-max-lines", type=int, default=4)
    parser.add_argument("--garbage-timer", type=int, default=1)
    args = parser.parse_args()

    if args.workers != 1:
        raise NotImplementedError("Only --workers 1 is supported in phase 1.")
    if args.dirichlet_alpha != 0.0 or args.dirichlet_eps != 0.0:
        raise NotImplementedError("Dirichlet noise is not implemented in phase 1.")
    if args.games <= 0:
        raise ValueError("--games must be positive.")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    if args.garbage_rate < 0.0 or args.garbage_rate > 1.0:
        raise ValueError("--garbage-rate must be between 0 and 1.")
    if args.garbage_min_lines <= 0 or args.garbage_max_lines < args.garbage_min_lines:
        raise ValueError("Invalid garbage line range.")

    bundle = load_tetrisformer_checkpoint(args.model_path, device=args.device, eval_mode=True)
    if bundle.spin_mode and bundle.spin_mode != args.spin_mode:
        print(
            f"[warn] checkpoint spin_mode={bundle.spin_mode!r}, self-play spin_mode={args.spin_mode!r}",
            file=sys.stderr,
        )

    init_buffer_dir(args.output_dir, append=bool(args.append))
    existing_manifest = None
    shard_names: list[str] = []
    games_so_far = 0
    plies_so_far = 0
    next_shard_index = 0
    if args.append:
        existing_manifest = load_manifest(args.output_dir)
        _validate_existing_manifest(
            existing_manifest,
            spin_mode=args.spin_mode,
            checkpoint_path=bundle.checkpoint_path,
        )
        shard_names = list(existing_manifest.get("shards", []))
        games_so_far = int(existing_manifest.get("games", 0))
        plies_so_far = int(existing_manifest.get("plies", 0))
        next_shard_index = len(shard_names)

    shard_games = []
    generated_games = 0
    generated_plies = 0
    top_out_games = 0
    total_attack = 0.0

    metadata = {
        "spin_mode": args.spin_mode,
        "checkpoint_path": bundle.checkpoint_path,
        "checkpoint_arch": bundle.checkpoint_arch,
        "rank_q_alpha": bundle.rank_q_alpha if args.rank_q_alpha is None else float(args.rank_q_alpha),
    }

    for game_offset in range(int(args.games)):
        game_seed = int(args.seed) + game_offset
        game = run_single_selfplay_game(
            model_bundle=bundle,
            game_id=games_so_far + generated_games,
            max_plies=args.max_plies,
            beam_width=args.beam_width,
            top_k=args.top_k,
            temperature=args.temperature,
            spin_mode=args.spin_mode,
            rank_q_alpha=args.rank_q_alpha,
            seed=game_seed,
            garbage_rate=args.garbage_rate,
            garbage_min_lines=args.garbage_min_lines,
            garbage_max_lines=args.garbage_max_lines,
            garbage_timer=args.garbage_timer,
        )
        shard_games.append(game)
        generated_games += 1
        generated_plies += int(game.get("stats", {}).get("plies", len(game.get("steps", []))))
        total_attack += float(game.get("stats", {}).get("total_attack", 0.0))
        if game.get("termination") == "top_out":
            top_out_games += 1

        if len(shard_games) >= args.shard_size:
            shard_name = write_game_shard(args.output_dir, next_shard_index, shard_games, metadata)
            shard_names.append(shard_name)
            next_shard_index += 1
            shard_games = []

    if shard_games:
        shard_name = write_game_shard(args.output_dir, next_shard_index, shard_games, metadata)
        shard_names.append(shard_name)

    manifest = SelfPlayManifest(
        buffer_version=SELFPLAY_BUFFER_VERSION,
        spin_mode=args.spin_mode,
        checkpoint_path=bundle.checkpoint_path,
        checkpoint_arch=bundle.checkpoint_arch,
        games=games_so_far + generated_games,
        plies=plies_so_far + generated_plies,
        shards=shard_names,
    )
    write_manifest(args.output_dir, manifest)

    avg_plies = float(generated_plies) / float(generated_games) if generated_games > 0 else 0.0
    avg_attack = float(total_attack) / float(generated_plies) if generated_plies > 0 else 0.0
    top_out_rate = float(top_out_games) / float(generated_games) if generated_games > 0 else 0.0
    print(f"games_generated={generated_games}")
    print(f"total_plies={generated_plies}")
    print(f"avg_plies_per_game={avg_plies:.3f}")
    print(f"top_out_rate={top_out_rate:.3f}")
    print(f"avg_attack_per_ply={avg_attack:.3f}")
    print(f"output_dir={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
