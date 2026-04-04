from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


SELFPLAY_BUFFER_VERSION = 1


@dataclass(frozen=True)
class SelfPlayManifest:
    buffer_version: int
    spin_mode: str
    checkpoint_path: str
    checkpoint_arch: str
    games: int
    plies: int
    shards: list[str]


def _manifest_path(output_dir: str) -> Path:
    return Path(output_dir) / "manifest.json"


def init_buffer_dir(output_dir: str, append: bool) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(output_dir)
    shard_paths = list(out.glob("shard_*.pt"))
    if append:
        return
    if manifest_path.exists() or shard_paths:
        raise RuntimeError(
            f"Output directory already contains a self-play buffer: {out}. Use --append to add shards."
        )


def write_game_shard(output_dir: str, shard_index: int, shard_games: list[dict], metadata: dict) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_name = f"shard_{int(shard_index):06d}.pt"
    final_path = out / shard_name
    temp_path = out / f"{shard_name}.tmp"

    payload = {
        "buffer_version": SELFPLAY_BUFFER_VERSION,
        "spin_mode": metadata["spin_mode"],
        "checkpoint_path": metadata["checkpoint_path"],
        "checkpoint_arch": metadata.get("checkpoint_arch"),
        "rank_q_alpha": float(metadata["rank_q_alpha"]),
        "games": shard_games,
    }
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, final_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return shard_name


def write_manifest(output_dir: str, manifest: SelfPlayManifest | dict[str, Any]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(output_dir)
    temp_path = out / "manifest.json.tmp"

    payload = asdict(manifest) if isinstance(manifest, SelfPlayManifest) else dict(manifest)
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(temp_path, manifest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_manifest(output_dir: str) -> dict[str, Any]:
    manifest_path = _manifest_path(output_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Self-play manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)
