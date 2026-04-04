from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl.schemas import REPLAY_SHARD_VERSION, ReplayShardV1, TrainingSampleV1, validate_replay_shard, validate_training_sample


def _manifest_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "manifest.json"


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_json_save(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def init_buffer_dir(output_dir: str | Path, append: bool = False) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if append:
        return
    if _manifest_path(out).exists() or list(out.glob("shard_*.pt")):
        raise RuntimeError(f"Replay buffer already exists at {out}. Use append=True to add shards.")


def load_manifest(output_dir: str | Path) -> dict[str, Any]:
    path = _manifest_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"Replay manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> None:
    payload = dict(manifest)
    payload.setdefault("buffer_version", REPLAY_SHARD_VERSION)
    _atomic_json_save(_manifest_path(output_dir), payload)


def write_replay_shard(
    output_dir: str | Path,
    shard_index: int,
    samples: list[dict[str, Any] | TrainingSampleV1],
    metadata: dict[str, Any],
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_name = f"shard_{int(shard_index):06d}.pt"
    shard_path = out / shard_name
    validated_samples = [validate_training_sample(dict(sample)) for sample in samples]
    payload = ReplayShardV1(
        version=REPLAY_SHARD_VERSION,
        metadata=dict(metadata),
        samples=validated_samples,
    )
    _atomic_torch_save(shard_path, payload)
    return shard_name


def load_replay_shard(path: str | Path) -> ReplayShardV1:
    payload = torch.load(Path(path), weights_only=False)
    return validate_replay_shard(dict(payload))


def sample_batch(
    output_dir: str | Path,
    batch_size: int,
    *,
    seed: int = 0,
) -> list[TrainingSampleV1]:
    manifest = load_manifest(output_dir)
    shards = [Path(output_dir) / shard_name for shard_name in list(manifest.get("shards") or [])]
    if not shards:
        raise RuntimeError("Replay manifest contains no shards.")
    all_samples: list[TrainingSampleV1] = []
    for shard in shards:
        all_samples.extend(load_replay_shard(shard)["samples"])
    if not all_samples:
        raise RuntimeError("Replay buffer contains no samples.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(all_samples), size=int(batch_size))
    return [all_samples[int(index)] for index in indices]
