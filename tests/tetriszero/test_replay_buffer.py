from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rl.replay_buffer import init_buffer_dir, load_manifest, load_replay_shard, sample_batch, write_manifest, write_replay_shard
from rl.schemas import validate_replay_shard


def test_write_manifest_and_shard_round_trip(tmp_path: Path, synthetic_replay_samples):
    output_dir = tmp_path / "buffer"
    init_buffer_dir(output_dir, append=False)
    shard_name = write_replay_shard(output_dir, 0, synthetic_replay_samples, {"stage": "fixture"})
    write_manifest(output_dir, {"buffer_version": 1, "shards": [shard_name]})

    manifest = load_manifest(output_dir)
    shard = load_replay_shard(output_dir / shard_name)

    assert manifest["shards"] == [shard_name]
    assert shard["version"] == 1
    assert len(shard["samples"]) == len(synthetic_replay_samples)


def test_atomic_shard_write_does_not_leave_partial_file(tmp_path: Path, synthetic_replay_samples, monkeypatch):
    output_dir = tmp_path / "buffer"
    init_buffer_dir(output_dir, append=False)

    import rl.replay_buffer as replay_buffer

    def broken_save(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(replay_buffer.torch, "save", broken_save)
    with pytest.raises(RuntimeError, match="boom"):
        write_replay_shard(output_dir, 0, synthetic_replay_samples, {"stage": "fixture"})

    assert not (output_dir / "shard_000000.pt").exists()


def test_schema_mismatch_fails_loudly(tmp_path: Path):
    payload = {"version": 999, "metadata": {}, "samples": []}
    path = tmp_path / "bad.pt"
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="Unsupported replay shard version"):
        validate_replay_shard(torch.load(path, weights_only=False))


def test_sample_batch_returns_requested_count(tmp_path: Path, synthetic_replay_samples):
    output_dir = tmp_path / "buffer"
    init_buffer_dir(output_dir, append=False)
    shard_name = write_replay_shard(output_dir, 0, synthetic_replay_samples, {"stage": "fixture"})
    write_manifest(output_dir, {"buffer_version": 1, "shards": [shard_name]})

    batch = sample_batch(output_dir, batch_size=3, seed=0)

    assert len(batch) == 3
