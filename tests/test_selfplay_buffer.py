from pathlib import Path

import torch

from rl.selfplay_buffer import (
    SELFPLAY_BUFFER_VERSION,
    SelfPlayManifest,
    init_buffer_dir,
    load_manifest,
    write_game_shard,
    write_manifest,
)
from tools.generate_selfplay_buffer import _validate_existing_manifest


def test_manifest_created_correctly_on_first_write(tmp_path: Path):
    output_dir = tmp_path / "buffer"
    init_buffer_dir(str(output_dir), append=False)

    manifest = SelfPlayManifest(
        buffer_version=SELFPLAY_BUFFER_VERSION,
        spin_mode="all_spin",
        checkpoint_path="ckpt.pt",
        checkpoint_arch="tetrisformer_v4r1",
        games=2,
        plies=5,
        shards=["shard_000000.pt"],
    )
    write_manifest(str(output_dir), manifest)

    loaded = load_manifest(str(output_dir))
    assert loaded["buffer_version"] == SELFPLAY_BUFFER_VERSION
    assert loaded["spin_mode"] == "all_spin"
    assert loaded["games"] == 2


def test_append_rejects_mismatched_spin_mode():
    manifest = {
        "buffer_version": SELFPLAY_BUFFER_VERSION,
        "spin_mode": "t_only",
        "checkpoint_path": "ckpt.pt",
        "checkpoint_arch": "tetrisformer_v4r1",
        "games": 0,
        "plies": 0,
        "shards": [],
    }
    try:
        _validate_existing_manifest(manifest, spin_mode="all_spin", checkpoint_path="ckpt.pt")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "spin_mode" in str(exc)


def test_write_game_shard_is_readable_and_matches_schema_version(tmp_path: Path):
    output_dir = tmp_path / "buffer"
    init_buffer_dir(str(output_dir), append=False)
    shard_name = write_game_shard(
        str(output_dir),
        shard_index=0,
        shard_games=[{"game_id": 1, "steps": []}],
        metadata={
            "spin_mode": "all_spin",
            "checkpoint_path": "ckpt.pt",
            "checkpoint_arch": "tetrisformer_v4r1",
            "rank_q_alpha": 0.3,
        },
    )

    payload = torch.load(output_dir / shard_name, weights_only=False)
    assert payload["buffer_version"] == SELFPLAY_BUFFER_VERSION
    assert payload["spin_mode"] == "all_spin"
    assert payload["games"][0]["game_id"] == 1


def test_atomic_shard_write_does_not_leave_partial_final_file(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "buffer"
    init_buffer_dir(str(output_dir), append=False)

    import rl.selfplay_buffer as sb

    def broken_save(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sb.torch, "save", broken_save)
    final_path = output_dir / "shard_000000.pt"

    try:
        write_game_shard(
            str(output_dir),
            shard_index=0,
            shard_games=[{"game_id": 1, "steps": []}],
            metadata={
                "spin_mode": "all_spin",
                "checkpoint_path": "ckpt.pt",
                "checkpoint_arch": "tetrisformer_v4r1",
                "rank_q_alpha": 0.3,
            },
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "boom" in str(exc)

    assert not final_path.exists()
