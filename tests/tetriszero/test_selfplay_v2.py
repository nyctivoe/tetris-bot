from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import torch
import pytest

from rl.selfplay_v2 import generate_pvp_episode, generate_singleplayer_episode
from .fixtures import make_engine


def test_generate_pvp_episode_draw_targets_are_neutral(monkeypatch):
    monkeypatch.setattr(
        "rl.selfplay_v2.run_pvp_turn",
        lambda active, passive, action, move_number, cfg: {
            "move_number": int(move_number),
            "terminated": False,
            "reason": None,
            "action": dict(action),
            "stats": {},
            "resolve": {},
            "sent_to_opponent": 0,
            "active_pending_after": active.get_pending_garbage_summary(),
            "passive_pending_after": passive.get_pending_garbage_summary(),
        },
    )

    episode = generate_pvp_episode(max_plies=1, seed=1)

    assert episode["winner"] is None
    assert episode["player_a_samples"]
    assert episode["player_b_samples"]
    assert all(sample["value_target"] == 0.0 for sample in episode["player_a_samples"])
    assert all(sample["value_target"] == 0.0 for sample in episode["player_b_samples"])
    assert all(sample["survival_target"] == 1.0 for sample in episode["player_a_samples"])
    assert all(sample["survival_target"] == 1.0 for sample in episode["player_b_samples"])


def test_generate_pvp_episode_records_post_turn_start_garbage_state(monkeypatch):
    def make_engine_with_pending_garbage(seed, spin_mode="all_spin"):
        return make_engine(
            seed=0 if seed is None else int(seed),
            spin_mode=spin_mode,
            current="T",
            incoming_garbage=[{"lines": 1, "timer": 1, "col": 0}],
        )

    monkeypatch.setattr("rl.selfplay_v2._make_engine", make_engine_with_pending_garbage)
    episode = generate_pvp_episode(max_plies=1, seed=3)

    assert episode["player_a_samples"]
    assert float(episode["player_a_samples"][0]["context_scalars"][5]) == 0.0


def test_selfplay_cli_writes_replay_buffer(tmp_path: Path):
    output_dir = tmp_path / "buffer"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl.selfplay_v2",
            "--mode",
            "singleplayer",
            "--episodes",
            "2",
            "--max-plies",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "samples=" in proc.stdout
    assert (output_dir / "manifest.json").exists()


def test_selfplay_cli_accepts_model_checkpoint(tmp_path: Path, tiny_model):
    checkpoint_path = tmp_path / "tiny.pt"
    from model_v2 import save_checkpoint

    save_checkpoint(checkpoint_path, tiny_model, None, tiny_model.config, {"tag": "selfplay"})
    output_dir = tmp_path / "buffer"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl.selfplay_v2",
            "--mode",
            "singleplayer",
            "--episodes",
            "1",
            "--max-plies",
            "1",
            "--output-dir",
            str(output_dir),
            "--model-path",
            str(checkpoint_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert (output_dir / "manifest.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this regression test.")
def test_singleplayer_episode_supports_cuda_model(tiny_model):
    model = tiny_model.to("cuda").eval()

    episode = generate_singleplayer_episode(max_plies=1, seed=0, model=model)

    assert episode["samples"]
