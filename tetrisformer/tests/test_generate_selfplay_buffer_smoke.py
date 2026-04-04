import subprocess
import sys
from pathlib import Path

import torch


def test_generate_selfplay_buffer_cli_smoke(checkpoint_factory, tmp_path: Path):
    ckpt_path = checkpoint_factory(name="selfplay_ckpt.pt", spin_mode="all_spin")
    output_dir = tmp_path / "selfplay_buffer_v1"
    root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tetrisformer.tools.generate_selfplay_buffer",
            "--model-path",
            str(ckpt_path),
            "--output-dir",
            str(output_dir),
            "--games",
            "1",
            "--max-plies",
            "3",
            "--beam-width",
            "1",
            "--top-k",
            "2",
            "--shard-size",
            "1",
            "--seed",
            "0",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest_path = output_dir / "manifest.json"
    shard_path = output_dir / "shard_000000.pt"
    assert manifest_path.exists()
    assert shard_path.exists()

    payload = torch.load(shard_path, weights_only=False)
    assert payload["buffer_version"] == 1
    assert payload["games"]
    assert "games_generated=1" in result.stdout
