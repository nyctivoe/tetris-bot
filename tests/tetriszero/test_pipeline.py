from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_pipeline_cli_runs_end_to_end(tmp_path: Path):
    run_dir = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl.pipeline",
            "--run-dir",
            str(run_dir),
            "--cycles",
            "1",
            "--episodes-per-cycle",
            "2",
            "--max-plies",
            "2",
            "--epochs-per-cycle",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--eval-games",
            "2",
            "--eval-max-plies",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "pipeline_complete" in proc.stdout
    assert (run_dir / "replay_buffer" / "manifest.json").exists()
    assert (run_dir / "checkpoints" / "tetriszero_cycle_0001.pt").exists()
    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "status.json").exists()
    assert (run_dir / "checkpoints" / "latest.json").exists()
    assert (run_dir / "logs" / "cycle_metrics.csv").exists()
    assert (run_dir / "logs" / "train_history.csv").exists()
    assert (run_dir / "logs" / "events.jsonl").exists()
    assert (run_dir / "reports" / "latest_eval.json").exists()
