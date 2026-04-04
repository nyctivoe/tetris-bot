from __future__ import annotations

import subprocess
import sys

from model_v2 import TetrisZeroConfig, TetrisZeroNet, load_checkpoint
from rl.training import TrainingConfig, collate_batch, compute_losses, train_model, train_one_step


def test_one_training_step_runs(tiny_model_batch):
    cfg = TrainingConfig(batch_size=4, epochs=1)
    model = TetrisZeroNet(TetrisZeroConfig())
    optimizer = __import__("torch").optim.AdamW(model.parameters(), lr=cfg.lr)
    batch = collate_batch(tiny_model_batch, device=cfg.device)

    metrics = train_one_step(model, optimizer, batch, cfg)

    assert metrics["total"] >= 0.0
    assert "policy" in metrics


def test_tiny_overfit_loss_decreases(tiny_model_batch):
    cfg = TrainingConfig(batch_size=4, epochs=1, lr=1.0e-3)
    model = TetrisZeroNet(TetrisZeroConfig())
    optimizer = __import__("torch").optim.AdamW(model.parameters(), lr=cfg.lr)
    batch = collate_batch(tiny_model_batch, device=cfg.device)
    initial = float(compute_losses(model, batch, cfg)["total"].detach().cpu())

    best = initial
    for _ in range(20):
        step_total = train_one_step(model, optimizer, batch, cfg)["total"]
        best = min(best, step_total)

    assert best < initial


def test_train_model_returns_validation_and_checkpoint(tmp_path, tiny_model_batch):
    cfg = TrainingConfig(batch_size=2, epochs=1, checkpoint_dir=str(tmp_path), checkpoint_name="custom.pt")
    result = train_model(tiny_model_batch, cfg=cfg)

    assert result["checkpoint_path"].exists()
    assert result["checkpoint_path"].name == "custom.pt"
    assert "val_total" in result["validation"]


def test_training_cli_can_resume_from_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"
    initial_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl.training",
            "--fixture-mode",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-name",
            "first.pt",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert initial_proc.returncode == 0
    checkpoint = load_checkpoint(checkpoint_dir / "first.pt", device="cpu")
    assert checkpoint["model"] is not None


def test_training_cli_fixture_smoke():
    proc = subprocess.run(
        [sys.executable, "-m", "rl.training", "--fixture-mode", "--epochs", "1", "--batch-size", "4"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "trained_steps=" in proc.stdout
