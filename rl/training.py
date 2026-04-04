from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model_v2 import TetrisZeroConfig, TetrisZeroNet, load_checkpoint, save_checkpoint
from rl.replay_buffer import load_manifest, load_replay_shard, sample_batch
from rl.schemas import TrainingSampleV1, validate_training_sample


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    lr: float = 2.0e-3
    epochs: int = 1
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    policy_weight: float = 1.0
    value_weight: float = 1.0
    attack_weight: float = 0.3
    survival_weight: float = 0.1
    surge_weight: float = 0.3
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "tetriszero_fixture.pt"


def make_fixture_samples(count: int = 16, seed: int = 0) -> list[TrainingSampleV1]:
    rng = np.random.default_rng(seed)
    samples: list[TrainingSampleV1] = []
    for idx in range(int(count)):
        candidate_count = 3 + (idx % 3)
        board = np.zeros((12, 20, 10), dtype=np.float32)
        board[0, -1, idx % 10] = 1.0
        board[2, :, :] = (idx % 4) / 4.0
        pieces = np.array([1, 0, 1, 3, 4, 5, 7], dtype=np.int64)
        context = np.linspace(0.0, 1.0, 28, dtype=np.float32)
        context[0] = idx / max(1, count)
        candidate_features = rng.normal(loc=0.0, scale=0.25, size=(candidate_count, 32)).astype(np.float32)
        candidate_features[:, 0] = 0.0
        candidate_features[0, 0] = 1.0
        policy = np.zeros((candidate_count,), dtype=np.float32)
        policy[0] = 1.0
        samples.append(
            validate_training_sample(
                {
                    "version": 1,
                    "board_tensor": board,
                    "piece_ids": pieces,
                    "context_scalars": context,
                    "candidate_features": candidate_features,
                    "candidate_count": candidate_count,
                    "policy_target": policy,
                    "value_target": 1.0 if idx % 2 == 0 else -1.0,
                    "attack_target": float(idx % 5),
                    "survival_target": 1.0 if idx % 2 == 0 else 0.0,
                    "surge_target": float(idx % 3),
                    "metadata": {"fixture_index": idx},
                }
            )
        )
    return samples


def load_samples_from_buffer(buffer_dir: str | Path) -> list[TrainingSampleV1]:
    manifest = load_manifest(buffer_dir)
    samples: list[TrainingSampleV1] = []
    for shard_name in list(manifest.get("shards") or []):
        shard = load_replay_shard(Path(buffer_dir) / shard_name)
        samples.extend(shard["samples"])
    return samples


def collate_batch(samples: list[TrainingSampleV1], device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_candidates = max(sample["candidate_features"].shape[0] for sample in samples)
    board = np.stack([sample["board_tensor"] for sample in samples]).astype(np.float32)
    pieces = np.stack([sample["piece_ids"] for sample in samples]).astype(np.int64)
    context = np.stack([sample["context_scalars"] for sample in samples]).astype(np.float32)
    candidate_features = np.zeros((batch_size, max_candidates, 32), dtype=np.float32)
    candidate_mask = np.zeros((batch_size, max_candidates), dtype=np.bool_)
    policy_target = np.zeros((batch_size, max_candidates), dtype=np.float32)

    for index, sample in enumerate(samples):
        count = int(sample["candidate_count"])
        candidate_features[index, :count] = sample["candidate_features"]
        candidate_mask[index, :count] = True
        policy_target[index, :count] = sample["policy_target"]

    return {
        "board_tensor": torch.from_numpy(board).to(device),
        "piece_ids": torch.from_numpy(pieces).to(device),
        "context_scalars": torch.from_numpy(context).to(device),
        "candidate_features": torch.from_numpy(candidate_features).to(device),
        "candidate_mask": torch.from_numpy(candidate_mask).to(device),
        "policy_target": torch.from_numpy(policy_target).to(device),
        "value_target": torch.tensor([sample["value_target"] for sample in samples], dtype=torch.float32, device=device),
        "attack_target": torch.tensor([sample["attack_target"] for sample in samples], dtype=torch.float32, device=device),
        "survival_target": torch.tensor([sample["survival_target"] for sample in samples], dtype=torch.float32, device=device),
        "surge_target": torch.tensor([sample["surge_target"] for sample in samples], dtype=torch.float32, device=device),
    }


def compute_losses(model: TetrisZeroNet, batch: dict[str, torch.Tensor], cfg: TrainingConfig) -> dict[str, torch.Tensor]:
    outputs = model(
        batch["board_tensor"],
        batch["piece_ids"],
        batch["context_scalars"],
        batch["candidate_features"],
        batch["candidate_mask"],
    )
    log_policy = F.log_softmax(outputs["policy_logits"], dim=-1)
    policy_loss = -(batch["policy_target"] * log_policy).sum(dim=-1).mean()
    value_loss = F.mse_loss(outputs["value"], batch["value_target"])
    attack_loss = F.smooth_l1_loss(outputs["attack"], batch["attack_target"])
    survival_loss = F.binary_cross_entropy(outputs["survival"], batch["survival_target"])
    surge_loss = F.smooth_l1_loss(outputs["realized_surge"], batch["surge_target"])
    total = (
        cfg.policy_weight * policy_loss
        + cfg.value_weight * value_loss
        + cfg.attack_weight * attack_loss
        + cfg.survival_weight * survival_loss
        + cfg.surge_weight * surge_loss
    )
    return {
        "total": total,
        "policy": policy_loss,
        "value": value_loss,
        "attack": attack_loss,
        "survival": survival_loss,
        "surge": surge_loss,
    }


def train_one_step(
    model: TetrisZeroNet,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    cfg: TrainingConfig,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = compute_losses(model, batch, cfg)
    losses["total"].backward()
    if cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def evaluate_samples(model: TetrisZeroNet, samples: list[TrainingSampleV1], cfg: TrainingConfig) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        batch = collate_batch(samples, device=cfg.device)
        losses = compute_losses(model, batch, cfg)
    return {f"val_{name}": float(value.detach().cpu()) for name, value in losses.items()}


def train_model(
    samples: list[TrainingSampleV1],
    *,
    cfg: TrainingConfig | None = None,
    model: TetrisZeroNet | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or TrainingConfig()
    model = model or TetrisZeroNet(TetrisZeroConfig())
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    if not samples:
        raise RuntimeError("No training samples provided.")

    history: list[dict[str, float]] = []
    validation_set = samples[: min(len(samples), max(2, cfg.batch_size))]
    train_set = samples
    rng = np.random.default_rng(0)

    for _epoch in range(int(cfg.epochs)):
        order = rng.permutation(len(train_set))
        for start in range(0, len(train_set), int(cfg.batch_size)):
            batch_samples = [train_set[int(index)] for index in order[start : start + int(cfg.batch_size)]]
            batch = collate_batch(batch_samples, device=cfg.device)
            history.append(train_one_step(model, optimizer, batch, cfg))

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / cfg.checkpoint_name
    save_checkpoint(checkpoint_path, model, optimizer, model.config, dict(checkpoint_metadata or {}))
    metrics = evaluate_samples(model, validation_set, cfg)
    return {
        "model": model,
        "optimizer": optimizer,
        "history": history,
        "checkpoint_path": checkpoint_path,
        "validation": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the root-level TetrisZero model.")
    parser.add_argument("--fixture-mode", action="store_true")
    parser.add_argument("--buffer-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="tetriszero_fixture.pt")
    parser.add_argument("--init-checkpoint", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = TrainingConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=str(args.device),
        checkpoint_dir=str(args.checkpoint_dir),
        checkpoint_name=str(args.checkpoint_name),
    )

    model = None
    optimizer_state_dict = None
    if args.init_checkpoint:
        bundle = load_checkpoint(args.init_checkpoint, device=cfg.device)
        model = bundle["model"]
        optimizer_state_dict = bundle.get("optimizer_state_dict")

    if args.fixture_mode or not args.buffer_dir:
        samples = make_fixture_samples(count=max(8, int(args.batch_size) * 2))
    else:
        samples = load_samples_from_buffer(args.buffer_dir)
        if not samples:
            samples = sample_batch(args.buffer_dir, batch_size=max(8, int(args.batch_size)))

    result = train_model(
        samples,
        cfg=cfg,
        model=model,
        optimizer_state_dict=optimizer_state_dict,
        checkpoint_metadata={
            "fixture_mode": bool(args.fixture_mode),
            "buffer_dir": args.buffer_dir,
            "init_checkpoint": args.init_checkpoint,
        },
    )
    print(
        f"trained_steps={len(result['history'])} "
        f"val_total={result['validation']['val_total']:.4f} "
        f"checkpoint={result['checkpoint_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
