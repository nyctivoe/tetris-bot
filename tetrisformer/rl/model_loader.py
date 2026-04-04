from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tetrisformer.model import NUM_BOARD_CHANNELS, NUM_STATS, TetrisFormerV4, get_device


DEFAULT_RANK_Q_ALPHA = 0.3
DEFAULT_NUM_HEADS = 6
DEFAULT_DEPTH = 4


@dataclass(frozen=True)
class LoadedModelBundle:
    model: TetrisFormerV4
    device: torch.device
    checkpoint_path: str
    checkpoint_arch: str
    arch: str
    cache_version: int | None
    spin_mode: str | None
    bootstrap_enabled: bool | None
    rank_q_alpha: float
    config: dict[str, Any]


def _infer_embed_dim_from_state_dict(state: dict[str, Any]) -> int:
    cls_token = state.get("cls_token")
    if hasattr(cls_token, "shape") and len(cls_token.shape) >= 3:
        return int(cls_token.shape[-1])

    weight = state.get("piece_embedding.weight")
    if hasattr(weight, "shape") and len(weight.shape) >= 2:
        return int(weight.shape[-1])

    raise RuntimeError("Unable to infer embed_dim from checkpoint state_dict.")


def _infer_num_heads_from_checkpoint(ckpt: dict[str, Any], embed_dim: int) -> int:
    config = ckpt.get("config", {}) or {}
    num_heads = config.get("num_heads", config.get("nhead", DEFAULT_NUM_HEADS))
    try:
        num_heads = int(num_heads)
    except Exception as exc:
        raise RuntimeError(f"Invalid num_heads value in checkpoint config: {num_heads!r}") from exc

    if num_heads <= 0:
        raise RuntimeError(f"num_heads must be positive, got {num_heads}.")
    if embed_dim % num_heads != 0:
        raise RuntimeError(
            f"Incompatible checkpoint config: embed_dim={embed_dim} is not divisible by num_heads={num_heads}."
        )
    return num_heads


def _infer_depth_from_state_dict(state: dict[str, Any]) -> int:
    max_index = -1
    for key in state:
        if not key.startswith("transformer.layers."):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[2])
        except ValueError:
            continue
        max_index = max(max_index, idx)

    if max_index >= 0:
        return max_index + 1
    return DEFAULT_DEPTH


def load_tetrisformer_checkpoint(
    checkpoint_path: str,
    device: torch.device | str | None = None,
    eval_mode: bool = True,
) -> LoadedModelBundle:
    checkpoint_path = str(Path(checkpoint_path).resolve())
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    target_device = get_device() if device is None else torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(
            "Unsupported checkpoint format: expected a checkpoint dict containing model_state_dict."
        )

    if "model_state_dict" not in ckpt:
        raise RuntimeError(
            "Unsupported checkpoint format: missing model_state_dict. "
            "Expected checkpoints saved by the current training pipeline."
        )

    state = ckpt["model_state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Invalid checkpoint: model_state_dict is not a state dict.")

    config = dict(ckpt.get("config", {}) or {})
    embed_dim = _infer_embed_dim_from_state_dict(state)
    num_heads = _infer_num_heads_from_checkpoint(ckpt, embed_dim)
    depth = _infer_depth_from_state_dict(state)

    board_channels = config.get("board_channels", NUM_BOARD_CHANNELS)
    num_stats = config.get("num_stats", NUM_STATS)
    try:
        board_channels = int(board_channels)
        num_stats = int(num_stats)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid board_channels/num_stats in checkpoint config: {board_channels!r}, {num_stats!r}"
        ) from exc

    model = TetrisFormerV4(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        board_channels=board_channels,
        num_stats=num_stats,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint is incompatible with TetrisFormerV4 under strict loading."
        ) from exc

    model.to(target_device)
    if eval_mode:
        model.eval()

    rank_q_alpha = config.get("rank_q_alpha", DEFAULT_RANK_Q_ALPHA)
    try:
        rank_q_alpha = float(rank_q_alpha)
    except Exception as exc:
        raise RuntimeError(f"Invalid rank_q_alpha in checkpoint config: {rank_q_alpha!r}") from exc

    checkpoint_arch = ckpt.get("checkpoint_arch")
    arch = ckpt.get("arch")

    return LoadedModelBundle(
        model=model,
        device=target_device,
        checkpoint_path=checkpoint_path,
        checkpoint_arch="" if checkpoint_arch is None else str(checkpoint_arch),
        arch="" if arch is None else str(arch),
        cache_version=ckpt.get("cache_version"),
        spin_mode=ckpt.get("spin_mode"),
        bootstrap_enabled=ckpt.get("bootstrap_enabled"),
        rank_q_alpha=rank_q_alpha,
        config=config,
    )
