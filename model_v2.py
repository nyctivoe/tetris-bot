from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TetrisZeroConfig:
    board_channels: int = 12
    hidden_channels: int = 128
    num_blocks: int = 12
    num_pieces: int = 8
    piece_dim: int = 64
    context_features: int = 28
    fused_dim: int = 384
    candidate_features: int = 32
    policy_hidden: int = 128


def resolve_module_device(module: Any) -> torch.device:
    if isinstance(module, nn.Module):
        for tensor in module.parameters():
            return tensor.device
        for tensor in module.buffers():
            return tensor.device
    return torch.device("cpu")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.relu(self.bn1(x))
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        return h + residual


class BoardEncoder(nn.Module):
    """Input: (B, 12, 20, 10) -> Output: (B, 128, 20, 10)."""

    def __init__(self, in_channels: int = 12, hidden_channels: int = 128, num_blocks: int = 12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return h


class AttentionPool(nn.Module):
    """Input: (B, 128, 20, 10) -> Output: (B, 256)."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = features.shape
        attn_logits = self.attn_conv(features).reshape(b, 1, -1)
        attn = torch.softmax(attn_logits, dim=-1)
        flat = features.reshape(b, c, -1)
        weighted = (flat * attn).sum(dim=-1)
        avg = flat.mean(dim=-1)
        return torch.cat([weighted, avg], dim=-1)


class ContextEncoder(nn.Module):
    """piece_ids: (B, 7), context_scalars: (B, 28) -> (B, 192)."""

    def __init__(self, num_pieces: int = 8, piece_dim: int = 64, context_features: int = 28):
        super().__init__()
        self.piece_embed = nn.Embedding(num_pieces, piece_dim)
        self.slot_position_embed = nn.Embedding(7, piece_dim)
        self.context_mlp = nn.Sequential(
            nn.Linear(context_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.piece_fusion = nn.Sequential(
            nn.Linear(7 * piece_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, piece_ids: torch.Tensor, context_scalars: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(7, device=piece_ids.device)
        piece_embeds = self.piece_embed(piece_ids) + self.slot_position_embed(positions).unsqueeze(0)
        piece_summary = self.piece_fusion(piece_embeds.flatten(1))
        context_summary = self.context_mlp(context_scalars)
        return torch.cat([piece_summary, context_summary], dim=-1)


class FusionLayer(nn.Module):
    """board_summary: (B, 256), context: (B, 192) -> (B, 384)."""

    def __init__(self, board_dim: int = 256, context_dim: int = 192, fused_dim: int = 384):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(board_dim + context_dim, 512),
            nn.ReLU(),
            nn.Linear(512, fused_dim),
            nn.ReLU(),
        )

    def forward(self, board_summary: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([board_summary, context], dim=-1))


class PolicyHead(nn.Module):
    """state_embedding: (B, 384), delta_features: (B, N, 32) -> logits: (B, N)."""

    def __init__(self, state_dim: int = 384, delta_features: int = 32, hidden: int = 128):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(state_dim + delta_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state_embedding: torch.Tensor, delta_features: torch.Tensor) -> torch.Tensor:
        b, n, _ = delta_features.shape
        state_expanded = state_embedding.unsqueeze(1).expand(b, n, -1)
        combined = torch.cat([state_expanded, delta_features], dim=-1)
        return self.score_mlp(combined).squeeze(-1)


class ValueHead(nn.Module):
    def __init__(self, state_dim: int = 384):
        super().__init__()
        self.value_mlp = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        return self.value_mlp(state_embedding).squeeze(-1)


class AuxiliaryHeads(nn.Module):
    def __init__(self, state_dim: int = 384):
        super().__init__()
        self.attack_head = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.survival_head = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.realized_surge_head = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, state_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "attack": self.attack_head(state_embedding).squeeze(-1),
            "survival": self.survival_head(state_embedding).squeeze(-1),
            "realized_surge": self.realized_surge_head(state_embedding).squeeze(-1),
        }


class TetrisZeroNet(nn.Module):
    """Board: (B, 12, 20, 10), pieces: (B, 7), context: (B, 28), candidates: (B, N, 32)."""

    def __init__(self, config: TetrisZeroConfig | None = None):
        super().__init__()
        self.config = config or TetrisZeroConfig()
        self.board_encoder = BoardEncoder(
            in_channels=self.config.board_channels,
            hidden_channels=self.config.hidden_channels,
            num_blocks=self.config.num_blocks,
        )
        self.attention_pool = AttentionPool(channels=self.config.hidden_channels)
        self.context_encoder = ContextEncoder(
            num_pieces=self.config.num_pieces,
            piece_dim=self.config.piece_dim,
            context_features=self.config.context_features,
        )
        self.fusion = FusionLayer(fused_dim=self.config.fused_dim)
        self.policy_head = PolicyHead(
            state_dim=self.config.fused_dim,
            delta_features=self.config.candidate_features,
            hidden=self.config.policy_hidden,
        )
        self.value_head = ValueHead(state_dim=self.config.fused_dim)
        self.aux_heads = AuxiliaryHeads(state_dim=self.config.fused_dim)

    def encode_state(
        self,
        board_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        context_scalars: torch.Tensor,
    ) -> torch.Tensor:
        board_features = self.board_encoder(board_tensor)
        board_summary = self.attention_pool(board_features)
        context_summary = self.context_encoder(piece_ids, context_scalars)
        return self.fusion(board_summary, context_summary)

    def forward(
        self,
        board_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        context_scalars: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state = self.encode_state(board_tensor, piece_ids, context_scalars)
        policy_logits = self.policy_head(state, candidate_features)
        if candidate_mask is not None:
            candidate_mask = candidate_mask.to(dtype=torch.bool, device=policy_logits.device)
            policy_logits = policy_logits.masked_fill(~candidate_mask, -1.0e9)
        aux = self.aux_heads(state)
        return {
            "policy_logits": policy_logits,
            "value": self.value_head(state),
            "attack": aux["attack"],
            "survival": aux["survival"],
            "realized_surge": aux["realized_surge"],
        }


def save_checkpoint(
    path: str | Path,
    model: TetrisZeroNet,
    optimizer: torch.optim.Optimizer | None,
    config: TetrisZeroConfig | dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "model_config": asdict(config) if isinstance(config, TetrisZeroConfig) else (config or asdict(model.config)),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, Path(path))


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = TetrisZeroConfig(**dict(payload.get("model_config") or {}))
    model = TetrisZeroNet(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "optimizer_state_dict": payload.get("optimizer_state_dict"),
        "config": config,
        "metadata": dict(payload.get("metadata") or {}),
        "raw": payload,
    }
