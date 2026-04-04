import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tetrisformer.model import CACHE_VERSION, CHECKPOINT_ARCH, NUM_BOARD_CHANNELS, NUM_STATS, TetrisFormerV4


@pytest.fixture
def checkpoint_factory(tmp_path: Path):
    def _write(
        *,
        name: str = "synthetic.pt",
        embed_dim: int = 192,
        num_heads: int = 6,
        depth: int = 4,
        rank_q_alpha: float = 0.3,
        spin_mode: str = "t_only",
    ) -> Path:
        model = TetrisFormerV4(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            board_channels=NUM_BOARD_CHANNELS,
            num_stats=NUM_STATS,
        )
        path = tmp_path / name
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "arch": "tetrisformer_v4",
                "checkpoint_arch": CHECKPOINT_ARCH,
                "cache_version": CACHE_VERSION,
                "spin_mode": spin_mode,
                "bootstrap_enabled": False,
                "config": {
                    "rank_q_alpha": rank_q_alpha,
                    "board_channels": NUM_BOARD_CHANNELS,
                    "num_stats": NUM_STATS,
                    "num_heads": num_heads,
                },
            },
            path,
        )
        return path

    return _write
