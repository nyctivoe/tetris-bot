# TetrisFormer

A transformer-based deep learning model that learns to play Tetris from expert human replays. Uses a CNN + transformer hybrid with three output heads trained via multi-task imitation learning and Q-learning.

## Overview

TetrisFormer V4 ("Search-Distilled Candidate Q Learning") processes 76,693 expert TETR.IO game replays (7.7 GB CSV) to learn piece placement decisions. For each board state, it evaluates candidate placements produced by BFS and ranks them using a combination of:

- **Rank head** — soft ranking via rollout-return distributions over candidate moves
- **Q head** — per-candidate rollout return regression (SmoothL1Loss)
- **Attack head** — immediate garbage attack prediction (SmoothL1Loss)
- **Imitation loss** — weak expert move classification on the rank logits (CrossEntropyLoss with label_smoothing=0.05)

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA GPU recommended (falls back to CPU)
- Optional: [numba](https://numba.pydata.org/) for accelerated garbage-row matching in data loading

## Quick Start

```bash
# Install dependencies
uv sync

# Build game index (required once, creates game_index.csv)
# This is done automatically if game_index.csv does not exist.

# Preprocess BFS cache (currently required for training, creates game_cache_v2/)
uv run python preparse_games.py

# Run training
uv run python model.py
```

## Architecture

### Source Files

| File | Role |
|------|------|
| `model.py` | Model definition (`TetrisFormerV4`), dataset (`SmartRolloutRankDataset`), training loop |
| `tetrisEngine.py` | Tetris simulation engine with SRS wall kicks + BFS candidate generation |
| `fileParsing.py` | CSV loading with byte-offset indexing, board parsing, BFS match alignment |
| `preparse_games.py` | Offline BFS cache builder (outputs to `game_cache_v2/`) |

### Model: TetrisFormerV4

```
7-channel board features (40x10)
    -> CNN stem (Conv 7->32, 3x3 + BN + ReLU)
    -> Residual block (32->64, stride 2)
    -> Residual block (64->192, stride 2)
    -> 30 grid tokens (10x3, 192-dim) + 2D sin/cos positional encoding

CLS token + Stats token (7 scalars -> Linear 7->192) + Queue tokens (7: current/placed + hold + 5 next)
    -> Transformer Encoder (4 layers, 6 heads, 192-dim, FFN=768, dropout=0.1, pre-LN)
    -> CLS output

CLS -> Rank Head    (Linear 192->256->1)
    -> Attack Head  (Linear 192->128->1)
    -> Q Head       (Linear 192->256->64->1, with dropout)
```

Board feature channels: base occupancy, result occupancy, placement diff, height map, holes,
row fill ratio, and T-slot cavity map.

### Inference Scoring

```
Score(placement) = Rank_Score + alpha * Q_Score    (default alpha = 0.3)
```

## Training Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--k-candidates` | 10 | Candidates per move (1 expert + 9 negatives) |
| `--batch-size` | 128 | Batch size |
| `--lr` | 2e-4 | Learning rate (AdamW, weight_decay=1e-2) |
| `--epochs` | 25 | Training epochs |
| `--samples-per-epoch` | 250,000 | Samples per epoch |
| `--rollout-depth` | 6 | Beam-search rollout lookahead depth |
| `--rollout-beam` | 2 | Beam width |
| `--rollout-expand` | 6 | Expansion width per beam node |
| `--gamma` | 0.97 | Discount factor |
| `--soft-target-temp` | 0.5 | Temperature for soft rank targets |

### Loss Weights

| Loss | Weight | Description |
|------|--------|-------------|
| Soft rank | 1.0 | KL-divergence against rollout-return distribution |
| Q regression | 1.0 | SmoothL1 on per-candidate rollout returns |
| Attack regression | 0.1 | SmoothL1 on immediate garbage prediction |
| Imitation | 0.5 | CrossEntropy on expert move label |

### Reward Shaping

| Parameter | Default |
|-----------|---------|
| Attack weight | 1.0 |
| T-spin bonus | 1.5 |
| B2B bonus | 0.5 |
| Height penalty | 0.1 |
| Holes penalty | 0.5 |
| Topout penalty | 10.0 |
| T-slot ready bonus | 0.3 |

### Other Defaults

- **Importance weighting**: Moves with line clears, attack > 0, or expert Q >= 1.5 get 6x sample weight
- **Label smoothing**: 0.05 (on imitation CE loss)
- **Target network**: EMA with tau=0.005, bootstrapped leaf values activate at epoch 3
- **Gradient clipping**: clip grad norm at 1.0 by default (`--grad-clip 0` disables it)
- **LR schedule**: linear warmup for 500 steps, then cosine decay to `1e-6`
- **Train/val split**: 90/10

## Data Files

| File | Size | Description |
|------|------|-------------|
| `data.csv` | ~7.7 GB | Raw game replays (board state, piece info, ratings, attack stats) |
| `game_index.csv` | ~2.5 MB | Byte-offset index for fast random game access |
| `game_cache_v2/` | ~48 GB | Preprocessed BFS states (currently required for training) |
| `checkpoints/` | varies | Saved model weights |

## CLI Reference

```bash
uv run python model.py --help
```

Key arguments: `--data-path`, `--index-path`, `--epochs`, `--lr`, `--batch-size`, `--k-candidates`, `--num-workers`, `--rollout-depth`, `--rollout-beam`, `--samples-per-epoch`

Use `--num-workers 0` to disable multiprocessing if issues arise.
