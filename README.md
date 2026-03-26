# TetrisFormer

A transformer-based deep learning model that learns to play Tetris from expert human replays. Uses a CNN + transformer hybrid with three output heads trained via multi-task imitation learning and search-distilled Q-learning.

## Overview

TetrisFormer evaluates candidate piece placements by encoding each board state as a 6-channel image and passing it through a CNN → Transformer pipeline. Rather than relying on handcrafted heuristics, the model learns board evaluation directly from 76,693 expert games.

The current version (V4) combines:
- **Imitation learning** — soft cross-entropy targets derived from beam-search rollouts over the expert's future piece sequence
- **Q-learning** — per-candidate rollout returns as regression targets, with a DQN-style EMA target network
- **Attack prediction** — regression of immediate garbage lines sent

## Requirements

- Python 3.14+
- PyTorch 2.9+ (CUDA 13.0 build)
- `uv` package manager

Install dependencies:

```bash
uv sync
```

Dependencies are pinned in `pyproject.toml` / `uv.lock` and pull from the PyTorch CUDA 13.0 index automatically.

## Data Files

| File / Directory | Size | Description |
|-----------------|------|-------------|
| `data.csv` | 7.7 GB | Raw game replays (22 columns: board state, piece info, ratings, attack stats) |
| `game_index.csv` | 2.5 MB | Byte-offset index for fast random access into `data.csv` (76,693 games) |
| `game_cache_v2/` | ~48 GB | Optional preprocessed BFS states; speeds up training significantly |
| `checkpoints/` | — | Saved model weights (`tetrisformer_v4_ep1.pt`, etc.) |

`data.csv` and `game_index.csv` must be present to train. The cache is optional but recommended.

## Preprocessing (one-time, optional)

Pre-compute BFS placements and cache them to `game_cache_v2/`:

```bash
uv run python preparse_games.py
```

This takes a long time but only needs to run once. Training falls back to on-the-fly BFS if the cache is absent.

## Training

```bash
uv run python model.py
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-path` | `data.csv` | Path to raw replay CSV |
| `--index-path` | `game_index.csv` | Path to game index |
| `--save-dir` | `checkpoints/` | Where to write checkpoints |
| `--epochs` | 25 | Number of training epochs |
| `--lr` | 2e-4 | Learning rate (AdamW) |
| `--batch-size` | 128 | Training batch size |
| `--k-candidates` | 10 | BFS candidates evaluated per move |
| `--num-workers` | 8 | DataLoader worker processes |
| `--rollout-depth` | 6 | Beam-search rollout depth for Q targets |
| `--rollout-beam` | 2 | Beam width during rollout |

A checkpoint is saved after each epoch: `checkpoints/tetrisformer_v4_ep{N}.pt`.

### Multiprocessing note

On Windows (and Python 3.14+ which defaults to `spawn`), worker functions must be globally importable. The codebase uses a `register_game_functions()` global registry to work around pickling restrictions. If you hit worker errors, try `--num-workers 0` to disable multiprocessing.

## Architecture

### Model: `TetrisFormerV4`

```
For each of K candidate placements:

  6-channel board (40×10)
      → CNN Encoder (3 conv layers, stride-2 pool ×2)
      → ~100 grid tokens (128-dim) + 2D sincos positional embeddings

  CLS token
  + Stats token  (7-dim game state projected to 128-dim)
  + Queue tokens (7 pieces: placed, hold, next×5)
      → Transformer Encoder (4 layers, 4 heads, 128-dim, FFN=512)
      → CLS output (128-dim)

CLS → Rank Head    (score for ranking this candidate)
    → Attack Head  (predicted immediate garbage sent)
    → Q Head       (predicted beam-search rollout return)
```

### Input channels (6 × 40 × 10)

| Ch | Description |
|----|-------------|
| 0 | Base board occupancy before placement |
| 1 | Result board occupancy after placement |
| 2 | Piece diff (newly filled cells) |
| 3 | Column heights (normalized, broadcast over rows) |
| 4 | Holes map (empty cells with blocks above) |
| 5 | Row fullness (fraction filled, broadcast over columns) |

### Stats vector (7-dim)

Incoming garbage, combo, back-to-back chain (all log-scaled), board max height, hole count, bumpiness, and game phase — each normalized to roughly [0, 1].

### Loss function

```
Loss = SoftRankLoss + Q_Loss + 0.5·ImitLoss + 0.1·AttackLoss
```

- **SoftRankLoss**: Cross-entropy against soft targets derived from beam-search rollout returns (temperature-sharpened softmax over candidate Q values)
- **Q_Loss**: SmoothL1 regression of per-candidate rollout return; target network updated via EMA (τ=0.005)
- **ImitLoss**: Hard cross-entropy against the expert's actual placement (imitation regularizer)
- **AttackLoss**: SmoothL1 regression of immediate garbage for the expert's placement

Moves with high rollout Q (≥ 1.5) get 6× sample weight to focus training on impactful decisions.

### Target network

A DQN-style EMA copy of the model is kept on CPU shared memory for stable Q-value leaf estimation during beam rollouts. It activates after epoch 5 (`--bootstrap-start-epoch`) and is updated each batch with `--target-ema-tau 0.005`.

## Inference scoring

At inference time, all three heads are combined:

```
Score(placement) = Rank_Score + α · (Attack_Score + Q_Score)
```

where `α` (`--rank-q-alpha`, default 0.3) balances imitation against the strategic estimate.

## Source files

| File | Role |
|------|------|
| `model.py` | Model definition (`TetrisFormerV4`), dataset (`SmartRolloutRankDataset`), beam rollout, training loop |
| `tetrisEngine.py` | Tetris simulation: piece spawning, SRS wall kicks, line clears, B2B/combo tracking, BFS placement enumeration |
| `fileParsing.py` | CSV loading, byte-offset game indexing, board parsing, BFS match lookup |
| `preparse_games.py` | Offline BFS cache builder (writes to `game_cache_v2/`) |

## Design notes

**Why beam-search rollout targets?** Expert replays contain many routine moves; simple imitation overweights low-value placements. Rolling out K candidates under the actual future piece sequence produces per-candidate quality estimates that distinguish T-spin setups, B2B preservation, and board cleanliness from equivalent-looking alternatives.

**Why a target network?** Q targets derived from the live model are non-stationary and can cause divergence. The EMA target network (kept on CPU shared memory) provides stable regression targets without the communication overhead of separate processes.

**Why SmoothL1 for regression?** Attack values are spiky (large spikes from 4-line clears, back-to-back bonuses). Huber/SmoothL1 is more robust to these outliers than MSE.
