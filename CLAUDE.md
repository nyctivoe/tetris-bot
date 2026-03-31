# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TetrisFormer is a transformer-based deep learning model that learns to play Tetris from expert human replays (76,693 games, 7.7 GB CSV). It uses a CNN + transformer hybrid with three output heads trained via multi-task imitation learning and Q-learning.

## Commands

**Install dependencies** (requires `uv`):
```bash
uv sync
```

**Run training**:
```bash
uv run python model.py
```
Key CLI args: `--data-path`, `--index-path`, `--epochs`, `--lr`, `--batch-size`, `--k-candidates`, `--num-workers`, `--rollout-depth`, `--rollout-beam`, `--samples-per-epoch`

**Preprocess/cache BFS states** (one-time, outputs to `game_cache_v2/`):
```bash
uv run python preparse_games.py
```

There is no test suite.

## Documentation Conventions

- Use `docs/` for local planning notes, recommendations, implementation plans, scratch documentation, and other ad-hoc markdown files created during work.
- The `docs/` folder is gitignored on purpose, so files placed there will stay local and not be committed unless explicitly requested.
- Only put documentation in the repo root when it is intended to be part of the tracked project itself (for example: `README.md` or other intentionally versioned project docs).
- If asked to "add docs" without further clarification, prefer creating them under `docs/`.

## Architecture

Four source files:

| File | Role |
|------|------|
| `model.py` | Model definition, dataset, training loop |
| `tetrisEngine.py` | Tetris simulation + BFS candidate generation |
| `fileParsing.py` | CSV loading, game indexing, board parsing |
| `preparse_games.py` | Offline BFS cache builder |

### Data Flow

1. `fileParsing.py` reads `data.csv` (7.7 GB) using byte offsets from `game_index.csv` (2.5 MB index of 76,693 games).
2. For each expert move, `tetrisEngine.py`'s `TetrisEngine.bfs_all_placements()` enumerates all valid piece placements via BFS (SRS wall kicks).
3. The expert's actual move is matched against BFS candidates. K=10 candidates are sampled (1 expert + 9 negatives) by default.
4. Each candidate is encoded into **7 channels** (40×10): base board, result board, piece diff, column heights, holes map, row fullness, T-slot cavity map.
5. `SmartRolloutRankDataset` (IterableDataset) handles this pipeline per-worker, loading from the BFS cache in `game_cache_v2/`.

### Model: `TetrisFormerV4`

```
7-channel board (40×10)
    → CNN Encoder (3 conv layers, stride-2 on layers 2–3) → 30 grid tokens (128-dim)

CLS token + Stats token (7-dim game state) + Queue tokens (7: placed + hold + 5 next)
    → Transformer Encoder (4 layers, 4 heads, 128-dim, FFN=512)
    → CLS output

CLS → Rank Head    (Linear 128→256→1)
    → Attack Head  (Linear 128→128→1, SmoothL1Loss)
    → Q Head       (Linear 128→256→64→1 with Dropout(0.1), SmoothL1Loss)
```

**Loss**: `1.0·SoftRankLoss + 1.0·QLoss + 0.1·AttackLoss + 0.5·ImitationLoss` (defaults). The imitation term is `CrossEntropyLoss(label_smoothing=0.05)` applied to the rank logits.

**Target network**: DQN-style EMA copy (tau=0.005) kept on CPU shared memory for stable Q targets. Bootstrapped leaf values activate at epoch 5 by default.

**Importance weighting**: Moves with line clears, attack > 0, or expert Q ≥ 1.5 get 6× sample weight.

**Rollout**: Beam-search with depth=6, beam=2, expand=6, gamma=0.97 (defaults). Soft rank targets are generated from rollout returns at temperature=0.5.

**Gradient clipping**: `--grad-clip` (default=1.0). Set to 0 to disable.

**LR schedule**: Linear warmup (`--warmup-steps`, default=500) then cosine decay to `--min-lr` (default=1e-6).

### Inference Scoring

```
Score(placement) = Rank_Score + α · Q_Score    (default α = 0.3)
```

Note: Attack predictions are not included in inference scoring.

### Multiprocessing Note

On Windows/Python 3.14+, spawn is the default start method. Worker functions must be globally importable (not closures/lambdas). The codebase uses a `register_game_functions()` global registry pattern to work around pickling restrictions. Use `--num-workers 0` to disable multiprocessing if issues arise.

## Key Data Files

- `data.csv` — 7.7 GB raw game replays (board state, piece info, ratings, attack stats)
- `game_index.csv` — byte-offset index for fast random game access
- `game_cache_v2/` — ~48 GB preprocessed BFS states (required for training; build with `preparse_games.py`)
- `checkpoints/` — saved model weights (V4 architecture)

## Current Training Defaults (CLI)

These are the argparse defaults in `model.py` `main()`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--k-candidates` | 10 | Candidates per move |
| `--batch-size` | 128 | |
| `--lr` | 2e-4 | AdamW, weight_decay=1e-2 |
| `--epochs` | 25 | |
| `--rollout-depth` | 6 | |
| `--rollout-beam` | 2 | |
| `--rollout-expand` | 6 | |
| `--gamma` | 0.97 | |
| `--soft-target-temp` | 0.5 | |
| `--q-target-scale` | 20.0 | |
| `--attack-target-scale` | 12.0 | |
| `--soft-rank-loss-weight` | 1.0 | |
| `--q-loss-weight` | 1.0 | |
| `--attack-loss-weight` | 0.1 | |
| `--imit-loss-weight` | 0.5 | |
| `--important-weight` | 6.0 | |
| `--important-q-threshold` | 1.5 | |
| `--rank-q-alpha` | 0.3 | Inference scoring blend |
| `--bootstrap-start-epoch` | 3 | |
| `--target-ema-tau` | 0.005 | |
| `--samples-per-epoch` | 250,000 | |
| `--grad-clip` | 1.0 | Set to 0 to disable |
| `--warmup-steps` | 500 | Linear LR warmup before cosine decay |
| `--min-lr` | 1e-6 | Cosine decay floor |
| `--max-garbage` | 8 | Max incoming garbage rows per move |
| `--seed` | 6767 | |
| `--reward-attack-weight` | 1.0 | |
| `--reward-tspin-bonus` | 1.5 | |
| `--reward-b2b-bonus` | 0.5 | |
| `--reward-height-penalty` | 0.1 | |
| `--reward-holes-penalty` | 0.5 | |
| `--reward-topout-penalty` | 10.0 | |
