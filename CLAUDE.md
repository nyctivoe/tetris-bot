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
Key CLI args: `--data-path`, `--index-path`, `--epochs`, `--lr`, `--batch-size`, `--k-candidates`, `--num-workers`

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
3. The expert's actual move is matched against BFS candidates. K=64 candidates are sampled (1 expert + 63 negatives).
4. Each candidate is encoded into **6 channels** (40×10): base board, result board, piece diff, column heights, holes map, row fullness.
5. `SmartRolloutRankDataset` (IterableDataset) handles this pipeline per-worker, with optional cache loading from `game_cache_v2/`.

### Model: `TetrisFormerV4`

```
6-channel board (40×10)
    → CNN Encoder (3 conv layers, stride-2 pool twice) → 100 grid tokens (128-dim)

CLS token + Stats token (7-dim game state) + Queue tokens (7 pieces)
    → Transformer Encoder (4 layers, 4 heads, 128-dim)
    → CLS output

CLS → Rank Head    (CrossEntropyLoss, label_smoothing=0.1)
    → Attack Head  (SmoothL1Loss, immediate garbage)
    → Q Head       (SmoothL1Loss, beam-search rollout return)
```

**Loss**: `RankLoss + 0.5·AttackLoss + 0.5·QLoss + λ·ImitationLoss`

**Target network**: DQN-style EMA copy (tau=0.005) kept on CPU shared memory for stable Q targets.

**Importance weighting**: Moves with line clears or high future value (≥0.5) get 8× sample weight.

### Inference Scoring

```
Score(placement) = Rank_Score + α·(Attack_Score + Value_Score)
```

### Multiprocessing Note

On Windows/Python 3.14+, spawn is the default start method. Worker functions must be globally importable (not closures/lambdas). The codebase uses a `register_game_functions()` global registry pattern to work around pickling restrictions. Use `--num-workers 0` to disable multiprocessing if issues arise.

## Key Data Files

- `data.csv` — 7.7 GB raw game replays (22 columns: board state, piece info, ratings, attack stats)
- `game_index.csv` — byte-offset index for fast random game access
- `game_cache_v2/` — 48 GB preprocessed BFS states (optional but speeds up training)
- `checkpoints/tetrisformer_v4_ep1.pt` — saved model weights (V4 architecture)
