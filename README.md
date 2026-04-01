# TetrisFormer

TetrisFormer is a CNN + transformer model trained on expert TETR.IO replays. The current training path uses rollout-ranked BFS candidates, replay-aligned T-only spin semantics, and cache v3 preprocessing.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- CUDA recommended, CPU supported

Install dependencies with:

```bash
uv sync
```

## Quick Start

```bash
# Build the byte-offset index if game_index.csv does not exist yet.

# Precompute cache v3 from replay data.
uv run python preparse_games.py --cache-dir game_cache_v3

# Train against cache v3.
uv run python model.py --cache-dir game_cache_v3
```

Smoke commands used for the current pipeline:

```bash
uv run python preparse_games.py --limit-games 10 --cache-dir game_cache_v3
uv run python tools/replay_alignment_report.py --cache-dir game_cache_v3 --sample-size 5000
uv run python model.py --cache-dir game_cache_v3 --num-workers 0 --batch-size 2 --batches-per-epoch 1 --val-batches 1
```

## Source Files

| File | Role |
|------|------|
| `model.py` | `TetrisFormerV4`, cache v3 dataset loading, rollout target generation, training loop |
| `tetrisEngine.py` | Tetris simulation engine, BFS placement generation, default T-only spin detection |
| `fileParsing.py` | CSV loading, replay indexing, board parsing, BFS/result alignment |
| `preparse_games.py` | Cache v3 builder |
| `tools/replay_alignment_report.py` | Replay-vs-engine audit for cache v3 |

## Model Architecture

```text
7-channel board features (40x10)
  -> Conv2d(7 -> 32, 3x3) + BN + ReLU
  -> Residual block (32 -> 64, stride 2)
  -> Residual block (64 -> 192, stride 2)
  -> 30 grid tokens (10x3, 192-dim) + 2D sin/cos positional encoding

Queue tokens: 7 total
  current/placed + hold + 5 next pieces

Transformer encoder
  d_model=192
  depth=4
  heads=6
  ffn=768
  dropout=0.1
  norm_first=True

Heads
  rank:   192 -> 256 -> 1
  attack: 192 -> 128 -> 1
  q:      192 -> 256 -> 64 -> 1
```

Board feature channels are: base occupancy, result occupancy, placement diff, height map, holes, row fill ratio, and T-slot cavity heuristic.

Inference score:

```text
score = rank + alpha * q
```

Default `alpha` is `0.3`.

## Training Semantics

- Cache version: `game_cache_v3`
- Spin mode: replay-aligned T-only by default
- Bootstrap / EMA target network: disabled
- Checkpoint filename: `tetrisformer_v4r1_ep{epoch}.pt`
- Checkpoint metadata includes `cache_version=3`, `spin_mode="t_only"`, and `bootstrap_enabled=false`

Validation now resets each epoch. Training hard-fails on non-v3 caches and instructs you to rerun preprocessing.

## Training Defaults

| Parameter | Default |
|-----------|---------|
| `--k-candidates` | `10` |
| `--batch-size` | `128` |
| `--epochs` | `25` |
| `--lr` | `2e-4` |
| `--rollout-depth` | `6` |
| `--rollout-beam` | `2` |
| `--rollout-expand` | `6` |
| `--gamma` | `0.97` |
| `--soft-target-temp` | `0.5` |
| `--q-target-scale` | `20.0` |
| `--attack-target-scale` | `12.0` |
| `--grad-clip` | `1.0` |
| `--warmup-steps` | `500` |
| `--min-lr` | `1e-6` |

Reward defaults:

| Parameter | Default |
|-----------|---------|
| `--reward-attack-weight` | `1.0` |
| `--reward-tspin-bonus` | `1.5` |
| `--reward-b2b-bonus` | `0.5` |
| `--reward-height-penalty` | `0.1` |
| `--reward-holes-penalty` | `0.5` |
| `--reward-topout-penalty` | `10.0` |
| `--reward-tslot-ready-bonus` | `0.3` |

Other defaults:

- importance weighting: `6x` for line-clear / attack / high-Q samples
- label smoothing: `0.05`
- train/val split: `90/10`

## Data Artifacts

| Path | Description |
|------|-------------|
| `data.csv` | Raw replay rows |
| `game_index.csv` | Byte-offset replay index |
| `game_cache_v3/` | Required preprocessed cache for training |
| `checkpoints/` | Saved checkpoints |

## CLI Notes

Useful arguments:

- `model.py`: `--cache-dir`, `--data-path`, `--index-path`, `--num-workers`, `--batch-size`, `--batches-per-epoch`, `--val-batches`
- `preparse_games.py`: `--cache-dir`, `--overwrite`, `--limit-games`
- `tools/replay_alignment_report.py`: `--cache-dir`, `--sample-size`, `--seed`

Use `--num-workers 0` if multiprocessing causes issues on your machine.
