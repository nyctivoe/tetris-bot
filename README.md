# Tetris Bot

The legacy TetrisFormer architecture now lives under `tetrisformer/` so the repo root stays clear for the TetrisZero rewrite described in `docs/arch_2.md`. `tetrisEngine.py` remains at the root because both architectures reuse it.

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
uv run python -m tetrisformer.preparse_games --cache-dir game_cache_v3

# Train against cache v3.
uv run python -m tetrisformer.model --cache-dir game_cache_v3
```

## TetrisZero Training

The root-level TetrisZero stack writes training runs into a dedicated run directory.

Recommended one-command pipeline:

```bash
uv run python -m rl.pipeline --run-dir runs/exp1 --cycles 5 --episodes-per-cycle 50 --max-plies 32 --epochs-per-cycle 1 --batch-size 16 --device cuda --eval-games 4 --eval-max-plies 32
```

You can still run the two steps separately:

```bash
uv run python -m rl.selfplay_v2 --mode pvp --episodes 50 --max-plies 32 --output-dir runs/exp1/replay_buffer
uv run python -m rl.training --buffer-dir runs/exp1/replay_buffer --epochs 1 --batch-size 16 --device cuda --checkpoint-dir runs/exp1/checkpoints --checkpoint-name tetriszero_manual.pt
```

### Run Layout

Each training run under `runs/<name>/` uses this layout:

| Path | Description |
|------|-------------|
| `runs/<name>/run_config.json` | Frozen CLI/config used for the run |
| `runs/<name>/status.json` | Latest pipeline status and current checkpoint |
| `runs/<name>/replay_buffer/` | Replay shards and replay manifest |
| `runs/<name>/checkpoints/` | Periodic TetrisZero checkpoints |
| `runs/<name>/checkpoints/latest.json` | Pointer to the latest checkpoint |
| `runs/<name>/logs/cycle_metrics.csv` | One row per cycle with validation/eval metrics |
| `runs/<name>/logs/train_history.csv` | One row per optimizer step |
| `runs/<name>/logs/events.jsonl` | Structured event stream for cycle/train/eval events |
| `runs/<name>/reports/latest_eval.json` | Latest arena evaluation summary |

### Practical Notes

- `--max-plies` is match length in rounds, not individual search steps.
- `--selfplay-source beam` is the current stable default.
- Periodic eval runs current checkpoint vs beam and vs the previous checkpoint when available.
- Generated runs and artifacts are ignored by git via `.gitignore`.

Smoke commands used for the current pipeline:

```bash
uv run python -m tetrisformer.preparse_games --limit-games 10 --cache-dir game_cache_v3
uv run python -m tetrisformer.tools.replay_alignment_report --cache-dir game_cache_v3 --sample-size 5000
uv run python -m tetrisformer.model --cache-dir game_cache_v3 --num-workers 0 --batch-size 2 --batches-per-epoch 1 --val-batches 1
```

## Source Files

| File | Role |
|------|------|
| `tetrisformer/model.py` | `TetrisFormerV4`, cache v3 dataset loading, rollout target generation, training loop |
| `tetrisEngine.py` | Tetris simulation engine, BFS placement generation, default T-only spin detection |
| `tetrisformer/fileParsing.py` | CSV loading, replay indexing, board parsing, BFS/result alignment |
| `tetrisformer/preparse_games.py` | Cache v3 builder |
| `tetrisformer/tools/replay_alignment_report.py` | Replay-vs-engine audit for cache v3 |

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

- `tetrisformer.model`: `--cache-dir`, `--data-path`, `--index-path`, `--num-workers`, `--batch-size`, `--batches-per-epoch`, `--val-batches`
- `tetrisformer.preparse_games`: `--cache-dir`, `--overwrite`, `--limit-games`
- `tetrisformer.tools.replay_alignment_report`: `--cache-dir`, `--sample-size`, `--seed`

Use `--num-workers 0` if multiprocessing causes issues on your machine.
