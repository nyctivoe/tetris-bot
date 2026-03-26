# Gradient Clipping Implementation Plan for TetrisFormer V4

**Project:** TetrisFormer V4
**Date:** 2026-03-25
**Goal:** Add gradient clipping to stabilize training against noisy rollout Q-values

---

## Overview

Gradient clipping prevents gradients from becoming too large during backpropagation, which can cause training instability or NaN losses. This is especially important for TetrisFormer because the rollout Q-values are noisy (beam search approximation, sparse rewards, Monte Carlo variance).

**What needs to change:**
- One line added after `loss.backward()`
- One optional helper function for gradient monitoring
- Logging additions to track gradient norms

---

## Step-by-Step Implementation

### Step 1: Add Gradient Clipping Call

**Location:** `model.py`, training loop, after `loss.backward()` and before `optimizer.step()`

**Current code (around line 700-720):**
```python
loss.backward()
optimizer_step(optimizer)
```

**Change to:**
```python
loss.backward()

# Gradient clipping: prevent gradients from exploding due to noisy Q-values
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer_step(optimizer)
```

**Note:** `torch.nn.utils.clip_grad_norm_` returns the total gradient norm before clipping, which is useful for monitoring.

---

### Step 2: Add Gradient Norm Logging (Optional but Recommended)

**Location:** Same section of the training loop where you log loss metrics

**Current code (around line 730-750):**
```python
if args.log_every > 0 and (total_samples % args.log_every) < B:
    master_print(
        f"Ep {epoch} | Samples {total_samples} | Batches {total_batches} | "
        f"Loss {total_loss/n:.4f} "
        f"(SoftRank {total_soft/n:.4f}, Q {total_q/n:.4f}, "
        f"Atk {total_atk/n:.4f}, Imit {total_imit/n:.4f}) | "
        f"ExpertAcc {100.0*total_expert_acc/n:.2f}% | "
        f"QBestAcc {100.0*total_qbest_acc/n:.2f}% | "
        f"FinalAcc {100.0*total_final_acc/n:.2f}%"
    )
```

**Change to:**
```python
if args.log_every > 0 and (total_samples % args.log_every) < B:
    master_print(
        f"Ep {epoch} | Samples {total_samples} | Batches {total_batches} | "
        f"Loss {total_loss/n:.4f} "
        f"(SoftRank {total_soft/n:.4f}, Q {total_q/n:.4f}, "
        f"Atk {total_atk/n:.4f}, Imit {total_imit/n:.4f}) | "
        f"GradNorm {total_norm:.4f} | "  # NEW
        f"ExpertAcc {100.0*total_expert_acc/n:.2f}% | "
        f"QBestAcc {100.0*total_qbest_acc/n:.2f}% | "
        f"FinalAcc {100.0*total_final_acc/n:.2f}%"
    )
```

---

### Step 3: Track Gradient Statistics (Optional Enhancement)

Add a running tracker for gradient norms to detect trends:

**Add near the top of the training loop (after optimizer initialization):**
```python
# Gradient norm tracking
grad_norms = []  # List to store gradient norms for analysis
```

**Modify the gradient clipping section:**
```python
loss.backward()

# Gradient clipping: prevent gradients from exploding due to noisy Q-values
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
grad_norms.append(total_norm)

optimizer_step(optimizer)
```

**Add periodic gradient stats logging (every epoch end):**
```python
# At the end of each epoch, log gradient statistics
if len(grad_norms) > 0:
    import numpy as np
    grad_mean = np.mean(grad_norms)
    grad_std = np.std(grad_norms)
    grad_max = np.max(grad_norms)
    grad_min = np.min(grad_norms)
    clipped_count = sum(1 for n in grad_norms if n > 1.0)
    clipped_pct = 100.0 * clipped_count / len(grad_norms)
    
    master_print(
        f"  [Gradients] Mean: {grad_mean:.4f} | Std: {grad_std:.4f} | "
        f"Min: {grad_min:.4f} | Max: {grad_max:.4f} | "
        f"Clipped: {clipped_pct:.1f}% ({clipped_count}/{len(grad_norms)})"
    )
    grad_norms.clear()  # Reset for next epoch
```

---

## Exact Code Diff

```diff
--- a/model.py
+++ b/model.py
@@ -695,8 +695,12 @@ def main():
             master_print(f"  [DEBUG] Batch {total_batches + 1} fetched. Running forward/backward pass...")

             # ... batch preparation code ...

             optimizer.zero_grad(set_to_none=True)

             rank_scores, pred_attack, pred_q = model(boards_flat, queues_flat, stats_flat)
             # ... loss computation ...

             loss = (
                 args.soft_rank_loss_weight * soft_rank_loss
                 + args.q_loss_weight * q_loss
                 + args.attack_loss_weight * attack_loss
                 + args.imit_loss_weight * imit_loss
             )

             if not torch.isfinite(loss):
                 master_print(f"[WARN] Non-finite loss at epoch {epoch}; skipping batch.")
                 continue

             loss.backward()
+
+            # Gradient clipping: prevent gradients from exploding due to noisy Q-values
+            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
+
             optimizer_step(optimizer)

             # Soft-update the EMA target network after every gradient step.
@@ -725,6 +729,7 @@ def main():
             if args.log_every > 0 and (total_samples % args.log_every) < B:
                 master_print(
                     f"Ep {epoch} | Samples {total_samples} | Batches {total_batches} | "
                     f"Loss {total_loss/n:.4f} "
                     f"(SoftRank {total_soft/n:.4f}, Q {total_q/n:.4f}, "
                     f"Atk {total_atk/n:.4f}, Imit {total_imit/n:.4f}) | "
+                    f"GradNorm {total_norm:.4f} | "
                     f"ExpertAcc {100.0*total_expert_acc/n:.2f}% | "
                     f"QBestAcc {100.0*total_qbest_acc/n:.2f}% | "
                     f"FinalAcc {100.0*total_final_acc/n:.2f}%"
```

---

## Validation Plan

### 1. Smoke Test (1 epoch)

**Command:**
```bash
uv run python model.py --epochs 1 --batch-size 128 --samples-per-epoch 5000
```

**What to verify:**
- [ ] Training runs without errors
- [ ] No NaN losses printed
- [ ] GradNorm values in logs are reasonable (0.1-2.0)
- [ ] Training loss decreases over the epoch

### 2. Longer Training Test (3 epochs)

**Command:**
```bash
uv run python model.py --epochs 3 --batch-size 128 --samples-per-epoch 10000
```

**What to verify:**
- [ ] All 3 epochs complete successfully
- [ ] GradNorm stays consistent (not growing unbounded)
- [ ] ExpertAcc improves over epochs
- [ ] FinalAcc improves over epochs
- [ ] Clipping rate < 20% of batches

### 3. Stress Test (with noisy Q-values)

**Command:**
```bash
uv run python model.py --epochs 5 --rollout-depth 9 --rollout-beam 2
```

**What to verify:**
- [ ] Deep rollouts don't cause gradient explosions
- [ ] GradNorm spikes are clipped appropriately
- [ ] Training remains stable

---

## Monitoring Checklist

### During Training (Live)

| Metric | Healthy Range | Warning Sign | Action |
|--------|---------------|--------------|--------|
| `GradNorm` | 0.1 - 1.0 | > 2.0 | Check rollout logic |
| `GradNorm` | - | > 5.0 | Increase max_norm to 2.0 |
| `Loss` | Decreasing | NaN | Reduce max_norm or LR |
| `ExpertAcc` | Increasing | Decreasing | Check learning rate |

### At Epoch End (Summary)

| Metric | Healthy | Warning |
|--------|----------|---------|
| Mean GradNorm | 0.3 - 0.8 | > 1.0 |
| Std GradNorm | 0.1 - 0.3 | > 0.5 |
| Max GradNorm | < 2.0 | > 5.0 |
| Clipped % | < 10% | > 30% |

---

## Potential Pitfalls

### 1. Forgetting to Import torch.nn.utils

**Symptom:** `AttributeError: module 'torch.nn.utils' has no attribute 'clip_grad_norm_'`

**Fix:** Ensure `torch` is imported (it already is in model.py)

### 2. Clipping Before vs After optimizer.step()

**Problem:** Clipping after `optimizer.step()` has no effect.

**Fix:** Always clip between `loss.backward()` and `optimizer.step()`:
```python
loss.backward()       # Compute gradients
clip_grad_norm_(...)   # Clip gradients  ← correct position
optimizer.step()       # Apply gradients
```

### 3. max_norm Too Restrictive

**Symptom:** GradNorm always equals max_norm, "Clipped: 100%"

**Fix:** Increase max_norm to 2.0 or 3.0

### 4. max_norm Too Permissive

**Symptom:** GradNorm never exceeds 0.3, no clipping occurs

**Fix:** Reduce max_norm to 0.5 for tighter control

### 5. Incompatible with AMP (Automatic Mixed Precision)

**If using AMP:** Use `unscale_grad` before clipping:
```python
from torch.cuda.amp import GradScaler

scaler = GradScaler()

with autocast():
    # forward pass
    loss = ...

scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # Required before clipping with AMP
clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

---

## Alternative: CLI Argument for max_norm

For flexibility, add `max_norm` as a command-line argument:

```python
parser.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="Maximum gradient norm for clipping (0 to disable).")

# In training loop:
if args.max_grad_norm > 0:
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=args.max_grad_norm
    )
else:
    total_norm = 0.0  # No clipping

# Logging:
master_print(f"  [GradClip] max_norm={args.max_grad_norm}, "
            f"grad_norm={total_norm:.4f}")
```

---

## Implementation Time Estimate

| Task | Time | Complexity |
|------|------|------------|
| Add clipping call | 1 minute | Trivial |
| Add logging | 2 minutes | Trivial |
| Add CLI argument (optional) | 5 minutes | Easy |
| Test 1 epoch | 5-10 minutes | Easy |
| Verify logs | 1 minute | Easy |
| **Total** | **15-20 minutes** | - |

---

## Quick Reference

### Minimum Implementation (2 lines changed)

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ADD THIS
optimizer_step(optimizer)
```

### With Monitoring (4 lines changed)

```python
loss.backward()
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ADD THIS
optimizer_step(optimizer)

# In logging (add "GradNorm {total_norm:.4f} |" to existing log format)
```

---

## Next Steps After Implementation

1. **Monitor for 1-2 epochs** - watch GradNorm values
2. **Adjust max_norm if needed:**
   - If clipped < 5% → reduce to 0.5
   - If clipped > 30% → increase to 2.0
3. **Consider adaptive clipping** if gradient norms are highly variable
4. **Reduce weight decay** to 1e-3 (currently 1e-2) for better synergy with clipping
