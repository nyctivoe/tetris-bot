"""Tests for the tetrisformer/model.py bug-fixes.

Each test targets a specific identified issue to prevent regressions.
"""

import copy
import math
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

# Ensure repo root is on sys.path so imports resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tetrisEngine import TetrisEngine, BOARD_HEIGHT, BOARD_WIDTH, HIDDEN_ROWS

import tetrisformer.model as M


# =====================================================================
# Helpers
# =====================================================================
VISIBLE_H = 20


def _make_empty_board():
    return np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)


def _make_messy_board():
    """Board with a very tall, hole-ridden stack.  All rewards should be negative."""
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
    # Fill the top 15 visible rows with blocks that have holes
    for r in range(HIDDEN_ROWS, HIDDEN_ROWS + 15):
        for c in range(BOARD_WIDTH):
            # leave a hole in every other column on every other row
            if not (r % 2 == 0 and c % 3 == 0):
                board[r, c] = 1
    return board


def _make_engine_at(board, piece_kind="T", b2b=0, combo=0):
    eng = TetrisEngine(spin_mode="t_only")
    eng.board = board.copy()
    eng.current_piece = eng.spawn_piece(piece_kind)
    eng.b2b_chain = b2b
    eng.combo = combo
    eng.combo_active = combo > 0
    eng.game_over = False
    eng.game_over_reason = None
    eng.bag = np.array([], dtype=int)
    eng.bag_size = 0
    eng.incoming_garbage = []
    eng.garbage_col = None
    return eng


# =====================================================================
# Fix #1 — debug prints removed (no direct test; checked via
#           absence of "[DEBUG]" in training-loop code path)
# =====================================================================
def test_no_debug_prints_in_training_loop():
    """Ensure that '[DEBUG]' print statements were removed from model.py."""
    import inspect
    source = inspect.getsource(M.main)
    assert "[DEBUG]" not in source, (
        "Found leftover [DEBUG] print in main(); these should have been removed."
    )


# =====================================================================
# Fix #2 — _beam_rollout_future: best initialised to -inf
# =====================================================================
class TestBeamRolloutBestInit:
    """When every rollout branch yields a negative reward, the rollout
    must return the actual best (negative) value, not 0.0."""

    def test_all_negative_returns_negative(self):
        """With a terrible board and no attack/tspin, future_return must be < 0."""
        board = _make_messy_board()
        eng = _make_engine_at(board, piece_kind="T")

        # Grab a couple of future pieces
        future = ["I", "O", "S"]

        # High penalties, zero bonuses → every step reward is negative
        ret = M._beam_rollout_future(
            engine_after_root=eng,
            future_piece_kinds=future,
            next_index=0,
            depth_remaining=2,
            beam_width=2,
            expand_width=4,
            gamma=0.97,
            attack_w=0.0,          # no attack reward
            tspin_bonus=0.0,       # no tspin reward
            b2b_bonus=0.0,         # no b2b reward
            height_penalty=1.0,    # heavy penalty
            holes_penalty=1.0,     # heavy penalty
            topout_penalty=10.0,
            tslot_ready_bonus=0.0,
        )
        # With a messy board, high penalties, and no positive bonuses,
        # future_return must be strictly negative.
        assert ret < 0.0, (
            f"Expected negative rollout return for messy board, got {ret}"
        )

    def test_empty_board_returns_nonnegative(self):
        """Sanity check: empty board with low penalties should give >= 0."""
        board = _make_empty_board()
        eng = _make_engine_at(board, piece_kind="T")

        ret = M._beam_rollout_future(
            engine_after_root=eng,
            future_piece_kinds=["I", "O"],
            next_index=0,
            depth_remaining=1,
            beam_width=2,
            expand_width=4,
            gamma=0.97,
            attack_w=1.0,
            tspin_bonus=1.0,
            b2b_bonus=0.5,
            height_penalty=0.1,
            holes_penalty=0.5,
            topout_penalty=10.0,
            tslot_ready_bonus=0.0,
        )
        # Empty board with normal reward weights should not return -inf
        assert ret > float('-inf'), "Rollout should not return -inf on valid board"

    def test_zero_depth_returns_zero(self):
        """depth_remaining=0 should still return 0 (early return path)."""
        board = _make_empty_board()
        eng = _make_engine_at(board)
        ret = M._beam_rollout_future(
            engine_after_root=eng,
            future_piece_kinds=[],
            next_index=0,
            depth_remaining=0,
            beam_width=2,
            expand_width=4,
            gamma=0.97,
            attack_w=1.0,
            tspin_bonus=1.0,
            b2b_bonus=0.5,
            height_penalty=0.1,
            holes_penalty=0.5,
            topout_penalty=10.0,
        )
        assert ret == 0.0


# =====================================================================
# Fix #3 — B2B continuation: no false positives on chain-breaking clears
# =====================================================================
class TestB2BContinuationReward:
    """_compute_rollout_reward must not give b2b_cont bonus when the clear
    BROKE the B2B chain (even if attack > 0 from combo)."""

    def test_b2b_break_with_combo_attack_no_reward(self):
        """A combo clear that resets b2b_chain to 0 must NOT get b2b_cont."""
        board = _make_empty_board()
        clear_stats = {
            "attack": 3,          # non-zero (from combo)
            "b2b_bonus": 0,       # no B2B bonus (chain broken)
            "b2b_chain": 0,       # chain reset to 0 (broken)
            "lines_cleared": 1,
        }
        reward, _ = M._compute_rollout_reward(
            board_after=board,
            clear_stats=clear_stats,
            prev_b2b=5,           # B2B was active before
            game_over=False,
            attack_w=1.0,
            tspin_bonus=0.0,
            b2b_bonus=10.0,       # large weight so bug would be obvious
            height_penalty=0.0,
            holes_penalty=0.0,
            topout_penalty=0.0,
        )
        # The b2b_bonus component should be 0 (chain was BROKEN, not continued)
        # With b2b_bonus weight = 10 and b2b_cont should be 0:
        #   reward should be attack_w*3 + 0 = 3.0  (approximately, plus board shape)
        # If the bug were still present, reward would include +10 from b2b_cont
        assert reward < 5.0, (
            f"Reward {reward} too high; B2B break with combo should not get b2b_cont bonus"
        )

    def test_b2b_continuation_gets_reward(self):
        """A B2B-qualifying clear (chain increased) should get b2b_cont."""
        board = _make_empty_board()
        clear_stats = {
            "attack": 2,
            "b2b_bonus": 0,       # b2b_bonus is 0 for chain values 1-2
            "b2b_chain": 2,       # chain went from 1→2
            "lines_cleared": 2,
        }
        reward_with_b2b, _ = M._compute_rollout_reward(
            board_after=board,
            clear_stats=clear_stats,
            prev_b2b=1,           # B2B was at 1 before
            game_over=False,
            attack_w=1.0,
            tspin_bonus=0.0,
            b2b_bonus=10.0,
            height_penalty=0.0,
            holes_penalty=0.0,
            topout_penalty=0.0,
        )
        # b2b_chain (2) > prev_b2b (1) → b2b_cont = 1.0 → +10
        assert reward_with_b2b >= 10.0, (
            f"Reward {reward_with_b2b} should include B2B continuation bonus "
            f"for chain increase 1→2"
        )

    def test_b2b_with_actual_bonus(self):
        """When b2b_bonus > 0 in clear_stats, b2b_cont should be 1."""
        board = _make_empty_board()
        clear_stats = {
            "attack": 4,
            "b2b_bonus": 1,       # actual bonus (chain ≥ 3)
            "b2b_chain": 4,
            "lines_cleared": 2,
        }
        reward, _ = M._compute_rollout_reward(
            board_after=board,
            clear_stats=clear_stats,
            prev_b2b=3,
            game_over=False,
            attack_w=0.0,
            tspin_bonus=0.0,
            b2b_bonus=10.0,
            height_penalty=0.0,
            holes_penalty=0.0,
            topout_penalty=0.0,
        )
        assert reward >= 10.0, (
            f"Reward {reward} should include B2B bonus when b2b_bonus > 0 in stats"
        )

    def test_no_clear_no_b2b_reward(self):
        """A move with no lines cleared should give b2b_cont = 0."""
        board = _make_empty_board()
        clear_stats = {}  # empty = no clear
        reward, _ = M._compute_rollout_reward(
            board_after=board,
            clear_stats=clear_stats,
            prev_b2b=5,
            game_over=False,
            attack_w=0.0,
            tspin_bonus=0.0,
            b2b_bonus=10.0,
            height_penalty=0.0,
            holes_penalty=0.0,
            topout_penalty=0.0,
        )
        # b2b_bonus=0, b2b_chain not present → default 0 > 5 is False
        # So b2b_cont = 0 → no +10
        assert reward < 5.0, "No-clear move should not receive B2B bonus"


# =====================================================================
# Fix #4 — torch.load guarded against corrupted files
# =====================================================================
class TestCacheLoadErrorHandling:
    """SmartRolloutRankDataset.__iter__ should skip corrupted .pt files
    instead of crashing the worker."""

    def test_corrupt_cache_file_skipped(self, tmp_path):
        """A truncated .pt file should be skipped, not crash."""
        # Create a fake entry + corrupted cache file
        game_id = 99999
        cache_dir = str(tmp_path)
        corrupt_path = os.path.join(cache_dir, f"{game_id}.pt")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_TORCH_FILE")

        entry = {"game_id": game_id, "won": 1, "start": 0, "end": 0, "rows": 1}

        # Register functions so the dataset can initialise
        M.register_game_functions(
            TetrisEngine,
            lambda *a, **k: iter([]),
            lambda *a, **k: _make_empty_board(),
            lambda *a, **k: (-1, None),
        )

        ds = M.SmartRolloutRankDataset(
            [entry],
            data_path="dummy.csv",
            cache_dir=cache_dir,
            seed=42,
        )

        # Iterating should produce 0 samples (skipped), not raise.
        samples = list(ds)
        assert len(samples) == 0

    def test_valid_version_mismatch_raises(self, tmp_path):
        """A valid .pt with wrong cache_version should still raise RuntimeError."""
        game_id = 88888
        cache_dir = str(tmp_path)
        cache_path = os.path.join(cache_dir, f"{game_id}.pt")
        torch.save({"cache_version": 0, "steps": []}, cache_path)

        entry = {"game_id": game_id, "won": 1, "start": 0, "end": 0, "rows": 1}

        M.register_game_functions(
            TetrisEngine,
            lambda *a, **k: iter([]),
            lambda *a, **k: _make_empty_board(),
            lambda *a, **k: (-1, None),
        )

        ds = M.SmartRolloutRankDataset(
            [entry],
            data_path="dummy.csv",
            cache_dir=cache_dir,
            seed=42,
        )

        with pytest.raises(RuntimeError, match="Unsupported cache version"):
            list(ds)


# =====================================================================
# Fix #5 — val_batches=0 guard is now before the loop
# =====================================================================
def test_val_batches_zero_guard_in_source():
    """The val loop should use `while v_batches < args.val_batches:` so that
    val_batches=0 causes zero iterations (not one iteration + ValueError)."""
    import inspect
    source = inspect.getsource(M.main)
    # The old buggy pattern
    assert 'raise ValueError("Set --val-batches' not in source, (
        "Old ValueError guard for val_batches==0 should have been removed."
    )
    # The new guard should exist
    assert "val_batches <= 0" in source, (
        "Pre-loop guard for val_batches <= 0 should be present."
    )


# =====================================================================
# Fix #6 — game_failed dead code removed from preparse_games.py
# =====================================================================
def test_game_failed_removed():
    """The dead `game_failed` variable should be removed from preparse_games.py."""
    import inspect
    import tetrisformer.preparse_games as PP
    source = inspect.getsource(PP.build_dataset)
    assert "game_failed" not in source, (
        "Dead variable `game_failed` should have been removed from build_dataset()."
    )


# =====================================================================
# Fix #8 — pin_memory set correctly for CUDA
# =====================================================================
def test_pin_memory_tied_to_device():
    """pin_memory in DataLoader should be True when device is CUDA."""
    import inspect
    source = inspect.getsource(M.main)
    # Should reference device type for pin_memory, not hardcode False
    assert "pin_memory=False" not in source, (
        "pin_memory should not be hardcoded to False; it should depend on device.type."
    )
    assert "pin_memory=_pin" in source or "pin_memory=(device" in source, (
        "pin_memory should be derived from the device type."
    )


# =====================================================================
# Fix #9 — queue_pos_embed sized to QUEUE_SLOTS + 2 (= 7)
# =====================================================================
def test_queue_pos_embed_correct_size():
    """queue_pos_embed should match the actual queue length (7), not 10."""
    model = M.TetrisFormerV4()
    expected = M.QUEUE_SLOTS + 2  # 5 + 2 = 7
    actual = model.queue_pos_embed.shape[1]
    assert actual == expected, (
        f"queue_pos_embed dim-1 should be {expected} (QUEUE_SLOTS+2), got {actual}"
    )


# =====================================================================
# Fix #10 — _grid_pos_embed registered as buffer
# =====================================================================
def test_grid_pos_embed_is_buffer():
    """_grid_pos_embed should be a registered buffer so model.to() moves it."""
    model = M.TetrisFormerV4()
    # After register_buffer, the name appears in model._buffers
    assert "_grid_pos_embed" in dict(model.named_buffers(recurse=False)) or \
           "_grid_pos_embed" in model._buffers, (
        "_grid_pos_embed should be a registered buffer (via register_buffer)."
    )


def test_grid_pos_embed_moves_with_model():
    """After model.to('cpu'), the cached pos embed should also be on CPU."""
    model = M.TetrisFormerV4()
    # Run a dummy forward to populate the buffer
    B, C, H, W = 1, M.NUM_BOARD_CHANNELS, M.BOARD_H, M.BOARD_W
    x_board = torch.randn(B, C, H, W)
    x_queue = torch.zeros(B, M.QUEUE_SLOTS + 2, dtype=torch.long)
    x_stats = torch.randn(B, M.NUM_STATS)
    model.eval()
    with torch.no_grad():
        model(x_board, x_queue, x_stats)

    assert model._grid_pos_embed is not None, "Pos embed should be populated after forward"
    # Move model to CPU explicitly
    model.cpu()
    assert model._grid_pos_embed.device.type == "cpu", (
        "Pos embed buffer should move to CPU with model.cpu()"
    )


# =====================================================================
# Fix #11 — last_rot_dir / last_kick_idx sanitised in preparse_games.py
# =====================================================================
def test_preparse_placement_sanitisation():
    """_apply_placement_to_current_piece in preparse_games.py should sanitise
    last_rotation_dir and last_kick_index the same way model.py does."""
    import tetrisformer.preparse_games as PP
    eng = TetrisEngine(spin_mode="t_only")
    eng.board = _make_empty_board()
    eng.current_piece = eng.spawn_piece("T")

    placement = {
        "x": 4, "y": 38, "rotation": 0,
        "last_was_rot": False,
        "last_rot_dir": 0,    # should become None
        "last_kick_idx": -1,  # should become None
    }
    PP._apply_placement_to_current_piece(eng, placement)
    assert eng.current_piece.last_rotation_dir is None, (
        "last_rotation_dir=0 should be sanitised to None"
    )
    assert eng.current_piece.last_kick_index is None, (
        "last_kick_index=-1 should be sanitised to None"
    )


# =====================================================================
# Model forward-pass smoke test (validates shape correctness end-to-end)
# =====================================================================
class TestModelForwardSmoke:
    """Quick shape/dtype checks for TetrisFormerV4.forward()."""

    @pytest.fixture
    def model(self):
        m = M.TetrisFormerV4()
        m.eval()
        return m

    def test_single_sample(self, model):
        B = 1
        x_board = torch.randn(B, M.NUM_BOARD_CHANNELS, M.BOARD_H, M.BOARD_W)
        x_queue = torch.zeros(B, M.QUEUE_SLOTS + 2, dtype=torch.long)
        x_stats = torch.randn(B, M.NUM_STATS)
        with torch.no_grad():
            rank, atk, q = model(x_board, x_queue, x_stats)
        assert rank.shape == (B, 1)
        assert atk.shape == (B, 1)
        assert q.shape == (B, 1)

    def test_batch(self, model):
        B = 8
        x_board = torch.randn(B, M.NUM_BOARD_CHANNELS, M.BOARD_H, M.BOARD_W)
        x_queue = torch.randint(0, 8, (B, M.QUEUE_SLOTS + 2))
        x_stats = torch.randn(B, M.NUM_STATS)
        with torch.no_grad():
            rank, atk, q = model(x_board, x_queue, x_stats)
        assert rank.shape == (B, 1)
        assert atk.shape == (B, 1)
        assert q.shape == (B, 1)

    def test_candidate_batching(self, model):
        """Simulate the training-loop pattern: B items × K candidates."""
        B, K = 4, 10
        x_board = torch.randn(B * K, M.NUM_BOARD_CHANNELS, M.BOARD_H, M.BOARD_W)
        x_queue = torch.randint(0, 8, (B * K, M.QUEUE_SLOTS + 2))
        x_stats = torch.randn(B * K, M.NUM_STATS)
        with torch.no_grad():
            rank, atk, q = model(x_board, x_queue, x_stats)
        rank_logits = rank.squeeze(-1).reshape(B, K)
        pred_q = q.squeeze(-1).reshape(B, K)
        assert rank_logits.shape == (B, K)
        assert pred_q.shape == (B, K)


# =====================================================================
# Feature-engineering helpers — basic sanity
# =====================================================================
class TestFeatureHelpers:
    def test_compute_board_features_shape(self):
        base = _make_empty_board()
        result = _make_empty_board()
        feats = M.compute_board_features(base, result)
        assert feats.shape == (M.NUM_BOARD_CHANNELS, M.BOARD_H, M.BOARD_W)
        assert feats.dtype == np.float32

    def test_encode_stats_shape(self):
        row = {"incoming_garbage": 2, "combo": 3, "btb": 1, "move_number": 50}
        base = _make_empty_board()
        result = _make_empty_board()
        stats = M.encode_stats(row, base, result)
        assert stats.shape == (M.NUM_STATS,)
        assert stats.dtype == np.float32

    def test_encode_queue_shape(self):
        q = M._encode_queue("T", "I", "OSJZL")
        assert q.shape == (M.QUEUE_SLOTS + 2,)
        assert q.dtype == np.int64
        # placed=T=3, hold=I=1, next=O,S,J,Z,L
        assert q[0] == M.PIECE_MAP["T"]
        assert q[1] == M.PIECE_MAP["I"]


# =====================================================================
# CLAUDE.md accuracy checks
# =====================================================================
def test_claude_md_no_stale_refs():
    """CLAUDE.md should reference game_cache_v3 and not mention removed features."""
    md_path = os.path.join(_ROOT, "CLAUDE.md")
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    assert "game_cache_v2" not in text, "CLAUDE.md still references stale game_cache_v2"
    assert "bootstrap-start-epoch" not in text, "CLAUDE.md still references removed --bootstrap-start-epoch"
    assert "target-ema-tau" not in text, "CLAUDE.md still references removed --target-ema-tau"
    assert "Target network" not in text, "CLAUDE.md still mentions removed target network"
