from __future__ import annotations

import numpy as np

from tetrisEngine import TetrisEngine

from .fixtures import make_non_t_all_spin_piece, make_piece


def test_classify_clear_marks_difficult_and_break_paths():
    engine = TetrisEngine(spin_mode="all_spin")

    difficult = engine.classify_clear(4, None)
    breaking = engine.classify_clear(1, None)

    assert difficult["is_difficult"] is True
    assert difficult["qualifies_b2b"] is True
    assert breaking["breaks_b2b"] is True
    assert breaking["qualifies_b2b"] is False


def test_compute_attack_for_clear_releases_surge_on_break():
    engine = TetrisEngine(spin_mode="all_spin")
    board_after_clear = np.ones((40, 10), dtype=np.int16)

    attack = engine.compute_attack_for_clear(
        1,
        None,
        board_after_clear=board_after_clear,
        combo=2,
        combo_active=True,
        b2b_chain=6,
        surge_charge=5,
    )

    assert attack["breaks_b2b"] is True
    assert attack["surge_send"] == 5
    assert attack["surge_charge"] == 0
    assert attack["b2b_chain"] == 0


def test_predict_post_lock_stats_matches_execute_without_mutation(seeded_engine):
    before_board = seeded_engine.board.copy()
    bfs_result = seeded_engine.bfs_all_placements(include_no_place=False, dedupe_final=True)[0]

    probe = seeded_engine.clone()
    assert probe.apply_placement(bfs_result["placement"]) is True
    predicted = probe.predict_post_lock_stats(probe.current_piece)

    executed = seeded_engine.clone()
    outcome = executed.execute_placement(bfs_result["placement"], run_end_phase=True)

    assert np.array_equal(seeded_engine.board, before_board)
    assert predicted["stats"]["attack"] == outcome["stats"]["attack"]
    assert predicted["stats"]["lines_cleared"] == outcome["stats"]["lines_cleared"]


def test_queue_snapshot_bag_remainder_and_pending_summary(opener_cancel_engine):
    opener_cancel_engine.hold = 1
    opener_cancel_engine.bag = np.array([3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 6, 7], dtype=np.int64)
    opener_cancel_engine.bag_size = len(opener_cancel_engine.bag)

    queue = opener_cancel_engine.get_queue_snapshot(next_slots=5)
    bag = opener_cancel_engine.get_bag_remainder_counts()
    pending = opener_cancel_engine.get_pending_garbage_summary()

    assert queue["hold"] == "I"
    assert queue["next_kinds"][:3] == ["T", "S", "Z"]
    assert bag["remaining"] == 2
    assert bag["bag_position"] == 5
    assert bag["counts"]["T"] == 1
    assert bag["counts"]["S"] == 1
    assert bag["counts"]["Z"] == 0
    assert pending["total_lines"] == 5
    assert pending["min_timer"] == 2
    assert pending["batch_count"] == 2


def test_is_opener_phase_and_resolve_outgoing_attack(opener_cancel_engine):
    result = opener_cancel_engine.resolve_outgoing_attack(3, opener_phase=True)

    assert opener_cancel_engine.is_opener_phase() is True
    assert result["used_opener_multiplier"] is True
    assert result["canceled"] == 5
    assert result["sent"] == 0
    assert opener_cancel_engine.total_attack_canceled == 5


def test_t_and_non_t_spin_detection_paths():
    t_engine = TetrisEngine(spin_mode="all_spin")
    t_piece = make_piece(t_engine, kind="T", rotation=0, last_rotation_dir=1)
    t_engine.board = np.zeros_like(t_engine.board)
    px, py = t_piece.position
    for dx, dy in [(0, 0), (2, 0), (0, 2)]:
        t_engine.board[py + dy, px + dx] = 9

    t_spin = t_engine.detect_spin(t_piece)

    all_spin_engine = TetrisEngine(spin_mode="all_spin")
    l_piece = make_non_t_all_spin_piece(all_spin_engine)
    l_spin = all_spin_engine.detect_spin(l_piece)

    assert t_spin is not None
    assert t_spin["spin_type"] == "t-spin"
    assert l_spin is not None
    assert l_spin["piece"] == "L"
