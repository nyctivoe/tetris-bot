from pathlib import Path

import numpy as np
import torch

from tools.replay_alignment_report import run_alignment_report


def test_replay_alignment_report_on_synthetic_fixture(tmp_path: Path):
    base_board = np.zeros((40, 10), dtype=np.uint8)
    placement = {
        "x": 4,
        "y": 18,
        "r": "N",
        "rotation": 0,
        "kind": "O",
        "last_was_rot": False,
        "last_rot_dir": None,
        "last_kick_idx": None,
    }
    step = {
        "move_number": 1,
        "base_board": base_board,
        "placed": "O",
        "queue_state": {
            "current": "O",
            "hold": None,
            "next_queue": "",
        },
        "pre_state": {
            "combo": 0,
            "combo_active": False,
            "b2b_chain": 0,
            "surge_charge": 0,
            "incoming_garbage_total": 0,
        },
        "expert_replay": {
            "x": 4,
            "y": 18,
            "r": "N",
            "t_spin": "N",
            "attack": 0,
            "cleared": 0,
        },
        "expert_match_index": 0,
        "valid_indices": [0],
        "bfs_boards": np.zeros((1, 40, 10), dtype=np.uint8),
        "bfs_placements": [placement],
        "future_pieces": [],
    }
    torch.save(
        {
            "cache_version": 3,
            "game_id": 1,
            "steps": [step],
        },
        tmp_path / "1.pt",
    )

    report = run_alignment_report(str(tmp_path), sample_size=10, seed=0)

    assert report["samples_evaluated"] == 1
    assert report["clear_mismatch_count"] == 0
    assert report["non_t_spin_positives"] == 0
    assert report["attack_exact_match_rate"] == 1.0
    assert report["spin_confusion_counts"] == {}
