from pathlib import Path

import numpy as np
import pytest
import torch

import preparse_games
from model import SmartRolloutRankDataset


def test_cache_v3_contains_pre_state_and_queue_state(monkeypatch, tmp_path: Path):
    row = {
        "playfield": "base",
        "placed": "O",
        "hold": "T",
        "next": "IJLSZ",
        "incoming_garbage": 4,
        "immediate_garbage": 0,
        "x": 4,
        "y": 18,
        "r": "N",
        "t_spin": "N",
        "attack": 0,
        "cleared": 0,
    }
    next_row = {"playfield": "next"}

    monkeypatch.setattr(preparse_games, "load_index", lambda index_path=None: [{"game_id": 1}])
    monkeypatch.setattr(
        preparse_games,
        "_collect_move_rows",
        lambda entry, data_path: [
            {"row": row, "next_row": next_row, "placed": "O", "move_number": 1}
        ],
    )
    monkeypatch.setattr(
        preparse_games,
        "playfield_to_board",
        lambda playfield: np.zeros((40, 10), dtype=np.uint8),
    )
    monkeypatch.setattr(
        preparse_games,
        "find_bfs_match_index",
        lambda bfs_results, target_board, max_garbage=None, shift=True: (0, 0),
    )

    def fake_bfs(self, include_no_place=False):
        piece = self.current_piece
        return [
            {
                "board": self.board.copy(),
                "placement": {
                    "x": piece.position[0],
                    "y": piece.position[1],
                    "r": "N",
                    "rotation": piece.rotation,
                    "kind": piece.kind,
                    "last_was_rot": False,
                    "last_rot_dir": None,
                    "last_kick_idx": None,
                },
            }
        ]

    monkeypatch.setattr(preparse_games.TetrisEngine, "bfs_all_placements", fake_bfs)

    preparse_games.build_dataset(
        data_path="unused.csv",
        index_path="unused.csv",
        cache_dir=str(tmp_path),
        overwrite=True,
        limit_games=0,
    )

    cache = torch.load(tmp_path / "1.pt", weights_only=False)
    assert cache["cache_version"] == 3
    step = cache["steps"][0]
    assert step["move_number"] == 1
    assert step["queue_state"] == {
        "current": "O",
        "hold": "T",
        "next_queue": "IJLSZ",
    }
    assert step["pre_state"]["incoming_garbage_total"] == 4
    assert step["expert_replay"]["t_spin"] == "N"
    assert step["expert_match_index"] == 0
    assert isinstance(step["valid_indices"], list)
    assert step["bfs_boards"].shape == (1, 40, 10)


def test_dataset_rejects_non_v3_cache(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    torch.save({"game_id": 1, "steps": []}, cache_dir / "1.pt")

    dataset = SmartRolloutRankDataset(
        entries=[{"game_id": 1}],
        data_path="unused.csv",
        cache_dir=str(cache_dir),
        k_candidates=1,
        max_samples=1,
    )

    with pytest.raises(RuntimeError, match="Unsupported cache version"):
        next(iter(dataset))
