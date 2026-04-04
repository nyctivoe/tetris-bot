import numpy as np

from tetrisformer.model import _set_engine_root_state
from tetrisEngine import TetrisEngine


def test_root_state_restore_uses_cached_pre_state_not_row_fields():
    base_board = np.zeros((40, 10), dtype=np.uint8)
    base_board[39, 0] = 8
    step_info = {
        "base_board": base_board,
        "placed": "I",
        "queue_state": {
            "current": "T",
            "hold": "L",
            "next_queue": "SZOIJ",
        },
        "pre_state": {
            "combo": 3,
            "combo_active": True,
            "b2b_chain": 4,
            "surge_charge": 2,
            "incoming_garbage_total": 5,
        },
        "row": {
            "combo": 99,
            "hold": "O",
        },
    }

    engine = TetrisEngine()
    _set_engine_root_state(engine, step_info)

    assert np.array_equal(engine.board, base_board)
    assert engine.combo == 3
    assert engine.combo_active is True
    assert engine.b2b_chain == 4
    assert engine.surge_charge == 2
    assert engine.hold == "L"
    assert engine.current_piece.kind == "T"
    assert engine.bag_size == 0
    assert len(engine.incoming_garbage) == 1
    assert engine.incoming_garbage[0]["lines"] == 5
