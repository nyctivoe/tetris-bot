from tetrisformer.model import _simulate_current_placement
from tetrisEngine import TetrisEngine


def test_preserves_lock_out_when_rollout_horizon_ends():
    engine = TetrisEngine()
    piece = engine.spawn_piece("I")
    placement = {
        "x": piece.position[0],
        "y": piece.position[1],
        "rotation": piece.rotation,
        "last_was_rot": False,
        "last_rot_dir": None,
        "last_kick_idx": None,
    }

    simulated, _reward, _attack = _simulate_current_placement(
        engine,
        placement,
        next_piece_kind=None,
        attack_w=1.0,
        tspin_bonus=1.5,
        b2b_bonus=0.5,
        height_penalty=0.1,
        holes_penalty=0.5,
        topout_penalty=10.0,
        tslot_ready_bonus=0.0,
    )

    assert simulated.game_over is True
    assert simulated.game_over_reason == "lock_out"
