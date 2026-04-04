import numpy as np

from tetrisEngine import TetrisEngine


def _make_piece(
    engine: TetrisEngine,
    *,
    kind: str = "T",
    rotation: int = 0,
    position=(4, 4),
    last_rotation_dir=1,
    last_kick_index=None,
):
    piece = engine.spawn_piece(kind, position=position, rotation=rotation)
    piece.last_action_was_rotation = True
    piece.last_rotation_dir = last_rotation_dir
    piece.last_kick_index = last_kick_index
    return piece


def _occupy_corners(engine: TetrisEngine, piece, corners):
    engine.board = np.zeros_like(engine.board)
    px, py = piece.position
    for dx, dy in corners:
        engine.board[py + dy, px + dx] = 9


def test_t_spin_full_by_front_corners():
    engine = TetrisEngine()
    piece = _make_piece(engine, rotation=0, last_rotation_dir=1)
    _occupy_corners(engine, piece, [(0, 0), (2, 0), (0, 2)])

    spin = engine.detect_spin(piece)

    assert spin is not None
    assert spin["spin_type"] == "t-spin"
    assert spin["is_mini"] is False
    assert spin["front_corners"] == 2


def test_t_spin_mini_by_single_front_corner():
    engine = TetrisEngine()
    piece = _make_piece(engine, rotation=0, last_rotation_dir=1)
    _occupy_corners(engine, piece, [(0, 0), (0, 2), (2, 2)])

    spin = engine.detect_spin(piece)

    assert spin is not None
    assert spin["spin_type"] == "t-spin"
    assert spin["is_mini"] is True
    assert spin["front_corners"] == 1


def test_t_spin_full_on_180_rotation():
    engine = TetrisEngine()
    piece = _make_piece(engine, rotation=0, last_rotation_dir=2)
    _occupy_corners(engine, piece, [(0, 0), (0, 2), (2, 2)])

    spin = engine.detect_spin(piece)

    assert spin is not None
    assert spin["is_180"] is True
    assert spin["is_mini"] is False


def test_t_spin_full_on_fin_kick():
    engine = TetrisEngine()
    piece = _make_piece(engine, rotation=0, last_rotation_dir=1, last_kick_index=4)
    _occupy_corners(engine, piece, [(0, 0), (0, 2), (2, 2)])

    spin = engine.detect_spin(piece)

    assert spin is not None
    assert spin["kick_index"] == 4
    assert spin["is_mini"] is False


def test_non_t_piece_does_not_spin_in_default_mode():
    engine = TetrisEngine()
    piece = _make_piece(engine, kind="L", rotation=0, last_rotation_dir=1)

    assert engine.detect_spin(piece) is None
