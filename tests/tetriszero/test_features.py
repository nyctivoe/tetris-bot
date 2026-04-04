from __future__ import annotations

import numpy as np

from features import build_model_inputs, encode_board, encode_context, extract_candidate_features


def test_encode_board_returns_expected_shape(seeded_engine):
    board_tensor = encode_board(seeded_engine)

    assert board_tensor.shape == (12, 20, 10)
    assert board_tensor.dtype == np.float32


def test_encode_context_fills_opponent_features(seeded_engine):
    opponent = seeded_engine.clone()
    opponent.total_attack_sent = 8
    opponent.board[-1, 0] = 9
    no_opp = encode_context(seeded_engine, None, move_number=3)
    with_opp = encode_context(seeded_engine, opponent, move_number=3)

    assert no_opp.shape == (28,)
    assert with_opp.shape == (28,)
    assert np.allclose(no_opp[24:28], 0.0)
    assert np.any(with_opp[24:28] > 0.0)


def test_candidate_features_distinguish_preserve_and_break(preserve_break_fixture):
    engine, preserve, break_move = preserve_break_fixture

    preserve_features = extract_candidate_features(engine, preserve, None, move_number=5)
    break_features = extract_candidate_features(engine, break_move, None, move_number=5)

    assert preserve_features.shape == (32,)
    assert break_features.shape == (32,)
    assert preserve_features[18] == 1.0
    assert break_features[19] == 1.0
    assert break_features[20] > preserve_features[20]
    assert break_features[23] > preserve_features[23]


def test_build_model_inputs_is_deterministic_and_non_mutating(preserve_break_fixture):
    engine, preserve, break_move = preserve_break_fixture
    before_board = engine.board.copy()
    before_piece = engine.current_piece.copy()

    payload = build_model_inputs(engine, [preserve, break_move], opponent_engine=engine.clone(), move_number=7)

    assert payload["board_tensor"].shape == (12, 20, 10)
    assert payload["piece_ids"].shape == (7,)
    assert payload["context_scalars"].shape == (28,)
    assert payload["candidate_features"].shape == (2, 32)
    assert payload["candidate_mask"].shape == (2,)
    assert np.array_equal(engine.board, before_board)
    assert engine.current_piece.position == before_piece.position
