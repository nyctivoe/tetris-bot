from __future__ import annotations

import numpy as np

from beam_search import BeamSearchConfig, _transposition_key, beam_search_select, generate_candidates, score_position
from tetrisEngine import KIND_TO_PIECE_ID

from .fixtures import make_manual_candidate


def test_beam_search_returns_legal_candidate_end_to_end(seeded_engine):
    selected = beam_search_select(seeded_engine, depth=1, width=8, cfg=BeamSearchConfig(depth=1, width=8))
    probe = seeded_engine.clone()

    assert probe.apply_placement(selected["placement"]) is True


def test_generate_candidates_includes_hold_branch(seeded_engine):
    seeded_engine.hold = KIND_TO_PIECE_ID["I"]

    candidates = generate_candidates(seeded_engine, include_hold=True)

    assert candidates
    assert any(candidate["used_hold"] for candidate in candidates)


def test_beam_search_prefers_preserve_over_break(monkeypatch, preserve_break_fixture):
    engine, preserve, break_move = preserve_break_fixture
    break_move = dict(break_move)
    break_move["stats"] = dict(break_move["stats"])
    break_move["stats"]["attack"] = 1
    break_move["stats"]["surge_send"] = 0
    break_move["stats"]["surge_charge"] = 0

    monkeypatch.setattr("beam_search.generate_candidates", lambda *_args, **_kwargs: [preserve, break_move])
    selected = beam_search_select(engine, depth=1, width=2, cfg=BeamSearchConfig(depth=1, width=2))

    assert selected["candidate_index"] == preserve["candidate_index"]


def test_beam_search_prefers_intentional_break_when_surge_cashout_is_large(monkeypatch, preserve_break_fixture):
    engine, preserve, break_move = preserve_break_fixture
    engine.incoming_garbage = [{"lines": 6, "timer": 1, "col": 0}]
    break_move = dict(break_move)
    break_move["stats"] = dict(break_move["stats"])
    break_move["stats"]["attack"] = 10
    break_move["stats"]["surge_send"] = 8

    monkeypatch.setattr("beam_search.generate_candidates", lambda *_args, **_kwargs: [preserve, break_move])
    selected = beam_search_select(engine, depth=1, width=2, cfg=BeamSearchConfig(depth=1, width=2))

    assert selected["candidate_index"] == break_move["candidate_index"]


def test_beam_search_tie_breaks_to_lowest_index(monkeypatch, seeded_engine):
    first = make_manual_candidate(seeded_engine, candidate_index=0, stats={"attack": 0, "surge_send": 0, "surge_charge": 0, "lines_cleared": 0, "is_difficult": False, "qualifies_b2b": False, "breaks_b2b": False})
    second = make_manual_candidate(seeded_engine, candidate_index=1, stats={"attack": 0, "surge_send": 0, "surge_charge": 0, "lines_cleared": 0, "is_difficult": False, "qualifies_b2b": False, "breaks_b2b": False})

    monkeypatch.setattr("beam_search.generate_candidates", lambda *_args, **_kwargs: [first, second])
    selected = beam_search_select(seeded_engine, depth=1, width=2, cfg=BeamSearchConfig(depth=1, width=2))

    assert selected["candidate_index"] == 0


def test_score_position_uses_post_move_engine_state(monkeypatch, seeded_engine):
    seeded_engine.hold = KIND_TO_PIECE_ID["I"]
    seeded_engine.bag = np.asarray(
        [KIND_TO_PIECE_ID[k] for k in ("O", "I", "S", "Z", "J", "L", "T", "O", "I", "S", "Z", "J", "L", "T", "O", "I")],
        dtype=np.int64,
    )
    seeded_engine.bag_size = len(seeded_engine.bag)
    good = make_manual_candidate(
        seeded_engine,
        candidate_index=0,
        stats={"attack": 0, "surge_send": 0, "surge_charge": 0, "lines_cleared": 0, "is_difficult": False, "qualifies_b2b": False, "breaks_b2b": False},
    )
    bad = make_manual_candidate(
        seeded_engine,
        candidate_index=1,
        stats={"attack": 0, "surge_send": 0, "surge_charge": 0, "lines_cleared": 0, "is_difficult": False, "qualifies_b2b": False, "breaks_b2b": False},
    )
    good["engine_after"].hold = KIND_TO_PIECE_ID["T"]
    bad["engine_after"].hold = KIND_TO_PIECE_ID["I"]
    for candidate in (good, bad):
        candidate["engine_after"].bag = seeded_engine.bag.copy()
        candidate["engine_after"].bag_size = len(candidate["engine_after"].bag)

    monkeypatch.setattr(
        "beam_search.summarize_board",
        lambda _board: {
            "max_height": 0.0,
            "holes": 0.0,
            "covered_hole_burden": 0.0,
            "row_transitions": 0.0,
            "column_transitions": 0.0,
            "t_slot_count": 1.0,
            "all_spin_slot_count": 0.0,
            "overflow": 0.0,
        },
    )

    assert score_position(seeded_engine, horizon_state=good) > score_position(seeded_engine, horizon_state=bad)


def test_score_position_penalizes_break_using_pre_move_b2b_chain(seeded_engine):
    candidate = make_manual_candidate(
        seeded_engine,
        candidate_index=0,
        stats={"attack": 0, "surge_send": 0, "surge_charge": 0, "lines_cleared": 1, "is_difficult": False, "qualifies_b2b": False, "breaks_b2b": True},
    )
    with_chain = seeded_engine.clone()
    with_chain.b2b_chain = 5
    without_chain = seeded_engine.clone()
    without_chain.b2b_chain = 0

    assert score_position(with_chain, horizon_state=candidate) < score_position(without_chain, horizon_state=candidate)


def test_transposition_key_distinguishes_combo_activity(seeded_engine):
    inactive = seeded_engine.clone()
    active = seeded_engine.clone()
    inactive.combo = 0
    inactive.combo_active = False
    active.combo = 0
    active.combo_active = True

    assert _transposition_key(inactive, 2) != _transposition_key(active, 2)
