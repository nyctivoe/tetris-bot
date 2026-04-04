from __future__ import annotations

import numpy as np
import torch
import pytest

from beam_search import BeamSearchConfig, beam_search_select, generate_candidates
from mcts import MctsConfig, run_mcts, select_action_from_visits
from model_v2 import TetrisZeroConfig, TetrisZeroNet

from .fixtures import make_manual_candidate


class DummyModel:
    def __call__(self, board_tensor, piece_ids, context_scalars, candidate_features, candidate_mask):
        batch, candidates, _ = candidate_features.shape
        logits = torch.zeros((batch, candidates), dtype=torch.float32)
        logits[:, 1] = 5.0
        return {
            "policy_logits": logits,
            "value": torch.ones((batch,), dtype=torch.float32) * 0.2,
            "attack": torch.zeros((batch,), dtype=torch.float32),
            "survival": torch.ones((batch,), dtype=torch.float32) * 0.5,
            "realized_surge": torch.zeros((batch,), dtype=torch.float32),
        }


def test_mcts_disabled_matches_beam_choice(seeded_engine):
    beam = beam_search_select(seeded_engine, depth=1, width=8, cfg=BeamSearchConfig(depth=1, width=8))
    mcts = run_mcts(seeded_engine, None, 1, None, None, cfg=MctsConfig(enabled=False, beam_cfg=BeamSearchConfig(depth=1, width=8)))

    assert mcts["selected_candidate"]["candidate_index"] == beam["candidate_index"]


def test_enabled_mcts_can_follow_model_priors(monkeypatch, seeded_engine):
    first = make_manual_candidate(seeded_engine, candidate_index=0, stats={"attack": 0, "surge_send": 0, "surge_charge": 0, "lines_cleared": 0, "is_difficult": False, "qualifies_b2b": False, "breaks_b2b": False})
    second = make_manual_candidate(seeded_engine, candidate_index=1, stats={"attack": 0, "surge_send": 0, "surge_charge": 0, "lines_cleared": 0, "is_difficult": False, "qualifies_b2b": False, "breaks_b2b": False})

    monkeypatch.setattr("mcts.generate_candidates", lambda *_args, **_kwargs: [first, second])
    result = run_mcts(
        seeded_engine,
        None,
        1,
        DummyModel(),
        [first, second],
        cfg=MctsConfig(enabled=True, simulations=16, temperature=0.0),
    )

    assert result["selected_candidate"]["candidate_index"] == 1
    assert int(np.argmax(result["visits"])) == 1


def test_enabled_mcts_uses_future_state_value(monkeypatch, seeded_engine):
    first = make_manual_candidate(
        seeded_engine,
        candidate_index=0,
        stats={"attack": 0, "surge_send": 0, "surge_charge": 0, "lines_cleared": 0, "is_difficult": False, "qualifies_b2b": False, "breaks_b2b": False},
    )
    second = make_manual_candidate(
        seeded_engine,
        candidate_index=1,
        stats={"attack": 0, "surge_send": 0, "surge_charge": 0, "lines_cleared": 0, "is_difficult": False, "qualifies_b2b": False, "breaks_b2b": False},
    )
    first["engine_after"].total_lines_cleared = 1
    second["engine_after"].total_lines_cleared = 2

    def fake_generate(engine, include_hold=True):
        del include_hold
        return [] if int(engine.total_lines_cleared) in {1, 2} else [first, second]

    def fake_score(engine, opponent_engine=None, move_number=0, horizon_state=None, cfg=None):
        del opponent_engine, move_number, cfg
        if horizon_state is not None:
            return 0.0
        return 6.0 if int(engine.total_lines_cleared) == 1 else -6.0

    monkeypatch.setattr("mcts.generate_candidates", fake_generate)
    monkeypatch.setattr("mcts.score_position", fake_score)
    result = run_mcts(
        seeded_engine,
        None,
        1,
        None,
        [first, second],
        cfg=MctsConfig(enabled=True, simulations=24, temperature=0.0, max_depth=2),
    )

    assert result["selected_candidate"]["candidate_index"] == 0
    assert result["q_values"][0] > result["q_values"][1]


def test_mcts_samples_hidden_piece_once_visible_queue_is_exhausted(monkeypatch, seeded_engine):
    candidate = generate_candidates(seeded_engine, include_hold=False)[0]
    observed: dict[str, object] = {}

    class RecordingRng:
        def choice(self, values, p=None):
            observed["values"] = list(np.asarray(values).tolist())
            observed["probs"] = None if p is None else list(np.asarray(p).tolist())
            return int(np.asarray(values)[0])

    monkeypatch.setattr("mcts.np.random.default_rng", lambda seed=None: RecordingRng())
    run_mcts(
        seeded_engine,
        None,
        1,
        None,
        [candidate],
        cfg=MctsConfig(enabled=True, simulations=2, temperature=0.0, max_depth=2, visible_queue_depth=0),
    )

    assert "values" in observed
    assert observed["values"]


def test_temperature_sampling_uses_non_fixed_rng_seed(monkeypatch):
    seen: dict[str, object] = {}

    class StubRng:
        def choice(self, values, p=None):
            del p
            return int(values[0] if not isinstance(values, int) else 0)

    def fake_default_rng(seed=None):
        seen["seed"] = seed
        return StubRng()

    monkeypatch.setattr("mcts.np.random.default_rng", fake_default_rng)
    select_action_from_visits(np.asarray([5, 4, 3], dtype=np.int32), temperature=1.0)

    assert seen["seed"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this regression test.")
def test_mcts_supports_cuda_models(seeded_engine):
    model = TetrisZeroNet(TetrisZeroConfig()).to("cuda").eval()

    result = run_mcts(
        seeded_engine,
        None,
        1,
        model,
        None,
        cfg=MctsConfig(enabled=True, simulations=2, temperature=0.0, max_depth=1),
    )

    assert result["selected_candidate"]["candidate_index"] >= 0
