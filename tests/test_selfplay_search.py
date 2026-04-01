import numpy as np

from rl.model_loader import load_tetrisformer_checkpoint
from rl.search import (
    generate_action_candidates,
    score_bfs_candidates,
    select_top_k_candidates,
    soft_policy_from_scores,
)
from rl.selfplay import create_selfplay_engine, run_single_selfplay_game


def test_score_bfs_candidates_returns_one_score_per_bfs_result(checkpoint_factory):
    ckpt_path = checkpoint_factory(spin_mode="all_spin")
    bundle = load_tetrisformer_checkpoint(str(ckpt_path), device="cpu")
    engine = create_selfplay_engine("all_spin", seed=0)
    bfs_results = engine.bfs_all_placements(include_no_place=False, dedupe_final=True)

    scores = score_bfs_candidates(bundle, engine, bfs_results, move_number=1)

    assert scores.shape == (len(bfs_results),)
    assert np.all(np.isfinite(scores))


def test_select_top_k_candidates_preserves_original_bfs_indices():
    bfs_results = [{"board": i} for i in range(5)]
    scores = np.array([0.1, 0.9, -0.5, 0.8, 0.3], dtype=np.float32)

    indices, retained_scores = select_top_k_candidates(bfs_results, scores, top_k=3)

    assert indices == [1, 3, 4]
    assert np.allclose(retained_scores, np.array([0.9, 0.8, 0.3], dtype=np.float32))


def test_soft_policy_from_scores_is_normalized_and_stable():
    scores = np.array([1000.0, 1001.0, 999.0], dtype=np.float32)

    policy = soft_policy_from_scores(scores, temperature=1.0)
    greedy = soft_policy_from_scores(scores, temperature=0.0)

    assert np.isclose(float(policy.sum()), 1.0)
    assert np.all(policy > 0.0)
    assert greedy.tolist() == [0.0, 1.0, 0.0]


def test_run_single_selfplay_game_emits_valid_step(checkpoint_factory):
    ckpt_path = checkpoint_factory(spin_mode="all_spin")
    bundle = load_tetrisformer_checkpoint(str(ckpt_path), device="cpu")

    game = run_single_selfplay_game(
        model_bundle=bundle,
        game_id=7,
        max_plies=2,
        beam_width=2,
        top_k=2,
        temperature=1.0,
        spin_mode="all_spin",
        seed=123,
    )

    assert game["game_id"] == 7
    assert game["steps"]
    first = game["steps"][0]
    assert first["root_stats"]["spin_mode"] == "all_spin"
    assert first["base_board"].shape == (40, 10)
    assert first["bfs_boards"].shape[0] == len(first["candidate_indices"])
    assert first["bfs_boards"].shape[1:] == (40, 10)
    assert len(first["bfs_placements"]) == len(first["candidate_indices"])
    assert np.isclose(float(first["policy_target"].sum()), 1.0)
    assert 0 <= int(first["chosen_index"]) < len(first["candidate_indices"])
    assert isinstance(first["value_target"], float)
    assert isinstance(first["n_step_value_target"], float)
    assert all("used_hold" in placement for placement in first["bfs_placements"])


def test_generate_action_candidates_includes_hold_branch():
    engine = create_selfplay_engine("all_spin", seed=0)
    engine.hold = 1  # I

    candidates = generate_action_candidates(engine, include_hold=True)

    assert candidates
    assert any(candidate["placement"].get("used_hold") for candidate in candidates)
