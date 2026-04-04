from __future__ import annotations

from inference_ui import build_search_player, resolve_model_paths
from model_v2 import save_checkpoint


def test_resolve_model_paths_fills_missing_side():
    both = resolve_model_paths("shared.pt", None, None)
    left_only = resolve_model_paths(None, "left.pt", None)
    right_only = resolve_model_paths(None, None, "right.pt")

    assert both == ("shared.pt", "shared.pt")
    assert left_only == ("left.pt", "left.pt")
    assert right_only == ("right.pt", "right.pt")


def test_search_player_selects_legal_action_with_heuristic_mcts(seeded_engine):
    player = build_search_player(
        "Player A",
        None,
        device="cpu",
        simulations=2,
        temperature=0.0,
        max_depth=1,
        visible_queue_depth=5,
        c_puct=1.5,
        include_hold=True,
        beam_depth=1,
        beam_width=8,
    )
    action = player.select_action(seeded_engine.clone(), seeded_engine.clone(), move_number=1)
    probe = seeded_engine.clone()

    assert probe.apply_placement(action) is True


def test_search_player_loads_tetriszero_checkpoint(tmp_path, tiny_model):
    checkpoint_path = tmp_path / "tiny.pt"
    save_checkpoint(checkpoint_path, tiny_model, None, tiny_model.config, {"tag": "ui"})

    player = build_search_player(
        "Player A",
        str(checkpoint_path),
        device="cpu",
        simulations=2,
        temperature=0.0,
        max_depth=1,
        visible_queue_depth=5,
        c_puct=1.5,
        include_hold=True,
        beam_depth=1,
        beam_width=8,
    )

    assert player.model is not None
    assert player.checkpoint_name == "tiny.pt"
