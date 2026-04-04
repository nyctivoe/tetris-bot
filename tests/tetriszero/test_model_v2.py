from __future__ import annotations

import torch

from model_v2 import TetrisZeroConfig, TetrisZeroNet, load_checkpoint, save_checkpoint


def test_forward_supports_variable_candidates_and_mask():
    model = TetrisZeroNet(TetrisZeroConfig())
    board = torch.randn(2, 12, 20, 10)
    pieces = torch.randint(0, 8, (2, 7))
    context = torch.randn(2, 28)
    candidates = torch.randn(2, 5, 32)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)

    outputs = model(board, pieces, context, candidates, mask)

    assert outputs["policy_logits"].shape == (2, 5)
    assert outputs["value"].shape == (2,)
    assert outputs["attack"].shape == (2,)
    assert outputs["survival"].shape == (2,)
    assert outputs["realized_surge"].shape == (2,)
    assert outputs["policy_logits"][0, 3] < -1.0e8


def test_eval_mode_is_deterministic():
    model = TetrisZeroNet(TetrisZeroConfig())
    model.eval()
    board = torch.randn(1, 12, 20, 10)
    pieces = torch.randint(0, 8, (1, 7))
    context = torch.randn(1, 28)
    candidates = torch.randn(1, 4, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)

    first = model(board, pieces, context, candidates, mask)
    second = model(board, pieces, context, candidates, mask)

    assert torch.allclose(first["policy_logits"], second["policy_logits"])
    assert torch.allclose(first["value"], second["value"])


def test_checkpoint_round_trip_preserves_outputs(tmp_path):
    model = TetrisZeroNet(TetrisZeroConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    model.eval()
    board = torch.randn(1, 12, 20, 10)
    pieces = torch.randint(0, 8, (1, 7))
    context = torch.randn(1, 28)
    candidates = torch.randn(1, 3, 32)
    mask = torch.ones(1, 3, dtype=torch.bool)
    before = model(board, pieces, context, candidates, mask)

    path = tmp_path / "tetriszero.pt"
    save_checkpoint(path, model, optimizer, model.config, {"tag": "roundtrip"})
    loaded = load_checkpoint(path, device="cpu")
    after = loaded["model"](board, pieces, context, candidates, mask)

    assert loaded["metadata"]["tag"] == "roundtrip"
    assert torch.allclose(before["policy_logits"], after["policy_logits"], atol=1e-6)
    assert torch.allclose(before["value"], after["value"], atol=1e-6)
