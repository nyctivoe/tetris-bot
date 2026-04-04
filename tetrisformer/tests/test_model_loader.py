from pathlib import Path

import torch

import tetrisformer.inferenceUI as inferenceUI
from tetrisformer.model import TetrisFormerV4
from tetrisformer.rl.model_loader import load_tetrisformer_checkpoint


def test_load_tetrisformer_checkpoint_loads_current_checkpoint(checkpoint_factory):
    ckpt_path = checkpoint_factory(rank_q_alpha=0.42)

    bundle = load_tetrisformer_checkpoint(str(ckpt_path), device="cpu", eval_mode=True)

    assert isinstance(bundle.model, TetrisFormerV4)
    assert bundle.model._embed_dim == 192
    assert bundle.model.transformer.layers[0].self_attn.num_heads == 6
    assert len(bundle.model.transformer.layers) == 4
    assert bundle.rank_q_alpha == 0.42
    assert bundle.checkpoint_arch == "tetrisformer_v4r1"
    assert bundle.cache_version == 3


def test_load_tetrisformer_checkpoint_rejects_raw_state_dict(tmp_path: Path):
    path = tmp_path / "raw_state.pt"
    torch.save({"cls_token": torch.randn(1, 1, 192)}, path)

    try:
        load_tetrisformer_checkpoint(str(path), device="cpu")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "missing model_state_dict" in str(exc)


def test_load_tetrisformer_checkpoint_rejects_inconsistent_num_heads(checkpoint_factory):
    ckpt_path = checkpoint_factory()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["config"]["num_heads"] = 5
    torch.save(ckpt, ckpt_path)

    try:
        load_tetrisformer_checkpoint(str(ckpt_path), device="cpu")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "not divisible" in str(exc)


def test_inference_ui_load_model_uses_current_loader(checkpoint_factory):
    ckpt_path = checkpoint_factory(spin_mode="all_spin")
    ui = inferenceUI.TetrisInferenceUI.__new__(inferenceUI.TetrisInferenceUI)
    ui.device = torch.device("cpu")
    ui.model = None
    ui.model_path = None

    ui._load_model(str(ckpt_path))

    assert isinstance(ui.model, TetrisFormerV4)
    assert ui.model_path == str(ckpt_path)
