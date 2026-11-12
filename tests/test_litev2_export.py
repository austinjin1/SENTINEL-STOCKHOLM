# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Multi-input export of the ANCHOR-Lite family (amendment §4.6) — random weights and random tensors only."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sentinel.models.lite import export as X
from sentinel.models.litev2 import export as EX
from sentinel.models.litev2 import nets as NN

torch.set_num_threads(4)
ARMS = {"anchor_lite": {}, "A1": dict(use_anchor=False), "A3": dict(use_context=False), "A5": dict(attention=False)}


def _model(arm: str = "anchor_lite", *, moved: bool = True, seed: int = 0) -> NN.AnchorLite:
    torch.manual_seed(seed)
    m = NN.AnchorLite(**ARMS[arm]).eval()
    if moved:                                # move every zero-initialised piece off zero: each path is exercised
        with torch.no_grad():
            for p in m.new_parameters():
                if p.abs().sum() == 0:
                    p.add_(0.05 * torch.randn_like(p) if p.ndim else 0.3)
            for h in m.res.heads:
                h[-1].weight[0].normal_(0, 0.3)
    return m


def _data(m, n: int, size: int, seed: int = 0, nan: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    chips = rng.integers(0, 256, (n, 4, size, size), dtype=np.uint8)
    chips[:, 3] = (chips[:, 3] * rng.uniform(0.2, 1.2, (n, 1, 1))).clip(0, 255).astype(np.uint8)
    d = EX.tile_inputs(chips, rng.uniform(0.3, 1.0, n))
    d["sc"] = rng.normal(size=(n, NN.N_SC)).astype(np.float32)
    if nan:
        d["sc"][rng.random(d["sc"].shape) < 0.05] = np.nan
    d["anchor"] = rng.normal(0.5, 0.6, (n, len(m.targets))).astype(np.float32)
    d["water"][0] = 0.0                                          # a chip without a water cell: every cell is used
    return d


def _wrapper(m, d: dict) -> tuple[np.ndarray, np.ndarray]:
    w = EX.ExportAnchorLite(m).eval()
    with torch.no_grad():
        o = w(*[torch.as_tensor(d[k]) for k in EX.input_names(m)]).numpy()
    T = o.shape[1] // 2
    return o[:, :T], np.exp(o[:, T:])


def test_input_names_follow_the_switches():
    assert EX.input_names(_model("anchor_lite", moved=False)) == ["x01", "sc", "anchor", "pool_w", "water", "gsd"]
    assert EX.input_names(_model("A1", moved=False)) == ["x01", "sc", "pool_w", "water", "gsd"]
    assert EX.input_names(_model("A3", moved=False)) == ["x01", "anchor", "pool_w", "water", "gsd"]
    assert EX.input_names(_model("A5", moved=False)) == ["x01", "sc", "anchor", "pool_w"]
    with pytest.raises(EX.ExportError):
        EX.ExportAnchorLite(NN.WaterSet())


@pytest.mark.parametrize("arm", list(ARMS))
def test_wrapper_equals_the_training_forward(arm):
    m = _model(arm)
    for size in (128, 256):
        d = _data(m, 5, size, seed=size)
        mu_t, sd_t = EX.torch_predict(m, d)
        mu_w, sd_w = _wrapper(m, d)
        assert np.isfinite(mu_w).all() and mu_w.shape == (5, len(m.targets))
        assert np.abs(mu_w - mu_t).max() < 1e-5 and np.abs(sd_w / sd_t - 1).max() < 1e-5


def test_aux_head_is_not_exported_and_counts():
    m = _model()
    c = EX.count_params(m)
    assert c["network"] == NN.count_parameters(m) and c["exported"] == c["network"] - c["aux_kd_head"] > 1_300_000
    assert c["aux_kd_head"] == sum(p.numel() for p in m.student.head.parameters()) > 0
    g = EX.count_macs(m, 512) / 1e9
    assert 0.9 < g < 1.3                                         # plan estimate 1.13 GMACs per 512² tile


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    m = _model()
    d = tmp_path_factory.mktemp("export")
    return m, EX.export_onnx(m, d / "fp32.onnx", size=256), d


def test_onnx_fp32_equals_torch_at_two_sizes(exported):
    m, fp32, _ = exported
    for size in (128, 256):
        d = _data(m, 4, size, seed=size + 1)
        mu_t, sd_t = EX.torch_predict(m, d)
        mu_o, sd_o = EX.onnx_predict(fp32, d, threads=2)
        assert np.abs(mu_o - mu_t).max() < 1e-4 and np.abs(sd_o / sd_t - 1).max() < 1e-4


def test_image_scope_is_the_convolutional_branch(exported):
    import onnx

    _, fp32, _ = exported
    g = onnx.load(str(fp32)).graph
    img = set(EX.image_nodes(fp32))
    conv = [n.name for n in g.node if n.op_type == "Conv"]
    assert [c for c in conv if c not in img] == [c for c in conv if "att_conv" in c] and len(conv) - 1 > 20
    assert not any(n.op_type == "Gemm" and n.name in img for n in g.node)      # MLPs stay outside


def test_int8_image_scope_runs_the_frozen_recipe_and_keeps_the_anchor_exact(tmp_path):
    """Zero-initialised model: μ equals the anchor (§4.3). With the image-branch scope that stays exact after INT8,
    because the anchor path is float32; the whole graph is processed by the frozen ``quant_pre_process`` call."""
    m = _model(moved=False)
    fp32 = EX.export_onnx(m, tmp_path / "fp32.onnx", size=256)
    rng = np.random.default_rng(0)
    tiles = X.calibration_tiles(rng.integers(0, 256, (16, 4, 128, 128), dtype=np.uint8), n=4, size=256)
    feeds = EX.calibration_feeds(m, tiles, rng.normal(size=(4, NN.N_SC)), rng.normal(size=(4, 6)), np.full(4, 0.6))
    assert len(feeds) == 4 and list(feeds[0]) == EX.input_names(m) and feeds[0]["x01"].shape == (1, 4, 256, 256)
    int8 = EX.quantise_int8(fp32, tmp_path / "int8.onnx", feeds, scope="image")
    assert int8.stat().st_size < 0.5 * fp32.stat().st_size
    d = _data(m, 4, 256, seed=3)
    mu8, sd8 = EX.onnx_predict(int8, d, threads=2)
    assert np.abs(mu8 - d["anchor"]).max() < 1e-6 and np.isfinite(sd8).all()
    with pytest.raises(EX.ExportError):
        EX.quantise_int8(fp32, tmp_path / "x.onnx", feeds, scope="tail")
