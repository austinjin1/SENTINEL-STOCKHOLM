# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Multi-input ONNX / INT8 export of the ANCHOR-Lite family (amendment §4.6: "A multi-input export wrapper (chip, s,
c, a) lives in a new file outside every frozen hash. Export recipe and the INT8-vs-FP32 report are as in the frozen
protocol"; frozen ``lite_protocol.md`` §9: ONNX opset 17 → ``quant_pre_process`` → static QDQ INT8, per-channel
weights, calibrated on train chips; desktop-proxy latency).

One ONNX file per seed. Inputs (those the arm uses, in this order):

    x01     (N, 4, H, W)   DN / 255
    sc      (N, 56)        raw [s, c]; a missing value may be NaN (it becomes the train mean, as in training)
    anchor  (N, T)         log10 anchors from the anchor file
    pool_w  (N, H/4, W/4)  frozen centre-window pooling weights (``lite.features.pooling_weights``)
    water   (N, H/4, W/4)  water fraction per output cell (frozen ``features.water_mask``)
    gsd     (N,)           metres per pixel

Output ``out`` (N, 2T): log10 μ, then log σ (natural log of σ in log10 units, as the frozen wrapper).
The auxiliary KD map head is not exported (§4.3: "not used at inference").

Readings decided here (for the lead; written before any trained v2 model existed):
* the attention weights are computed as exp(logit − max over the chip's water cells) · mask / sum, which equals the
  masked softmax of the training code but has no −inf in the graph;
* INT8 scope: the default (``scope="image"``) quantises the image branch — input normalisation, encoder and
  decoder, where all but about 0.1 M of the parameters and nearly all the operations are — with the frozen recipe
  and the frozen calibration tiles, and keeps the context MLPs, pooling and residual heads in float32 inside the
  same file. ``scope="graph"`` applies the same call to every node. Which one is the carrier's recipe is the
  lead's ruling; it is fixed before any INT8 number on a trained v2 model exists.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..lite import features as F
from . import nets as NN

OPSET = 17
IMAGE_SCOPE = "/image/"                                           # ONNX node-name prefix of the image branch
SCOPES = ("image", "graph")


class ExportError(RuntimeError):
    pass


def input_names(model: NN.AnchorLite) -> list[str]:
    names = ["x01"]
    if model.use_context:
        names.append("sc")
    if model.use_anchor:
        names.append("anchor")
    names.append("pool_w")
    if model.attention:
        names += ["water", "gsd"]
    return names


class _Image(nn.Module):
    """Input normalisation + the leader's encoder and decoder (stride-4, 48-channel map)."""

    def __init__(self, model: NN.AnchorLite):
        super().__init__()
        self.m = model

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        st = self.m.student
        return self.m.decoder((x01 - st.x_mean[None, :, None, None]) / st.x_std[None, :, None, None])


class _Tail(nn.Module):
    """FiLM, the two poolings and the residual heads — the code path of ``AnchorLite.forward`` after the decoder."""

    def __init__(self, model: NN.AnchorLite):
        super().__init__()
        self.m = model

    def attention(self, top: torch.Tensor, water: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, _, h, w = top.shape
        cell_m = gsd.float().view(-1, 1, 1) * NN.STRIDE
        yy = (torch.arange(h, dtype=torch.float32) - (h - 1) / 2.0).view(1, h, 1)
        xx = (torch.arange(w, dtype=torch.float32) - (w - 1) / 2.0).view(1, 1, w)
        r = torch.sqrt(yy ** 2 + xx ** 2) * cell_m / NN.R_UNIT_M
        logit = (m.att_conv(top)[:, 0] + m.att_a * r + m.att_b * r ** 2).reshape(n, h * w)
        mk = (water.reshape(n, h * w) >= NN.WATER_CELL_MIN).float()         # shapes taken from the map, see forward
        mk = torch.maximum(mk, 1.0 - mk.amax(1, keepdim=True))            # no water cell: every cell
        lo = logit.amin(1, keepdim=True)
        l2 = logit * mk + lo * (1.0 - mk)
        e = torch.exp(l2 - l2.amax(1, keepdim=True)) * mk
        return e / e.sum(1, keepdim=True)

    def forward(self, top: torch.Tensor, d: dict) -> torch.Tensor:
        m = self.m
        z = None
        if m.use_context:
            z = m.ctx(d["sc"])
            g, b = m.film(z).chunk(2, 1)
            top = top * (1 + g[:, :, None, None]) + b[:, :, None, None]
        n, c, h, w = top.shape
        # the per-cell inputs are reshaped with the map's own shape, so the graph has one set of symbolic
        # dimensions and the frozen ``quant_pre_process`` call (no dimension merging) can infer every shape
        flat = top.reshape(n, c, h * w)
        pooled = (flat * d["pool_w"].reshape(n, 1, h * w)).sum(2)
        if m.attention:
            aw = self.attention(top, d["water"], d["gsd"])
            pooled = torch.cat([pooled, (flat * aw.unsqueeze(1)).sum(2)], 1)
        anchor = d.get("anchor")
        parts = ([z] if m.use_context else []) + ([anchor.float()] if m.use_anchor else [])
        out = m.res(pooled, torch.cat(parts, 1), anchor)
        return torch.cat([out["mu"], 0.5 * out["logvar"]], 1)


class ExportAnchorLite(nn.Module):
    def __init__(self, model: NN.AnchorLite):
        super().__init__()
        if not isinstance(model, NN.AnchorLite):
            raise ExportError("the multi-input wrapper exports the ANCHOR-Lite family")
        self.names = input_names(model)
        self.image = _Image(model)
        self.tail = _Tail(model)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        d = dict(zip(self.names, args))
        return self.tail(self.image(d["x01"]), d)


def example_inputs(model: NN.AnchorLite, size: int = 512, n: int = 1, seed: int = 0) -> dict:
    """Random tensors of the right shapes (export tracing, MAC count, latency) — never data."""
    g = torch.Generator().manual_seed(seed)
    h = size // NN.STRIDE
    pw = torch.rand(n, h, h, generator=g)
    d = {"x01": torch.rand(n, 4, size, size, generator=g), "sc": torch.randn(n, NN.N_SC, generator=g),
         "anchor": torch.randn(n, len(model.targets), generator=g), "pool_w": pw / pw.sum((1, 2), keepdim=True),
         "water": torch.rand(n, h, h, generator=g), "gsd": torch.full((n,), 0.6)}
    return {k: d[k] for k in input_names(model)}


def export_onnx(model: NN.AnchorLite, path: str | Path, size: int = 512, opset: int = OPSET) -> Path:
    w = ExportAnchorLite(model.eval().cpu()).eval()
    ex = example_inputs(model, size)
    axes = {"x01": {0: "n", 2: "h", 3: "w"}, "sc": {0: "n"}, "anchor": {0: "n"}, "pool_w": {0: "n", 1: "h4", 2: "w4"},
            "water": {0: "n", 1: "h4", 2: "w4"}, "gsd": {0: "n"}, "out": {0: "n"}}
    torch.onnx.export(w, tuple(ex.values()), str(path), input_names=list(ex), output_names=["out"],
                      opset_version=opset, dynamic_axes={k: axes[k] for k in (*ex, "out")},
                      do_constant_folding=True, dynamo=False)
    return Path(path)


def image_nodes(onnx_path: str | Path) -> list[str]:
    """Names of the nodes of the image branch (normalisation, encoder, decoder) in an exported graph."""
    import onnx

    g = onnx.load(str(onnx_path)).graph
    names = [nd.name for nd in g.node if nd.name.startswith(IMAGE_SCOPE)]
    conv = [nd.name for nd in g.node if nd.op_type == "Conv"]
    if not names or not any(c in set(names) for c in conv):
        raise ExportError("no image-branch node was found in the graph (node names changed?)")
    return names


class _Feeds:
    def __init__(self, feeds: list):
        self.feeds = list(feeds)

    def get_next(self):
        return self.feeds.pop(0) if self.feeds else None

    def rewind(self):  # pragma: no cover - API completeness
        pass


def quantise_int8(fp32: str | Path, int8: str | Path, feeds: list, *, scope: str = "image") -> Path:
    """The frozen recipe (``lite.export.quantise_int8``: ``quant_pre_process`` → static QDQ, per-channel QInt8
    weights, QUInt8 activations, MinMax) for a multi-input graph. ``feeds``: calibration inputs, one dict per tile."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if scope not in SCOPES:
        raise ExportError(f"scope must be one of {SCOPES}")
    pre = Path(str(fp32).replace(".onnx", ".pre.onnx"))
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=False)
    extra = {"nodes_to_quantize": image_nodes(pre)} if scope == "image" else {}
    quantize_static(str(pre), str(int8), _Feeds(feeds), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    calibrate_method=CalibrationMethod.MinMax, **extra)
    return Path(int8)


def tile_inputs(chips_u8: np.ndarray, gsd_m) -> dict:
    """The image-derived inputs of the feature code: x01, pool_w, water, gsd (frozen feature functions)."""
    chips_u8 = np.ascontiguousarray(chips_u8)
    n, _, H, W = chips_u8.shape
    gsd = np.asarray(gsd_m, np.float32).reshape(n)
    water = F.water_mask(chips_u8).reshape(n, H // 4, 4, W // 4, 4).mean(axis=(2, 4)).astype(np.float32)
    return {"x01": chips_u8.astype(np.float32) / 255.0, "pool_w": F.pooling_weights(chips_u8, gsd, stride=4),
            "water": water, "gsd": gsd}


def calibration_feeds(model: NN.AnchorLite, tiles01: np.ndarray, sc: np.ndarray, anchor: np.ndarray | None,
                      gsd_m) -> list:
    """Calibration inputs: the frozen 2 × 2 mosaics of train chips (``lite.export.calibration_tiles``) with the
    [s, c], anchor and GSD of one train row each."""
    u8 = np.clip(np.round(np.asarray(tiles01) * 255.0), 0, 255).astype(np.uint8)
    t = tile_inputs(u8, gsd_m)
    t["x01"] = np.asarray(tiles01, np.float32)
    t["sc"] = np.asarray(sc, np.float32)
    if anchor is not None:
        t["anchor"] = np.asarray(anchor, np.float32)
    return [{k: t[k][i:i + 1] for k in input_names(model)} for i in range(len(u8))]


def session(path, threads: int = 1):
    from ..lite import export as X

    return X.session(path, threads)


def onnx_predict(path, inputs: dict, threads: int = 6, batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """(log10 μ, σ) of an exported model; ``inputs``: name → array, first axis = rows."""
    s = session(path, threads)
    names = [i.name for i in s.get_inputs()]
    n = len(inputs["x01"])
    out = []
    for i in range(0, n, batch):
        out.append(s.run(None, {k: np.ascontiguousarray(inputs[k][i:i + batch], dtype=np.float32) for k in names})[0])
    o = np.concatenate(out)
    T = o.shape[1] // 2
    return o[:, :T], np.exp(o[:, T:])


def torch_predict(model: NN.AnchorLite, inputs: dict, batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """The same quantities from ``AnchorLite.forward`` (the training code path), on CPU in float32."""
    model = model.eval().cpu()
    st = model.student
    mu, sd = [], []
    n = len(inputs["x01"])
    t = {k: torch.as_tensor(np.asarray(v, np.float32)) for k, v in inputs.items()}
    h = t["x01"].shape[-1] // NN.STRIDE
    with torch.no_grad():
        for i in range(0, n, batch):
            sl = slice(i, i + batch)
            k = len(t["x01"][sl])
            x = (t["x01"][sl] - st.x_mean.view(1, -1, 1, 1)) / st.x_std.view(1, -1, 1, 1)
            o = model(x, t["sc"][sl] if "sc" in t else torch.zeros(k, NN.N_SC),
                      t["anchor"][sl] if "anchor" in t else None, t["pool_w"][sl],
                      t["water"][sl] if "water" in t else torch.zeros(k, h, h),
                      t["gsd"][sl] if "gsd" in t else torch.ones(k))
            mu.append(o["mu"].numpy())
            sd.append(torch.exp(0.5 * o["logvar"]).numpy())
    return np.concatenate(mu), np.concatenate(sd)


def count_macs(model: NN.AnchorLite, size: int = 512) -> float:
    """MACs per tile as the frozen counter (``torch.utils.flop_counter``, conv / matmul only, FLOPs / 2)."""
    from torch.utils.flop_counter import FlopCounterMode

    w = ExportAnchorLite(model.eval().cpu()).eval()
    ex = example_inputs(model, size)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        w(*ex.values())
    return fc.get_total_flops() / 2.0


def count_params(model: NN.AnchorLite) -> dict:
    """Network parameters: all, and those of the exported graph (without the auxiliary KD map head)."""
    total = sum(p.numel() for p in model.parameters())
    aux = sum(p.numel() for p in model.student.head.parameters())
    return {"network": int(total), "aux_kd_head": int(aux), "exported": int(total - aux)}


def latency(path, feed: dict, *, threads: int = 1, warmup: int = 5, runs: int = 30) -> dict:
    """Desktop-proxy latency of the network part for one tile (frozen §9 settings); the feature code and the anchor
    trees are timed by the export script and added there."""
    s = session(path, threads)
    feed = {i.name: np.ascontiguousarray(feed[i.name], dtype=np.float32) for i in s.get_inputs()}
    for _ in range(warmup):
        s.run(None, feed)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        s.run(None, feed)
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "runs": runs,
            "threads": threads, "label": "desktop proxy (ONNX Runtime CPU EP), not device latency"}


__all__ = ["OPSET", "SCOPES", "ExportError", "ExportAnchorLite", "input_names", "example_inputs", "export_onnx",
           "image_nodes", "quantise_int8", "tile_inputs", "calibration_feeds", "onnx_predict", "torch_predict",
           "count_macs", "count_params", "latency"]
