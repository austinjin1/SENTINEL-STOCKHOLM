# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Export a trained SENTINEL-Lite student to ONNX FP32 + INT8, measure params / MACs / file sizes and the
desktop-proxy latency (1 and 4 threads), and the INT8-vs-FP32 Δ MdSA on the val role (protocol §9).

    python -B scripts/eval/lite_export.py --checkpoint $SENTINEL_RUNS/lite/<run>/model_seed0.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from sentinel.models.lite import runner as R

R.cpu_threads(6)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import evaluate as E  # noqa: E402
from sentinel.models.lite import export as X  # noqa: E402
from sentinel.models.lite import features as F  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402
from sentinel.models.lite.model import ExportWrapper, count_macs, count_params  # noqa: E402
from sentinel.models.lite.train import build_model, predict, Inputs  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args(argv)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = {**ck["cfg"], "pretrained": False, "gloria_init": None}
    if ck["cfg"].get("smoke") or "smoke" in Path(a.checkpoint).parts:
        a.no_register = True          # smoke checkpoints are never registered (review W9)
        print("smoke checkpoint: results will not be registered")
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    scene = cfg["input_mode"] == "scene"
    out = Path(a.out or Path(a.checkpoint).parent / f"export_seed{ck['seed']}")
    out.mkdir(parents=True, exist_ok=True)
    rep = {"checkpoint": a.checkpoint, "params": count_params(model),
           "gmacs_512": count_macs(ExportWrapper(model, scene_mode=scene, rgb_only=cfg.get("rgb_only", False))) / 1e9,
           "mac_counter": "torch.utils.flop_counter (conv/matmul only), FLOPs/2", "cpu": X.cpu_info()}
    fp32 = X.export_onnx(model, out / "lite_fp32.onnx", scene_mode=scene, rgb_only=cfg.get("rgb_only", False))
    split = G.load_split()
    lab = D.role_frame(split, ("train", "val"), training=True)
    G.assert_no_protected(lab["role"])
    tr, va = lab[lab.role == "train"].reset_index(drop=True), lab[lab.role == "val"].reset_index(drop=True)
    tr_chips = np.asarray(D.cached_chips(tr, "train"))
    calib = X.calibration_tiles(tr_chips, n=32)
    if scene:   # calibrate on scene-normalised tiles built from the same chips' DOQQ stats (approx: train median)
        sp = D.scene_norm_params(tr, D.scene_stats())
        c, s = np.median(sp[:, :, 0], 0), np.median(sp[:, :, 1], 0)
        calib = ((calib * 255.0) - c[None, :, None, None]) / s[None, :, None, None]
    int8 = X.quantise_int8(fp32, out / "lite_int8.onnx", calib.astype(np.float32))
    rep["onnx_fp32_mb"] = fp32.stat().st_size / 1e6
    rep["onnx_int8_mb"] = int8.stat().st_size / 1e6
    os.environ["OMP_NUM_THREADS"] = "1"
    for name, p in (("fp32", fp32), ("int8", int8)):
        for th in (1, 4):
            rep[f"latency_{name}_{th}thr"] = X.latency(p, threads=th, runs=30)
    # INT8 vs FP32 accuracy on val (chip level)
    va_chips = np.asarray(D.cached_chips(va, "val"))
    pw = F.pooling_weights(va_chips, va["gsd_m"].to_numpy(), 4)
    spv = D.scene_norm_params(va, D.scene_stats()) if scene else None
    T = model.n_targets
    crop = int(cfg.get("crop", 0) or 0)
    if crop and crop < va_chips.shape[-1]:   # crop-trained models are scored on the centre crop (as train.predict)
        c0 = (va_chips.shape[-1] - crop) // 2
        oc = va_chips[..., c0:c0 + crop, c0:c0 + crop]
        ow = pw[:, c0 // 4:(c0 + crop) // 4, c0 // 4:(c0 + crop) // 4]
        ow = ow / np.clip(ow.sum((1, 2), keepdims=True), 1e-8, None)
    else:
        oc, ow = va_chips, pw
    rep["onnx_eval_crop"] = crop
    mu32 = X.onnx_predict(fp32, np.ascontiguousarray(oc), ow, T, scene_params=spv)
    mu8 = X.onnx_predict(int8, np.ascontiguousarray(oc), ow, T, scene_params=spv)
    inp = Inputs(va_chips, D.scene_norm_params(va, D.scene_stats()), pw, None, "cpu")
    cfg_cpu = {**cfg, "amp": False}
    mut, _ = predict(model, inp, cfg_cpu)
    y = va[[f"y_{t}" for t in D.TARGETS]].to_numpy(float)
    acc = {}
    for j, t in enumerate(D.TARGETS):
        ok = np.isfinite(y[:, j])
        fn, mname, _ = E.primary_metric(t)
        acc[t] = {"metric": mname, "torch": fn(y[ok, j], mut[ok, j]), "onnx_fp32": fn(y[ok, j], mu32[ok, j]),
                  "onnx_int8": fn(y[ok, j], mu8[ok, j]),
                  "max_abs_fp32_vs_torch": float(np.max(np.abs(mu32 - mut)))}
        acc[t]["delta_int8"] = acc[t]["onnx_int8"] - acc[t]["onnx_fp32"]
    rep["val_accuracy"] = acc
    rep["budget"] = {"params_le_3M": rep["params"] <= 3e6, "gmacs_le_2": rep["gmacs_512"] <= 2.0,
                     "int8_le_4MB": rep["onnx_int8_mb"] <= 4.0,
                     "proxy_latency_le_100ms": rep["latency_int8_1thr"]["median_ms"] <= 100.0}
    R.write_json(out / "export_report.json", rep)
    print({k: v for k, v in rep.items() if k not in ("cpu",)})
    if not a.no_register:
        from sentinel.core.registry import log_result

        from sentinel.core.registry import registry_frame

        # one registry model per seed (lookup() must be unambiguous); rows of an earlier export of the same
        # artifact directory (same metric/target) are superseded
        old = registry_frame()
        old = old[(old["experiment"] == "lite_export") & old["artifacts"].apply(lambda x: str(out) in (x or []))]

        def prev(metric, target=None):
            h = old[(old["metric"] == metric) & ((old["target"] == target) if target else old["target"].isna())]
            return h.sort_values("timestamp")["id"].iloc[-1] if len(h) else None

        common = dict(workstream=R.WORKSTREAM, experiment="lite_export",
                      model=f"lite_student_{cfg.get('name', 'x')}_seed{ck['seed']}",
                      split="val", config=R.public_config(cfg), data_manifests=list(R.DATA_MANIFESTS),
                      artifacts=[str(out)])
        for metric, value, notes in (("params", rep["params"], "trainable+buffers excluded; parameters only"),
                                     ("gmacs_512x512", rep["gmacs_512"], rep["mac_counter"]),
                                     ("onnx_int8_mb", rep["onnx_int8_mb"], ""),
                                     ("latency_int8_1thr_median_ms", rep["latency_int8_1thr"]["median_ms"],
                                      "DESKTOP PROXY, not device latency: " + str(rep["cpu"].get("cpu")))):
            log_result(metric=metric, value=value, notes=notes, supersedes=prev(metric), **common)
        for t, v in acc.items():
            log_result(metric=f"{v['metric']}_int8_minus_fp32", value=v["delta_int8"], target=t,
                       n=int(np.isfinite(y[:, D.TARGETS.index(t)]).sum()), supersedes=prev(f"{v['metric']}_int8_minus_fp32", t),
                       notes=f"ONNX evaluated on the centre {crop}-px crop" if crop else "", **common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
