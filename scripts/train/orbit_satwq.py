#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Train one SatWQ-Net seed (orbit_protocol.md §4.1): GLORIA pretraining -> matchup fine-tuning (fit rows,
early stopping on inner rows) -> mixture predictions for inner + development val cells. Never loads
protected rows. One GPU through sentinel.core.gpu (prefer GPU 1).

    python -B scripts/train/orbit_satwq.py --config configs/orbit/satwq.yaml --seed 0 --run-name satwq_base_s0
    python -B scripts/train/orbit_satwq.py --config configs/orbit/satwq.yaml --smoke
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.runlog import TrainingRun  # noqa: E402
from sentinel.core.seed import seed_everything  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402


def deep_update(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_update(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="overrides key.sub=value (YAML values)")
    ap.add_argument("--version", default=D.CACHE_VERSION)
    ap.add_argument("--fit-scheme", default="base", choices=["base", "recent", "track_u"],
                    help="fit cell: base = train role x <=2021; recent = train role x <=2023 (cycle 2)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        node = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            node = node.setdefault(kk, {})
        node[ks[-1]] = yaml.safe_load(v)
    smoke = cfg.pop("smoke", {})
    if a.smoke:
        cfg = deep_update(cfg, smoke)
    cfg.update({"fit_scheme": a.fit_scheme, "seed": a.seed, "version": a.version, "smoke": bool(a.smoke), "config_file": a.config})
    name = a.run_name or f"satwq_s{a.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = paths.run_dir("orbit", f"smoke/{name}" if a.smoke else name)
    PL.setup_gpu(f"orbit_{name}")
    import torch

    from sentinel.models.orbit import satwq as S

    torch.set_num_threads(2)  # shared-host rule: GPU jobs use 2 CPU threads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(a.seed)
    PL.log_config_tried("satwq", cfg, run_dir, "started")
    with TrainingRun(PL.WORKSTREAM, f"{'SMOKE/' if a.smoke else ''}orbit_satwq/{name}", config=cfg,
                     data=[f"orbit_{a.version}", "gloria"], plan_ref=PL.PLAN_REF,
                     iteration=str(cfg.get("iteration", "base")), artifacts_dir=str(run_dir)) as run:
        df, pix, info, masks = PL.load_dev(a.version, a.fit_scheme)
        fit, inner = masks["fit"].copy(), masks["inner"]
        if a.smoke:
            rs = np.random.default_rng(0)
            idx = np.flatnonzero(fit)
            fit = np.zeros_like(fit)
            fit[rs.choice(idx, min(len(idx), int(smoke.get("smoke_n_fit", 500))), replace=False)] = True
            run.note("SMOKE TEST: <=500 fit rows, 2 epochs; NOT a result; nothing registered")
        geo, _ = PL.geo_ctx(df, pix)
        norm = PL.Norm(geo[fit], PL.ylog_matrix(df)[fit])
        tr = PL.make_set(df, pix, norm, fit)
        inn = PL.make_set(df, pix, norm, inner)
        model = S.SatWQNet(n_geo=len(PL.GEO_COLS), n_ctx=len(PL.CTX_COLS), d=cfg["d"], n_layers=cfg["n_layers"],
                           n_heads=cfg["n_heads"], k=cfg["k"], dropout=cfg["dropout"]).to(device)
        run.metric("n_params", S.n_params(model))
        print(f"SatWQ-Net params {S.n_params(model)}; fit {fit.sum()} inner {inner.sum()}")
        hist = {}
        sc = cfg.get("ssl", {}) or {}
        if sc.get("enabled"):          # orbit_protocol.md §11.6: SSL on unlabelled train-role <=2021 water pixels
            from sentinel.models.orbit import ssl as SSL

            t0 = time.time()
            hist["ssl"] = SSL.pretrain_ssl(model, epochs=int(sc["epochs"]), mask_frac=float(sc["mask_frac"]),
                                           lr=float(sc["lr"]), batch=int(sc["batch"]), device=device, seed=a.seed,
                                           max_pixels=sc.get("max_pixels"), variant=str(sc.get("pixels", "v1")))
            if a.fit_scheme == "track_u" and not str(sc.get("pixels", "v1")).endswith("_trackU"):
                raise RuntimeError("track-U refits must use the track-U SSL pixel set (ssl.pixels=v1_trackU)")
            run.metric("ssl_minutes", (time.time() - t0) / 60)
            run.metric("ssl_final_mse", hist["ssl"]["history"][-1]["ssl_mse"])
            run.note(f"SSL pixels {hist['ssl']['n_pixels']} ({hist['ssl']['manifest']}, sha {hist['ssl']['pixels_sha256'][:12]})")
        if cfg["pretrain"]["enabled"]:
            g = D.gloria_rows()
            gs = PL.gloria_set(g, norm)
            pc = cfg["pretrain"]
            t0 = time.time()
            hist["pretrain"] = S.run_epochs(model, gs, inner_set=None, epochs=pc["epochs"], lr=pc["lr"], batch=pc["batch"],
                                            device=device, seed=a.seed, aug=pc["aug"], y_mean=norm.y_mean,
                                            y_std=norm.y_std)
            run.metric("pretrain_minutes", (time.time() - t0) / 60)
            run.metric("pretrain_final_nll", hist["pretrain"]["history"][-1]["train_nll"])
            run.metric("n_gloria", len(g))
        fc = cfg["finetune"]
        t0 = time.time()
        hist["finetune"] = S.run_epochs(model, tr, inner_set=inn, epochs=fc["epochs"], lr=fc["lr"], batch=fc["batch"],
                                        device=device, seed=a.seed, aug=fc["aug"], y_mean=norm.y_mean, y_std=norm.y_std,
                                        patience=fc.get("patience", 15), select_targets=tuple(cfg["select_targets"]))
        run.metric("finetune_minutes", (time.time() - t0) / 60)
        run.metric("best_epoch", hist["finetune"]["best_epoch"])
        run.metric("inner_mdsa_select", hist["finetune"]["best_inner"])
        ckpt = run_dir / "model.pt"
        torch.save({"state_dict": model.state_dict(), "norm": norm.to_dict(), "config": cfg,
                    "arch": {"n_geo": len(PL.GEO_COLS), "n_ctx": len(PL.CTX_COLS), "d": cfg["d"],
                             "n_layers": cfg["n_layers"], "n_heads": cfg["n_heads"], "k": cfg["k"],
                             "dropout": cfg["dropout"]},
                    "bands": list(D.BANDS), "targets": list(D.TARGETS), "geo_cols": list(PL.GEO_COLS),
                    "ctx_cols": list(PL.CTX_COLS)}, ckpt)
        out_mask = ~masks["fit"]           # inner + development val cells (never protected: guard)
        ds = PL.make_set(df, pix, norm, out_mask)
        ctx_off = bool(cfg["finetune"]["aug"].get("ctx_off", False))
        w, mu, sg = S.predict_mixture(model, ds, device, norm.y_mean, norm.y_std, ctx_off=ctx_off)
        PL.save_predictions(run_dir / "preds.npz", df, out_mask, {"w": w, "mu": mu, "sg": sg})
        w2, mu2, sg2 = S.predict_mixture(model, ds, device, norm.y_mean, norm.y_std, ctx_off=True)
        PL.save_predictions(run_dir / "preds_ctx_off.npz", df, out_mask, {"w": w2, "mu": mu2, "sg": sg2})
        w3, mu3, sg3 = S.predict_mixture(model, ds, device, norm.y_mean, norm.y_std, ctx_off=ctx_off, band_subset=D.RGBN)
        PL.save_predictions(run_dir / "preds_rgbn.npz", df, out_mask, {"w": w3, "mu": mu3, "sg": sg3})
        PL.write_json(run_dir / "history.json", hist)
        # inner/val chl-a MdSA as run diagnostics (registry entries are written by orbit_compare.py)
        from sentinel.models.orbit.evaluate import mdsa_log10

        med = S.mixture_quantiles(w[:, 0], mu[:, 0], sg[:, 0], (0.5,))[:, 0]
        yl = np.log10(df.chla.to_numpy(float))[out_mask]
        for c in ("inner", PL.PRIMARY_CELL):
            m = (masks[c][out_mask]) & np.isfinite(yl)
            if m.any():
                run.metric(f"{c}_mdsa_chla", mdsa_log10(med[m] - yl[m]))
        PL.log_config_tried("satwq", cfg, run_dir, "completed",
                            {"inner_mdsa_select": hist["finetune"]["best_inner"], "smoke": a.smoke})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
