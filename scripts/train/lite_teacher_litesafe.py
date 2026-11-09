# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Lite-safe distillation teacher (protocol §5, review W8).

The stock M2 teacher (docs/model_cards/orbit_teacher.json) fitted on matchup and GLORIA rows at Lite val, test
and test_time water bodies (``distill.flag_overlap``), so it may not be used for Lite distillation as is. This
script retrains the **same recipe** (M2 code ``sentinel.models.orbit``, config ``configs/orbit/satwq.yaml``,
fit scheme ``base``, same seeds) with every teacher row removed that overlaps a Lite val/test/test_time site
(``distill.litesafe_drop_mask``: same site id, <= 2 km, or same HydroLAKES lake). Val is removed too, so KD
gains measured on Lite val are not inflated by a teacher that saw val lakes. No M2 file is written: outputs go
to ``$SENTINEL_RUNS/lite/teacher_litesafe/`` (checkpoints, ``seen_points.parquet``, ``manifest.json``).

    python -m sentinel.core.gpu --name lite_teacher_litesafe --prefer 2 -- \
        python -B scripts/train/lite_teacher_litesafe.py [--seeds 0 1 2 3 4] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import time

from sentinel.models.lite import runner as R

R.cpu_threads(2)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.seed import seed_everything  # noqa: E402
from sentinel.models.lite import distill as KD  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402

EVAL_ROLES = {"original": ("val", "test", "test_time"),
              "national": ("val", "val_n", "test", "test_time", "test_n", "test_time_n")}
ORBIT_CONFIG = paths.REPO_ROOT / "configs" / "orbit" / "satwq.yaml"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--track", default="original", choices=["original", "national"],
                    help="national: exclude every national val/test-like site (protocol §12)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args(argv)
    eval_roles = EVAL_ROLES[a.track]
    from sentinel.models.orbit import data as OD
    from sentinel.models.orbit import pipeline as PL

    cfg = yaml.safe_load(ORBIT_CONFIG.read_text())
    smoke = cfg.pop("smoke", {})
    if a.smoke:
        cfg = {**cfg, **{k: ({**cfg[k], **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v)
                         for k, v in smoke.items()}}
    tname = "teacher_litesafe" + ("" if a.track == "original" else "_national")
    out_root = paths.run_dir("lite", f"smoke/{tname}" if a.smoke else tname)

    # ---- overlap: stock teacher vs Lite sites; rows to drop
    if a.track == "original":
        split, labf = G.load_split(), None
    else:
        from sentinel.models.lite import national as N

        split, labf = N.load_split(), N.labels_path()
    sites = KD.lite_sites(split, labf)
    seen = KD.orbit_seen_points("base")
    flagged = KD.flag_overlap(sites, seen)
    before = KD.overlap_report(flagged)
    drop = KD.litesafe_drop_mask(seen, flagged[flagged["role"].isin(eval_roles)])
    kept = seen[~drop].reset_index(drop=True)
    after = KD.overlap_report(KD.flag_overlap(sites, kept))
    for r in eval_roles:
        if after.get(r, {}).get("water_bodies_overlap", 0):
            raise RuntimeError(f"litesafe exclusion left overlap in role {r}: {after[r]}")
    kept.to_parquet(out_root / "seen_points.parquet", index=False)
    drop_rows = set(seen.loc[drop & (seen["source"] == "matchup"), "key"])
    drop_gloria = set(seen.loc[drop & (seen["source"] == "gloria"), "key"])
    excl = {"stock_overlap": before, "litesafe_overlap": after, "eval_roles": list(eval_roles), "track": a.track,
            "n_seen_matchup": int((seen.source == "matchup").sum()), "n_seen_gloria": int((seen.source == "gloria").sum()),
            "n_drop_matchup": len(drop_rows), "n_drop_gloria": len(drop_gloria), "radius_km": KD.OVERLAP_KM}
    R.write_json(out_root / "exclusion.json", excl)
    print(json.dumps({k: v for k, v in excl.items() if k.startswith("n_")}))

    import torch

    from sentinel.models.orbit import satwq as S

    if a.device == "cuda":
        R.setup_gpu(f"lite_{tname}")
        torch.set_num_threads(2)
    else:
        torch.set_num_threads(6)
    device = torch.device(a.device)
    df, pix, info, masks = PL.load_dev(OD.CACHE_VERSION, "base")
    keep_row = ~df["row_id"].astype(str).isin(drop_rows).to_numpy()
    fit, inner = masks["fit"] & keep_row, masks["inner"] & keep_row
    g = OD.gloria_rows()
    g = g[~g["gloria_id"].astype(str).isin(drop_gloria)].reset_index(drop=True)
    ckpts, shas, runs = [], [], []
    for seed in a.seeds:
        seed_everything(seed)
        rcfg = {**cfg, "seed": seed, "fit_scheme": "base", "orbit_config": str(ORBIT_CONFIG), "litesafe": excl,
                "smoke": a.smoke}
        run_dir = paths.ensure(out_root / f"s{seed}")
        with R.training_run(f"lite_{tname}/s{seed}", rcfg, run_dir, a.smoke) as run:
            f = fit.copy()
            if a.smoke:
                idx = np.flatnonzero(f)
                f[:] = False
                f[np.random.default_rng(0).choice(idx, min(len(idx), 500), replace=False)] = True
                run.note("SMOKE TEST: <=500 fit rows; NOT a result")
            geo, _ = PL.geo_ctx(df, pix)
            norm = PL.Norm(geo[f], PL.ylog_matrix(df)[f])
            tr, inn = PL.make_set(df, pix, norm, f), PL.make_set(df, pix, norm, inner)
            model = S.SatWQNet(n_geo=len(PL.GEO_COLS), n_ctx=len(PL.CTX_COLS), d=cfg["d"], n_layers=cfg["n_layers"],
                               n_heads=cfg["n_heads"], k=cfg["k"], dropout=cfg["dropout"]).to(device)
            run.metric("n_fit", int(f.sum()))
            run.metric("n_inner", int(inner.sum()))
            run.metric("n_gloria", len(g))
            hist = {}
            if cfg["pretrain"]["enabled"]:
                pc = cfg["pretrain"]
                hist["pretrain"] = S.run_epochs(model, PL.gloria_set(g, norm), inner_set=None, epochs=pc["epochs"],
                                                lr=pc["lr"], batch=pc["batch"], device=device, seed=seed, aug=pc["aug"],
                                                y_mean=norm.y_mean, y_std=norm.y_std)
            fc = cfg["finetune"]
            t0 = time.time()
            hist["finetune"] = S.run_epochs(model, tr, inner_set=inn, epochs=fc["epochs"], lr=fc["lr"],
                                            batch=fc["batch"], device=device, seed=seed, aug=fc["aug"],
                                            y_mean=norm.y_mean, y_std=norm.y_std, patience=fc.get("patience", 15),
                                            select_targets=tuple(cfg["select_targets"]))
            run.metric("finetune_minutes", (time.time() - t0) / 60)
            run.metric("best_epoch", hist["finetune"]["best_epoch"])
            run.metric("inner_mdsa_select", hist["finetune"]["best_inner"])
            ck = run_dir / "model.pt"
            torch.save({"state_dict": model.state_dict(), "norm": norm.to_dict(), "config": rcfg,
                        "arch": {"n_geo": len(PL.GEO_COLS), "n_ctx": len(PL.CTX_COLS), "d": cfg["d"],
                                 "n_layers": cfg["n_layers"], "n_heads": cfg["n_heads"], "k": cfg["k"],
                                 "dropout": cfg["dropout"]},
                        "bands": list(OD.BANDS), "targets": list(OD.TARGETS), "geo_cols": list(PL.GEO_COLS),
                        "ctx_cols": list(PL.CTX_COLS)}, ck)
            R.write_json(run_dir / "history.json", hist)
            ckpts.append(str(ck))
            shas.append(G.artifact_sha256(ck))
            runs.append({"seed": seed, "best_inner": hist["finetune"]["best_inner"],
                         "best_epoch": hist["finetune"]["best_epoch"]})
            print(runs[-1], flush=True)
    man = {"label": "satwq_base_" + tname.replace("teacher_", ""), "track": a.track, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "checkpoints": ckpts,
           "sha256": shas, "seen_points": str(out_root / "seen_points.parquet"),
           "seen_points_sha256": G.artifact_sha256(out_root / "seen_points.parquet"),
           "exclusion": str(out_root / "exclusion.json"), "runs": runs, "smoke": a.smoke,
           "note": "M2 SatWQ-Net base recipe retrained by M7 without rows overlapping Lite val/test/test_time sites"}
    R.write_json(out_root / "manifest.json", man)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
