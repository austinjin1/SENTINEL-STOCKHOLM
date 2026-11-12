#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-F UK development design (protocol ``bioassess_f_protocol.md`` §9.4; gates G5 and G10).

Allowed UK rows only; no French file is read. Leave-one-area-out: training = UK rows up to 2015 outside the area
(1-km buffer), pseudo-test = ``pool_fit`` invertebrate rows of the area dated 2016-2020. References first (G5),
then #7 ``ma_zs`` and #8 ``ens_zs`` (seed 0). Every fit passes ``UKDev.assert_fit_ids`` -> ``ftrack.assert_fit_ids(``.

Stages (each resumable; a unit that has its prediction file is skipped):

    design                         build and store the design and the fold report (counts only)
    refs    --models a,b [--areas x,y]      comparators / B2 members, one file per (model, area)
    ma      [--areas x,y] [--chunk-epochs N] [--time-only E]   #7 fits (GPU through the lock helper)
    table                          scores of every stored model on the pseudo-test rows; #8 = mean of its members

GPU stages are started through the lock helper:

    SENTINEL_GPUS=$(python -c "from sentinel.models.bioassess import ftrack as F; \
        print(','.join(map(str, F.unoccupied_gpus())))") \
    python -m sentinel.core.gpu --name m5f_ukdev --min-free-mb 9000 -- \
        python -B scripts/eval/bioassess_f_ukdev.py ma --areas ... --chunk-epochs N

Exit codes: 0 done; 3 time limit reached (run again).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun
from sentinel.models.bioassess import fdata as FD
from sentinel.models.bioassess import fdev as FV
from sentinel.models.bioassess import ftrack as F

PLAN = "bioassess_f_protocol.md v1 §9.4 UK development design (seed 0)"
DATA = ["uk_ea_ecology", "taxonomy_harmonised"]
SPLIT = "ukdev: leave-one-area-out, UK <= 2015 train, UK 2016-20 pool_fit pseudo-test"
OUT = FV.UKDEV_DIR
PRIMARY = list(FD.PRIMARY["F-Z"])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_dev() -> FV.UKDev:
    f = OUT / "design.pkl"
    if not f.exists():
        raise SystemExit("run the 'design' stage first")
    with open(f, "rb") as fh:                       # our own file, written by the 'design' stage below
        D, role = pickle.load(fh)
    return FV.UKDev(D, role)


def pred_file(model: str, area: str, seed: int = 0) -> Path:
    d = OUT / "pred" / model / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d / (area.replace("/", "_").replace(" ", "_") + ".parquet")


def stage_design(a) -> int:
    F.assert_stamped()
    t0 = time.time()
    dev = FV.UKDev.build()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "design.pkl", "wb") as fh:
        pickle.dump((dev.D, dev.role), fh, protocol=4)
    rep = dev.report()
    doc = {"design": "UK development design (protocol §9.4)", "info": dev.D.info, "areas": rep,
           "train_to": FV.UKDEV_TRAIN_TO.isoformat(),
           "test_period": [FV.UKDEV_TEST_FROM.isoformat(), FV.UKDEV_TEST_TO.isoformat()]}
    (OUT / "folds.json").write_text(json.dumps(doc, indent=1, default=str))
    log(f"design: {dev.D.info['n_rows']} rows ({dev.D.info['n_invertebrate_rows']} invertebrate, "
        f"{dev.D.info['n_diatom_rows']} diatom-only); {len(rep)} areas; "
        f"pseudo-test rows {sum(v['n_test'] for v in rep.values())}; {time.time() - t0:.0f}s")
    for g, v in rep.items():
        log(f"  {g}: test {v['n_test']} ({v['n_test_clusters']} clusters), context {v['n_context']}, "
            f"train {v['n_train']} ({v['n_train_inv']} invertebrate)")
    return 0


def stage_refs(a) -> int:
    import os

    F.assert_stamped()
    dev = load_dev()
    TB = FV.TaxonBlocks(dev.D)
    areas = a.areas.split(",") if a.areas else dev.areas
    models = a.models.split(",")
    deadline = time.monotonic() + 60.0 * a.max_minutes
    device = "cpu" if a.cpu else "cuda"
    for model in models:
        done_units, t_model = 0, time.time()
        cfg = {"model": model, "seed": a.seed, "spec": repr(FV.ML_SPECS.get(model)), "targets": PRIMARY,
               "rf_max_rows": FV.RF_MAX_ROWS, "tabpfn_max_rows": FV.TABPFN_MAX_ROWS, "threads": a.threads}
        with TrainingRun("bioassess", f"bioassess_f/ukdev/{model}", config=cfg, data=DATA, plan_ref=PLAN,
                         iteration="dev", artifacts_dir=str(OUT / "pred" / model)) as run:
            for g in areas:
                f = pred_file(model, g, a.seed)
                if f.exists():
                    continue
                if time.monotonic() > deadline:
                    log(f"time limit reached before {model}/{g}; run again to resume")
                    run.note("time limit reached; resumable")
                    return 3
                tr, te = dev.training_ids(g), dev.test_ids(g)
                dev.assert_fit_ids(tr, g)                      # -> ftrack.assert_fit_ids( ... ) and the fold pool
                t0, out, info = time.time(), pd.DataFrame(index=te), {}
                for t in PRIMARY:
                    inf = {}
                    out[t] = FV.fit_predict(model, t, dev, TB, tr, te, seed=a.seed, n_jobs=a.threads, device=device,
                                            info=inf)
                    info[t] = inf
                out.to_parquet(f)
                f.with_suffix(".json").write_text(json.dumps({"model": model, "area": g, "seed": a.seed,
                                                              "n_train_pool": int(len(tr)), "n_test": int(len(te)),
                                                              "seconds": time.time() - t0, "info": info,
                                                              "device": device if "tabpfn" in model else "cpu",
                                                              "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}))
                done_units += 1
                log(f"{model} / {g}: {len(te)} rows, {time.time() - t0:.0f}s")
            run.metric("units_fitted", done_units)
            run.metric("seconds", time.time() - t_model)
            run.note("predictions stored; scores come from the 'table' stage")
    return 0


def stage_ma(a) -> int:
    import os

    import torch

    from sentinel.models.bioassess import bioset_ma as MA

    F.assert_stamped()
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    dev = load_dev()
    D, S = dev.D, dev.D.samples
    areas = a.areas.split(",") if a.areas else dev.areas
    k_of = {g: k for k, g in enumerate(dev.areas)}
    cols = list(FD.Y_COLUMNS)
    deadline = time.monotonic() + 60.0 * a.max_minutes
    name = "ma_zs"
    for g in areas:
        f = pred_file(name, g, a.seed)
        if f.exists() and not a.time_only:
            continue
        if time.monotonic() > deadline:
            log(f"time limit reached before {g}; run again to resume")
            return 3
        over = {"max_epochs": a.time_only, "patience": a.time_only} if a.time_only else {}
        cfg = MA.recipe(name, **over)
        tag = FV.fold_tag(g)
        rc = {"recipe": name, "cfg": cfg, "area": g, "seed": a.seed, "device": device,
              "chunk_epochs": a.chunk_epochs, "time_only": a.time_only}
        exp = f"bioassess_f/ukdev/{name}" + ("_timing" if a.time_only else "")
        with TrainingRun("bioassess", exp, config=rc, data=DATA, plan_ref=PLAN, iteration="dev",
                         artifacts_dir=str(OUT / "ckpt" / name)) as run:
            t0 = time.time()
            pool = MA.usable_ids(S, dev.training_ids(g), cfg["modalities"])
            dev.assert_fit_ids(pool, g)                       # -> ftrack.assert_fit_ids( ... ) and the fold pool
            fit_ids, val_ids = MA.inner_split(S, pool, float(cfg["inner_val_frac"]), float(cfg["inner_block_km"]),
                                              seed=a.seed)
            dev.assert_fit_ids(fit_ids.union(val_ids), g)
            feat = MA.MAFeaturizer(cfg).fit(D, pool)
            tr, va = feat.transform(D, fit_ids), feat.transform(D, val_ids)
            t_feat = time.time() - t0
            log(f"{g}: pool {len(pool)}, fit {len(fit_ids)}, inner-val {len(val_ids)}; featurized in {t_feat:.0f}s")
            fit_args = (cfg, feat, tr, D.Y.loc[fit_ids, cols].to_numpy(float), va,
                        D.Y.loc[val_ids, cols].to_numpy(float))
            # timing run: no inner-validation number is printed (§9.1: chunk length before any validation number)
            fit_kw = {"columns": cols, "primary": FD.PRIMARY["F-Z"], "seed": a.seed, "device": device,
                      "log": None if a.time_only else log}
            ck_dir = OUT / "ckpt" / name / f"seed{a.seed}"
            ck_dir.mkdir(parents=True, exist_ok=True)
            stem = g.replace("/", "_").replace(" ", "_")
            resume = ck_dir / f"{stem}.resume.pt"
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if a.time_only:
                fitted = MA.fit_ma(*fit_args, **fit_kw)
            else:
                state = MA.load_fit_state(resume, S, FV.TRACK, tag, FV.GUARD_DESIGN) if resume.exists() else None
                if state is not None:
                    log(f"{g}: resumed at epoch {state['epoch']} (resume point {len(state['chunks'])})")
                    log_result(workstream="bioassess", experiment=f"bioassess_f/ukdev/{name}/resume", model=name,
                               metric="resume_epoch", value=float(state["epoch"]), split=SPLIT, target=g,
                               notes="resume point of a chunked fit (lead ruling (o))",
                               extra={"area": g, "seed": a.seed, "chunks": state["chunks"]})
                fitted = None
                while fitted is None:
                    fitted, state = MA.fit_ma_chunk(*fit_args, state=state, chunk_epochs=a.chunk_epochs, **fit_kw)
                    if fitted is None:
                        MA.save_fit_state(resume, state, pool, track=FV.TRACK, g=tag, design=FV.GUARD_DESIGN)
                        if time.monotonic() > deadline:
                            log(f"time limit reached inside {g} at epoch {state['epoch']}; run again to resume")
                            run.note(f"stopped at a resume point (epoch {state['epoch']}); resumable")
                            return 3
            t_fit = time.time() - t0
            n_ep = len(fitted.history)
            vram = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else 0.0
            ep_s = t_fit / max(n_ep, 1)
            log(f"{g}: {n_ep} epochs, {ep_s:.1f} s/epoch, best epoch {fitted.best_epoch}, peak VRAM {vram:.0f} MB")
            for k, v in (("fit_rows", len(fit_ids)), ("inner_val_rows", len(val_ids)), ("epochs", n_ep),
                         ("epoch_seconds", ep_s), ("featurize_seconds", t_feat), ("peak_vram_train_mb", vram),
                         ("n_params", fitted.n_params)):
                run.metric(k, v)
            if a.time_only:
                run.note("timing run on UK rows: sets the chunk length; no pseudo-test prediction is made")
                log_result(workstream="bioassess", experiment=exp, model=name, metric="epoch_seconds", value=ep_s,
                           split="ukdev training rows (timing only)", n=len(fit_ids), target=g, config=rc,
                           notes="timing run that fixes the chunk schedule (protocol §9.1); no validation number",
                           extra={"peak_vram_train_mb": vram, "featurize_seconds": t_feat, "device": device,
                                  "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
                continue
            MA.save_checkpoint(ck_dir / f"{stem}.pt", fitted, feat, pool, track=FV.TRACK, g=tag,
                               design=FV.GUARD_DESIGN)
            te, ctx = dev.test_ids(g), dev.context_ids(g)
            q = feat.transform(D, ctx)                        # the candidate rows of the mirror rule (inputs only)
            p = fitted.predict(q, region=np.array([g] * len(q), object), device=device, seed=a.seed,
                               region_index={g: k_of[g]})
            pos = pd.Index(ctx).get_indexer(te)
            p0 = fitted.predict(feat.transform(D, te), region=None, device=device, seed=a.seed)
            for nm, loc, sc in ((name, p["loc"][pos], p["scale"][pos]), (name + "_noregion", p0["loc"], p0["scale"])):
                out = pd.DataFrame(loc, index=te, columns=cols)
                for j, c in enumerate(cols):
                    out["scale__" + c] = sc[:, j]
                out.to_parquet(pred_file(nm, g, a.seed))
            f.with_suffix(".json").write_text(json.dumps({
                "model": name, "area": g, "seed": a.seed, "n_pool": int(len(pool)), "n_fit": int(len(fit_ids)),
                "n_inner_val": int(len(val_ids)), "n_test": int(len(te)), "n_context": int(len(ctx)),
                "epochs": n_ep, "best_epoch": int(fitted.best_epoch), "epoch_seconds": ep_s, "fit_seconds": t_fit,
                "chunks": fitted.info.get("chunks"), "peak_vram_train_mb": vram, "device": device,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}))
            if resume.exists():
                resume.unlink()
                Path(str(resume) + F.SIDECAR_SUFFIX).unlink(missing_ok=True)
            run.note("predictions stored for the pseudo-test rows (with region context and with the slot masked); "
                     "scores come from the 'table' stage")
    return 0


def _stored(model: str, areas, seed: int = 0):
    fs = [pred_file(model, g, seed) for g in areas]
    if not all(f.exists() for f in fs):
        return None
    return pd.concat([pd.read_parquet(f) for f in fs])


def stage_table(a) -> int:
    from sentinel.models.bioassess import evaluate as EV

    dev = load_dev()
    S, Y = dev.D.samples, dev.D.Y
    areas = dev.areas
    te = pd.Index(np.concatenate([dev.test_ids(g).to_numpy(object) for g in areas]))
    region = S.loc[te, "region"].astype(str).to_numpy()
    fill = {g: {t: float(np.nanmean(Y.loc[dev.training_ids(g), t].to_numpy(float))) for t in PRIMARY} for g in areas}
    names = sorted(p.name for p in (OUT / "pred").iterdir() if p.is_dir())
    preds = {}
    for m in names:
        df = _stored(m, areas, a.seed)
        if df is not None:
            preds[m] = df.reindex(te)[PRIMARY]
    for ens, members in {**FV.ENSEMBLES, "ens_zs": ("ma_zs",) + FV.MEMBERS_B2,
                         "ens_zs_noregion": ("ma_zs_noregion",) + FV.MEMBERS_B2}.items():
        if all(m in preds for m in members):
            preds[ens] = sum(preds[m] for m in members) / len(members)
    cand = {"ma_zs", "ens_zs", "ma_zs_noregion", "ens_zs_noregion"}
    missing_refs = [m for m in FV.UK_REFERENCES if m not in preds]
    if not a.candidates or missing_refs:                      # gate G5: references first; §9.5: score after G4
        if a.candidates:
            log(f"gate G5: candidates not scored; references without stored predictions: {missing_refs}")
        preds = {m: P for m, P in preds.items() if m not in cand}
    rows = []
    for m, P in sorted(preds.items()):
        for t in PRIMARY:
            y = Y.loc[te, t].to_numpy(float)
            p = P[t].to_numpy(float)
            miss = ~np.isfinite(p) & np.isfinite(y)
            p = np.where(np.isfinite(p), p, np.array([fill[g][t] for g in region]))
            sm = EV.summary(y, p, region)
            sm.pop("r2_by_region", None)
            rows.append({"model": m, "target": t, "candidate": m in cand, "n_missing_filled": int(miss.sum()), **sm})
    T = pd.DataFrame(rows)
    if T.empty:
        log("no complete model stored yet")
        return 0
    T["selection_component"] = (T["r2"] + T["r2_anomaly"]) / 2
    T.to_csv(OUT / "ukdev_table.csv", index=False)
    piv = T.pivot(index="model", columns="target", values="selection_component")[PRIMARY]
    piv["mean"] = piv.mean(axis=1)
    log("mean over targets of (pooled R2 + r2_anomaly) / 2, seed 0:\n" + piv.round(4).to_string())
    if a.log_registry:
        for r in rows:
            for metric in ("r2", "r2_anomaly"):
                if np.isfinite(r[metric]):
                    log_result(workstream="bioassess", experiment=f"bioassess_f/ukdev/{r['model']}", model=r["model"],
                               metric=metric, value=float(r[metric]), split=SPLIT, n=int(r["n"]), target=r["target"],
                               is_baseline=not r["candidate"], config={"model": r["model"], "seed": a.seed},
                               data_manifests=DATA,
                               notes="UK development design, seed 0; point estimate on the pooled pseudo-test rows",
                               extra={"n_missing_filled": r["n_missing_filled"], "rmse": r["rmse"],
                                      "spearman": r["spearman"], "r2_region_median": r["r2_region_median"]})
        log(f"registry: {2 * len(rows)} rows")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["design", "refs", "ma", "table"])
    ap.add_argument("--models", default="")
    ap.add_argument("--areas", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=100.0)
    ap.add_argument("--chunk-epochs", type=int, default=None)
    ap.add_argument("--time-only", type=int, default=0, help="timing run of E epochs on the first area given")
    ap.add_argument("--log-registry", action="store_true")
    ap.add_argument("--candidates", action="store_true",
                    help="also score #7 / #8 (only when every UK reference is stored; after the lead's G4 stamp)")
    args = ap.parse_args()
    STAGES = {"design": stage_design, "refs": stage_refs, "ma": stage_ma, "table": stage_table}
    raise SystemExit(STAGES[args.stage](args))
