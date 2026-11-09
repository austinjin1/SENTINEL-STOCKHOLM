# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Fuse deep model — cross-validated training on development folds (validation only).

    SENTINEL_GPUS=4 python -m sentinel.core.gpu --name fuse_train -- \
        python -B scripts/train/fuse_train.py --tier d --seeds 0 --run-name fuse_d_v1

For every development fold k: train on the other folds (early stopping on ~20 % of the training
*sites*, chosen by salted hash), predict fold k. Out-of-fold predictions are written in the Stage-0
format (row, config='fuse…', mu, sigma | p) so comparisons with the GBM comparators use identical rows.
Inference-time leave-one-modality-out predictions (P3 preview) are written too.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun
from sentinel.eval.bootstrap import bootstrap_ci
from sentinel.models.fuse import baselines as bl
from sentinel.models.fuse import data as fd
from sentinel.models.fuse import splits as sp
from sentinel.models.fuse import stage0 as s0
from sentinel.models.fuse.model import FuseConfig
from sentinel.models.fuse.train import TrainConfig, anchor_to_torch, predict, to_torch, train_fuse

SPLIT_SHA_D = "20adbe595ed42a46c490da31e16743353ce819e9d84cb8eedffa15303a26610a"


def log(*a):
    print(*a, flush=True)


def inner_val_sites(sites, frac=0.2, salt="fuse-inner-v1"):
    u = sorted(set(sites), key=lambda s: hashlib.sha1(f"{salt}|{s}".encode()).hexdigest())
    n = max(1, int(round(frac * len(u))))
    return set(u[:n])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["d"], default="d")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--targets", default="all", help="all | p1")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--latents", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=0, help="steps per epoch (0 = rows/batch)")
    ap.add_argument("--no-dropout-schedule", action="store_true")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--max-folds", type=int, default=99)
    ap.add_argument("--lomo", action="store_true")
    ap.add_argument("--anchored", action="store_true",
                    help="persistence-anchored μ for targets with a same-variable last value (owner modality kept)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda (cuda only via sentinel.core.gpu)")
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True     # review observation: determinism across reruns
    torch.backends.cudnn.benchmark = False
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    run = args.run_name or f"fuse_{args.tier}_{dt.datetime.now():%Y%m%dT%H%M%S}"
    out = paths.run_dir("fuse", run)
    split = sp.load_split(sp.TIER_D_SPLIT, SPLIT_SHA_D)
    df, genus = fd.load_tier_d()
    df = s0.drop_protected(df, split["site_role"], "site")
    fold = df["site"].map(split["site_fold"]).astype(int).to_numpy()
    specs = fd.tier_d_modalities(df)
    hub = {s.name for s in specs if s.hub}
    targets = fd.tier_d_targets(df)
    if args.targets == "p1":
        targets = [t for t in targets if t.name in fd.P1_TARGETS]
    Y, M, C = fd.tier_d_target_matrix(df, targets)
    is_bin = np.array([t.kind == "binary" for t in targets])
    tspec = [(t.family, t.horizon, t.kind) for t in targets]
    seeds = [int(s) for s in args.seeds.split(",")]
    all_folds = sorted(np.unique(fold))
    folds = all_folds[: args.max_folds]
    split_name = "smoke" if len(folds) < len(all_folds) else f"val_cv_lodgo{len(all_folds)}"
    mcfg = FuseConfig(d=args.d, n_latents=args.latents, n_layers=args.layers, n_heads=args.heads,
                      dropout=args.dropout)
    cfg = vars(args) | {"run": run, "device": device, "n_targets": len(targets), "mods": [s.name for s in specs]}
    log(f"device={device} rows={len(df)} targets={len(targets)} mods={[s.name for s in specs]} folds={folds}")
    with TrainingRun("fuse", f"fuse_train_{args.tier}", config=cfg, data=["fuse_tier_d"],
                     plan_ref="RESEARCH_PLAN §6 Stage 3 (dev CV)", iteration=run, artifacts_dir=str(out)) as trn:
        preds = []
        infos = []
        for seed in seeds:
            for k in folds:
                trm = fold != k
                ivs = inner_val_sites(df.loc[trm, "site"])
                va = np.flatnonzero(trm & df["site"].isin(ivs).to_numpy())
                tr = np.flatnonzero(trm & ~df["site"].isin(ivs).to_numpy() & M.any(1))
                ev = np.flatnonzero(fold == k)
                pre = fd.FoldPreprocessor(specs)
                pre.fit_micro(df, np.isin(np.arange(len(df)), tr), genus)
                dfk = pre.augment(df, genus)
                pre.fit(dfk, np.isin(np.arange(len(df)), tr), Y)
                Yz = pre.standardize_y(Y)
                Yz[:, is_bin] = Y[:, is_bin]
                tens = pre.tensors(dfk)
                anc = None
                if args.anchored:
                    A = np.full(Y.shape, np.nan)
                    owners = []
                    for j, t in enumerate(targets):
                        col = s0.PERSISTENCE_D.get(t.name) if t.kind == "gauss" else None
                        if col is not None and col in df.columns:
                            A[:, j] = (df[col].to_numpy(float) - pre.y_mean_[j]) / pre.y_std_[j]
                            owners.append(s0.anchor_owner(col))
                        else:
                            owners.append(None)
                    anc = anchor_to_torch(A, np.isfinite(A), owners, device)
                gs = {s.name: [len(g) for g in s.groups] for s in specs}
                tcfg = TrainConfig(epochs=args.epochs, batch=args.batch, lr=args.lr, weight_decay=args.wd,
                                   beta_nll=args.beta, seed=seed, steps_per_epoch=args.steps or None, patience=args.patience,
                                   p_all=1.0 if args.no_dropout_schedule else 0.2)
                log(f"seed {seed} fold {k}: train {len(tr)} innerval {len(va)} eval {len(ev)}")
                model, info = train_fuse(tens, Yz, M, C, is_bin, tr, va, gs, tspec, hub, mcfg, tcfg, device, log,
                                         anc=anc)
                info.update({"seed": seed, "fold": int(k)})
                infos.append({kk: vv for kk, vv in info.items() if kk != "history"} | {"history": info["history"]})
                tt = to_torch(tens, device)
                variants = {"fuse": None}
                if args.lomo:
                    for s in specs:
                        if not s.hub:
                            variants[f"fuse_minus_{s.name}"] = s.name
                for vname, drop in variants.items():
                    kf = None
                    if drop is not None:
                        kf = (lambda bt, dm=drop: {dm: torch.zeros_like(bt[dm]["present"], dtype=torch.bool)})
                    o = predict(model, tt, ev, args.batch, device, keep_fn=kf, anc=anc)
                    for j, t in enumerate(targets):
                        d = pd.DataFrame({"row": ev, "config": vname, "target": t.name, "seed": seed, "fold": int(k)})
                        if t.kind == "binary":
                            d["p"] = 1 / (1 + np.exp(-o[:, j, 0]))
                        else:
                            mu = o[:, j, 0] * pre.y_std_[j] + pre.y_mean_[j]
                            sg = np.exp(np.clip(o[:, j, 1], -4, 3)) * pre.y_std_[j]
                            d["mu"], d["sigma"] = mu, sg
                        preds.append(d)
                torch.save(model.state_dict(), out / f"model_s{seed}_f{k}.pt")
        P = pd.concat(preds, ignore_index=True)
        P["site"] = df["site"].to_numpy()[P["row"]]
        P["date"] = df["date"].to_numpy()[P["row"]]
        P.to_parquet(out / "oof_fuse.parquet", index=False)
        (out / "train_info.json").write_text(json.dumps(infos, indent=1, default=str))
        # seed-averaged (moment-matched mixture across seeds) summary for gauss targets
        summ = {}
        groups = df["site"].to_numpy()
        for j, t in enumerate(targets):
            if t.kind != "gauss":
                continue
            p = P[(P.target == t.name) & (P.config == "fuse")]
            g = p.groupby("row")
            mu = g["mu"].mean()
            m2 = g.apply(lambda x: np.mean(x["sigma"] ** 2 + x["mu"] ** 2))
            sg = np.sqrt(np.clip(m2 - mu ** 2, 1e-12, None))
            rows = mu.index.to_numpy()
            yy = Y[rows, j]
            ok = np.isfinite(yy) & ~C[rows, j]
            cr = bl.crps_gauss_rows(yy[ok], mu.to_numpy()[ok], sg.to_numpy()[ok])
            ci = bootstrap_ci(np.mean, cr, groups=groups[rows[ok]], n_boot=1000)
            summ[t.name] = {"crps": float(cr.mean()), "n": int(ok.sum()), "ci": [ci["ci_low"], ci["ci_high"]]}
            if t.name in fd.P1_TARGETS or t.name in ("ylab__tn_mgl", "ylab__tp_mgl", "ylab__doc_mgl"):
                log_result(workstream="fuse", experiment="fuse_train_d", model=f"fuse:{run}", metric="crps",
                           value=float(cr.mean()), split=split_name, n=int(ok.sum()), ci_low=ci["ci_low"],
                           ci_high=ci["ci_high"], ci_method="site-cluster bootstrap (B=1000), seed-mixture",
                           target=t.name, config=cfg, data_manifests=["fuse_tier_d"],
                           artifacts=[str(out / "oof_fuse.parquet")],
                           notes="model-space CRPS on all dev rows with the target observed (uncensored)")
                trn.metric(f"crps_{t.name}", float(cr.mean()))
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        log(json.dumps({k: v for k, v in summ.items() if k in fd.P1_TARGETS}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
