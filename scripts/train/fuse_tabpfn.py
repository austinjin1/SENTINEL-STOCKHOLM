# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""TabPFN v2 comparator for SENTINEL-Fuse Tier D (protocol §6; review W10). Validation only.

    python -B scripts/train/fuse_tabpfn.py --targets p1 --run-name tabpfn_d_p1_v1 [--device cuda]

Per development fold and target: TabPFN (``tabpfn==2.2.1`` via the shared loader with the scikit-learn
shim, ``sentinel.models.chemtox.tabpfn_model.load_tabpfn``) on the concatenated Tier D features
(all modalities + hub, NaN where absent; ≤ 500 features), ≤ 10,000 training rows sampled with equal
quota per training site. Persistence-anchored like ``concat@a``: where the anchor exists the model
predicts y − anchor; rows without an anchor get a level model. Predictive CRPS from 99 quantiles
(τ = 0.01…0.99; CRPS ≈ 2·mean_τ pinball_τ), stored per row in ``crps`` (``mu`` = median) so that
``fuse_compare.py`` scores it on the identical rows. Test-role rows are dropped on load.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun
from sentinel.models.chemtox.tabpfn_model import TABPFN_PIN, load_tabpfn
from sentinel.models.fuse import data as fd
from sentinel.models.fuse import stage0 as s0
from sentinel.models.fuse import stats as fst

sys.path.insert(0, str(paths.REPO_ROOT / "scripts" / "train"))
from fuse_stage0 import tier_d_setup  # noqa: E402

TAUS = np.linspace(0.01, 0.99, 99)


def quantile_crps(y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """CRPS ≈ 2 · mean over τ of the pinball loss (rows × τ quantile matrix)."""
    d = y[:, None] - Q
    return 2.0 * np.mean(np.maximum(TAUS * d, (TAUS - 1) * d), axis=1)


def site_quota_sample(sites: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    idx = np.arange(len(sites))
    if len(idx) <= n_max:
        return idx
    rs = np.random.default_rng(seed)
    u = np.unique(sites)
    quota = int(np.ceil(n_max / len(u)))
    take = []
    for s in u:
        m = idx[sites == s]
        take.append(rs.choice(m, min(quota, len(m)), replace=False))
    take = np.concatenate(take)
    if len(take) > n_max:
        take = rs.choice(take, n_max, replace=False)
    return np.sort(take)


def fit_predict(Reg, Xtr, ytr, Xev, device, seed, chunk=2000):
    m = Reg(device=device, n_estimators=4, random_state=seed, ignore_pretraining_limits=True)
    m.fit(Xtr, ytr)
    out = []
    for i in range(0, len(Xev), chunk):
        q = m.predict(Xev[i:i + chunk], output_type="quantiles", quantiles=list(TAUS))
        out.append(np.column_stack(q))
    return np.concatenate(out, 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="p1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-rows", type=int, default=10000)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    Reg, ver = load_tabpfn()
    out = paths.run_dir("fuse", args.run_name)
    df, genus, fold, specs, mods, hub = tier_d_setup()
    targets = fd.tier_d_targets(df)
    if args.targets == "p1":
        targets = [t for t in targets if t.name in fd.P1_TARGETS]
    elif args.targets == "lab":
        targets = [t for t in targets if t.name in ("ylab__tn_mgl", "ylab__tp_mgl", "ylab__doc_mgl")]
    else:
        targets = [t for t in targets if t.name in args.targets.split(",")]
    sites = df["site"].to_numpy()
    cfg = vars(args) | {"tabpfn": ver, "pin": TABPFN_PIN}
    with TrainingRun("fuse", "tabpfn_d", config=cfg, data=["fuse_tier_d"], plan_ref="fuse_protocol §6 TabPFN",
                     iteration=args.run_name, artifacts_dir=str(out)) as trn:
        for t in targets:
            Y, M, C = fd.tier_d_target_matrix(df, [t])
            y = Y[:, 0]
            pcol = s0.PERSISTENCE_D.get(t.name)
            anc = df[pcol].to_numpy(float) if pcol in df.columns else np.full(len(df), np.nan)
            parts = []
            for k in sorted(np.unique(fold)):
                trm = (fold != k) & np.isfinite(y)
                evm = (fold == k) & np.isfinite(y)
                pre = fd.FoldPreprocessor(specs)
                pre.fit_micro(df, trm, genus)
                X = fd.tier_d_feature_frame(df, specs, pre, genus).to_numpy(np.float32)
                X = X[:, :500]
                Q = np.full((len(df), len(TAUS)), np.nan)
                for anchored in (True, False):
                    tr_rows = np.flatnonzero(trm & (np.isfinite(anc) if anchored else np.ones(len(df), bool)))
                    ev_rows = np.flatnonzero(evm & (np.isfinite(anc) if anchored else ~np.isfinite(anc)))
                    if len(ev_rows) == 0 or len(tr_rows) < 200:
                        continue
                    sub = tr_rows[site_quota_sample(sites[tr_rows], args.max_rows, seed=k)]
                    ytr = y[sub] - (anc[sub] if anchored else 0.0)
                    q = fit_predict(Reg, X[sub], ytr, X[ev_rows], args.device, seed=k)
                    Q[ev_rows] = q + (anc[ev_rows, None] if anchored else 0.0)
                ev = np.flatnonzero(evm)
                Qe = np.sort(Q[ev], axis=1)
                yv = np.where(C[ev, 0], np.nan, y[ev])
                crps = np.where(np.isfinite(yv), quantile_crps(np.nan_to_num(yv), Qe), np.nan)
                parts.append(pd.DataFrame({"row": ev, "config": "tabpfn@a", "mu": Qe[:, 49], "sigma": np.nan,
                                           "crps": crps, "fold": k}))
                print(f"{t.name} fold {k}: n_eval {len(ev)} mean CRPS {np.nanmean(crps):.4f}", flush=True)
            P = pd.concat(parts, ignore_index=True)
            P["site"] = sites[P["row"]]
            P["date"] = df["date"].to_numpy()[P["row"]]
            P["y"] = y[P["row"]]
            P.to_parquet(out / f"oof_{t.name}.parquet", index=False)
            ok = P["crps"].notna().to_numpy()
            r = fst.cluster_ci(np.mean, P["crps"].to_numpy()[ok], groups=P["site"].to_numpy()[ok], n_boot=1000)
            log_result(workstream="fuse", experiment="fuse_stage0_d", model="tabpfn@a", metric="crps", value=r["point"],
                       split="val_cv_lodgo5", n=int(ok.sum()), ci_low=r["ci_low"], ci_high=r["ci_high"],
                       ci_method="site-cluster bootstrap (B=1000) x sqrt(G/(G-1))", target=t.name, is_baseline=True,
                       config=cfg, data_manifests=["fuse_tier_d"], artifacts=[str(out / f"oof_{t.name}.parquet")],
                       notes="quantile CRPS (99 levels); all OOF rows with the target observed (not common rows)")
            trn.metric(f"crps_{t.name}", r["point"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
