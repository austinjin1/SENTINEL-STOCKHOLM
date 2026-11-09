# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Paired development-CV comparison of SENTINEL-Fuse against the Stage-0 comparators (validation only).

    python -B scripts/eval/fuse_compare.py --fuse-run fuse_d_v1 --stage0-run stage0_d_p1_v3

For each target: CRPS of the seed-mixture Fuse prediction and of every Stage-0 configuration on the
identical rows (rows every model predicts, target observed and uncensored). Pre-registered comparators
(protocol §6/§8): (a) best single modality = the ``single:*`` configuration (level or ``@a``
persistence-anchored) with the lowest dev-CV CRPS; (b) GBM concat = the better of ``concat`` /
``concat@a`` on dev CV; plus stacking, UME, hub and persistence. Skill CIs: site-cluster bootstrap with
the small-cluster factor sqrt(G/(G−1)) (§10), plus the leave-one-domain-out range. Also: a-priori hard
strata (§8.1) and inference-time LOMO ΔCRPS_m / CRPS_hub (P3 preview, §8). Rows are joined by position
in the same ``drop_protected`` frame; (site, date) equality is asserted (review W7).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.models.fuse import baselines as bl
from sentinel.models.fuse import data as fd
from sentinel.models.fuse import splits as sp
from sentinel.models.fuse import stage0 as s0
from sentinel.models.fuse import stats as fst

SPLIT_SHA_D = "20adbe595ed42a46c490da31e16743353ce819e9d84cb8eedffa15303a26610a"


def mix(p: pd.DataFrame) -> pd.DataFrame:
    """Moment-matched Gaussian of the equal-weight seed mixture per row."""
    g = p.assign(m2=p["sigma"] ** 2 + p["mu"] ** 2).groupby("row")
    mu = g["mu"].mean()
    m2 = g["m2"].mean()
    return pd.DataFrame({"mu": mu, "sigma": np.sqrt(np.clip(m2 - mu ** 2, 1e-12, None))})


def skill_stats(a, b, sites, domains, n_boot=1000) -> dict:
    r = fst.cluster_ci(fst.skill, a, b, groups=sites, n_boot=n_boot)
    j = fst.leave_one_cluster_out(fst.skill, a, b, clusters=domains)
    return {"skill": r["point"], "ci": [r["ci_low"], r["ci_high"]], "ci_raw": [r["ci_low_raw"], r["ci_high_raw"]],
            "n_sites": r["n_groups"], "lodo_min": j["min"], "lodo_max": j["max"], "n": int(len(a))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuse-run", required=True)
    ap.add_argument("--stage0-run", required=True, help="one or more comma-separated Stage-0 runs")
    ap.add_argument("--targets", default=",".join(fd.P1_TARGETS))
    ap.add_argument("--label", default=None, help="registry model name (default fuse:<run>)")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    fruns = [paths.RUNS_ROOT / "fuse" / r for r in args.fuse_run.split(",")]   # one run per seed (overflow jobs)
    fr = fruns[0]
    sruns = [paths.RUNS_ROOT / "fuse" / r for r in args.stage0_run.split(",")]
    F = pd.concat([pd.read_parquet(r / "oof_fuse.parquet") for r in fruns], ignore_index=True)
    assert not F.duplicated(["row", "config", "target", "seed"]).any(), "same seed in two fuse runs"
    split = sp.load_split(sp.TIER_D_SPLIT, SPLIT_SHA_D)
    mods = list(fd.ftd.NON_HUB_D)
    df = pd.read_parquet(paths.processed("fuse_tier_d") / "daily.parquet",
                         columns=["site", "date", "domain", "wx__precip_sum3"] + [f"{m}__present" for m in mods]
                         + sorted(set(s0.PERSISTENCE_D.values()))
                         + [f"{t}__cens" for t in args.targets.split(",") if t.startswith("ylab__")])
    df = s0.drop_protected(df, split["site_role"], "site")   # same row order as Stage 0 / Fuse OOF
    nnh = fst.n_nonhub_present(df, mods)
    label = args.label or f"fuse:{args.fuse_run}"
    out = {}
    p3_parts: dict[str, list] = {}
    for t in args.targets.split(","):
        src = [r for r in sruns if (r / f"oof_{t}.parquet").exists()]
        if not src:
            print(f"{t}: no Stage-0 OOF found; skipped")
            continue
        S = pd.concat([pd.read_parquet(r / f"oof_{t}.parquet") for r in src], ignore_index=True)
        S = S.drop_duplicates(["row", "config"])     # first run wins for a config present twice
        S1 = S.drop_duplicates("row").set_index("row").sort_index()
        # W7: positional join must agree on (site, date) for Stage 0, Fuse and the frame
        rows_s = S1.index.to_numpy()
        assert (S1["site"].to_numpy() == df["site"].to_numpy()[rows_s]).all(), "Stage-0 rows misaligned (site)"
        assert (pd.to_datetime(S1["date"]).to_numpy() == pd.to_datetime(df["date"]).to_numpy()[rows_s]).all(), \
            "Stage-0 rows misaligned (date)"
        fz = F[F.target == t]
        f1 = fz.drop_duplicates("row").set_index("row")
        assert (f1["site"].to_numpy() == df["site"].to_numpy()[f1.index.to_numpy()]).all(), "Fuse rows misaligned"
        assert (pd.to_datetime(f1["date"]).to_numpy() == pd.to_datetime(df["date"]).to_numpy()[f1.index.to_numpy()]).all()
        y = S1["y"]
        fm = mix(fz[fz.config == "fuse"])
        cfgs = [c for c in S.config.unique() if c not in ("persistence", "pattern")]
        common = set(fm.index)
        for c in cfgs:
            common &= set(S.loc[(S.config == c) & S.mu.notna(), "row"])
        common = np.array(sorted(r for r in common if np.isfinite(y.get(r, np.nan))))
        if t.startswith("ylab__"):    # censored values are bounds: trained on, never scored
            common = common[~df[f"{t}__cens"].fillna(False).to_numpy(bool)[common]]
        sites = df["site"].to_numpy()[common]
        doms = df["domain"].to_numpy()[common]
        yy = y.loc[common].to_numpy()
        cr = {"fuse": bl.crps_gauss_rows(yy, fm.loc[common, "mu"].to_numpy(), fm.loc[common, "sigma"].to_numpy())}
        SM = {c: S[S.config == c].set_index("row") for c in cfgs + ["persistence"] if (S.config == c).any()}
        for c in cfgs:
            if "crps" in SM[c].columns and SM[c]["crps"].notna().any():    # non-Gaussian comparator (TabPFN quantiles)
                cr[c] = SM[c].loc[common, "crps"].to_numpy(float)
            else:
                cr[c] = bl.crps_gauss_rows(yy, SM[c].loc[common, "mu"].to_numpy(), SM[c].loc[common, "sigma"].to_numpy())
        singles = {c: cr[c].mean() for c in cr if c.startswith("single:")}
        best = min(singles, key=singles.get)
        concats = {c: cr[c].mean() for c in ("concat", "concat@a") if c in cr}
        best_concat = min(concats, key=concats.get)
        res = {"n": int(len(common)), "n_sites": int(len(np.unique(sites))), "best_single": best,
               "best_concat": best_concat, "crps": {c: float(v.mean()) for c, v in cr.items()}, "skill": {}}
        refs = {"best_single": best, "gbm_concat": best_concat, "stack": "stack", "ume": "ume", "hub": "hub",
                "tabpfn": "tabpfn@a"}
        for key, ref in refs.items():
            if ref not in cr:
                continue
            st = skill_stats(cr["fuse"], cr[ref], sites, doms)
            res["skill"][key] = st | {"ref": ref}
            if not args.no_log:
                log_result(workstream="fuse", experiment="fuse_compare_d", model=label, metric=f"crps_skill_vs_{key}",
                           value=st["skill"], split="val_cv_lodgo5", n=st["n"], ci_low=st["ci"][0], ci_high=st["ci"][1],
                           ci_method="site-cluster bootstrap (B=1000) x sqrt(G/(G-1)), paired identical rows",
                           target=t, config={"fuse_run": args.fuse_run, "stage0_run": str(src[0].name), "ref": ref},
                           data_manifests=["fuse_tier_d"], artifacts=[str(fr / "oof_fuse.parquet")],
                           extra={"lodo_min": st["lodo_min"], "lodo_max": st["lodo_max"]})
        # persistence on its own rows
        pcol = s0.PERSISTENCE_D.get(t)
        if "persistence" in SM:
            ps = SM["persistence"]
            pr = np.array([r for r in common if np.isfinite(ps["mu"].get(r, np.nan))])
            if len(pr):
                a = bl.crps_gauss_rows(y.loc[pr].to_numpy(), fm.loc[pr, "mu"].to_numpy(), fm.loc[pr, "sigma"].to_numpy())
                b = bl.crps_gauss_rows(y.loc[pr].to_numpy(), ps.loc[pr, "mu"].to_numpy(), ps.loc[pr, "sigma"].to_numpy())
                res["skill"]["persistence"] = skill_stats(a, b, df["site"].to_numpy()[pr], df["domain"].to_numpy()[pr])
        # a-priori hard strata (§8.1): thresholds from the training folds of each evaluation row's fold
        if pcol is not None:
            fam = t.split("__")[1] if t.startswith("y__") else t
            owner = s0.anchor_owner(pcol)
            anc_all = df[pcol].to_numpy(float)
            fold_of = S1["fold"]
            strata_rows: dict[str, list] = {}
            for k in np.unique(fold_of.loc[common]):
                ev = common[fold_of.loc[common].to_numpy() == k]
                trr = S1.index.to_numpy()[fold_of.to_numpy() != k]
                ytr = S1.loc[trr, "y"].to_numpy()
                m = fst.hard_strata(S1.loc[ev, "y"].to_numpy(), anc_all[ev], ytr, anc_all[trr], fam,
                                    df[f"{owner}__present"].to_numpy(bool)[ev], nnh[ev],
                                    df["wx__precip_sum3"].to_numpy(float)[ev], df["wx__precip_sum3"].to_numpy(float)[trr])
                for name, mk in m.items():
                    strata_rows.setdefault(name, []).extend(ev[mk].tolist())
            pos = pd.Series(np.arange(len(common)), index=common)
            res["strata"] = {}
            for name, rr in strata_rows.items():
                idx = pos.loc[sorted(rr)].to_numpy()
                if len(idx) < 50 or len(np.unique(sites[idx])) < 3:
                    res["strata"][name] = {"n": int(len(idx)), "note": "too few rows/sites"}
                    continue
                res["strata"][name] = {"n": int(len(idx)),
                                       "vs_best_single": skill_stats(cr["fuse"][idx], cr[best][idx], sites[idx], doms[idx]),
                                       "vs_gbm_concat": skill_stats(cr["fuse"][idx], cr[best_concat][idx], sites[idx], doms[idx])}
        # LOMO (inference time; §8 P3): ΔCRPS_m / CRPS_hub on rows where m is present
        lomo = {}
        for cfg in sorted(c for c in fz.config.unique() if c.startswith("fuse_minus_")):
            m = cfg[len("fuse_minus_"):]
            lm = mix(fz[fz.config == cfg])
            sel = df[f"{m}__present"].to_numpy(bool)[common] & np.isin(common, lm.index)
            if sel.sum() < 50:
                continue
            rows = common[sel]
            c_wo = bl.crps_gauss_rows(yy[sel], lm.loc[rows, "mu"].to_numpy(), lm.loc[rows, "sigma"].to_numpy())
            f = lambda a1, a2, h: (np.mean(a2) - np.mean(a1)) / np.mean(h)
            r = fst.cluster_ci(f, cr["fuse"][sel], c_wo, cr["hub"][sel], groups=sites[sel], n_boot=1000)
            p3_parts.setdefault(m, []).append((sites[sel], cr["fuse"][sel], c_wo, cr["hub"][sel]))
            lomo[m] = {"rel_dcrps_hub": r["point"], "ci": [r["ci_low"], r["ci_high"]], "n": int(sel.sum()),
                       "n_sites": r["n_groups"]}
        res["lomo"] = lomo
        out[t] = res
        print(t, json.dumps({k: res[k] for k in ("n", "n_sites", "best_single", "best_concat")}),
              {k: round(v, 4) for k, v in res["crps"].items()},
              {k: [round(v["skill"], 3)] + [round(x, 3) for x in v["ci"]] for k, v in res["skill"].items()},
              {k: [round(v["rel_dcrps_hub"], 4)] + [round(x, 4) for x in v["ci"]] for k, v in lomo.items()}, flush=True)
    # P3 (protocol §8): pooled over the targets of this call (P1 + lab when both runs are given)
    p3 = {}
    for m, parts in p3_parts.items():
        r = fst.p3_statistic(parts)
        n_rows = int(sum(len(p[0]) for p in parts))
        p3[m] = r | {"n_rows": n_rows, "positive_ci": bool(r["ci_low"] > 0)}
        if not args.no_log:
            log_result(workstream="fuse", experiment="fuse_compare_d", model=label, metric=f"p3_lomo_rel_dcrps_hub:{m}",
                       value=r["point"], split="val_cv_lodgo5", n=n_rows, ci_low=r["ci_low"], ci_high=r["ci_high"],
                       ci_method="site-cluster bootstrap (B=1000, joint over targets) x sqrt(G/(G-1))",
                       target="+".join(k for k in out), config={"fuse_run": args.fuse_run, "stage0_run": args.stage0_run},
                       data_manifests=["fuse_tier_d"], notes="inference-time LOMO, dev CV (selection-affected)")
    out["_p3"] = p3
    print("P3", {m: [round(v["point"], 4), round(v["ci_low"], 4), round(v["ci_high"], 4)] for m, v in p3.items()},
          "n_positive", sum(v["positive_ci"] for v in p3.values()), flush=True)
    (fr / f"compare_{args.stage0_run.replace(',', '+')}.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
