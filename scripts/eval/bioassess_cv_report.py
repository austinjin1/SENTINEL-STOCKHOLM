#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Phase-1 CV report and model selection for SENTINEL-BioAssess (development data only).

Reads the current registry entries of experiment 'bioassess_phase1_cv' and their out-of-fold
prediction files, and

1. prints the comparison table (pooled R², anomaly R², median per-ecoregion R², per scheme) with
   registry ids;
2. computes paired cluster-bootstrap differences (unit = location cluster, B = --n-boot) of every
   BioSet candidate vs ``RandomForest[community]`` on the same rows, for pooled and anomaly R²;
3. applies the pre-registered selection rule (highest mean pooled LOEO R² over the 6 targets;
   ties within 0.005 -> no pretraining, then fewer parameters) and reports the number of
   configurations tried.

With ``--log`` the paired differences are written to the registry (metric 'delta_r2_vs_rf_community'
/ 'delta_r2_anomaly_vs_rf_community', experiment 'bioassess_phase1_selection') and the selection to
``<runs>/bioassess/phase1_selection.json``. These are CV statistics on development data — never test
scores.

    python -B scripts/eval/bioassess_cv_report.py [--log] [--n-boot 2000]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import log_result, registry_frame  # noqa: E402

EXPERIMENT = "bioassess_phase1_cv"
REF = "RandomForest[community]"
GBM_REF = "LightGBM[community]"
GEO_REF = "RandomForest[community+geography]"
TAB_REF = "TabPFNv2[community]"
ENS_REF = "Ensemble[LightGBM[community]+TabPFNv2[community]]"
ENS_GEO_REF = "Ensemble[RandomForest[community+geography]+TabPFNv2[community]]"
ENS3_REF = "Ensemble[RandomForest[community+geography]+LightGBM[community]+TabPFNv2[community]]"
CTX_REF = "RandomForest[community+context]"
REF_SLUG = {REF: "rf_community", GBM_REF: "gbm_community", GEO_REF: "rf_community_geography",
            TAB_REF: "tabpfn_community", ENS_REF: "ens_gbm_tabpfn", ENS_GEO_REF: "ens_rfgeo_tabpfn",
            ENS3_REF: "ens_rfgeo_gbm_tabpfn", CTX_REF: "rf_community_context"}
TARGETS = ("log10_tp_ug_l", "log10_tn_mg_l", "log10_cond_us_cm", "log10_chloride_mg_l", "log10_turbidity_ntu",
           "ph_lab")
SHORT = {"log10_tp_ug_l": "TP", "log10_tn_mg_l": "TN", "log10_cond_us_cm": "cond", "log10_chloride_mg_l": "Cl",
         "log10_turbidity_ntu": "turb", "ph_lab": "pH"}


def current_entries() -> pd.DataFrame:
    df = registry_frame()
    return df[(df["workstream"] == "bioassess") & (df["experiment"] == EXPERIMENT)].copy()


def oof_path(entry) -> str:
    return next(a for a in entry["artifacts"] if "/oof_" in a)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=6, help="<= 6 CPU threads per job (CONTRIBUTING_V2)")
    ap.add_argument("--schemes", nargs="+", default=["nrsa0814/loeo_cv", "nrsa0814/block50_cv"])
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict paired comparisons to these models (e.g. new improvement-cycle candidates)")
    a = ap.parse_args()
    a.n_jobs = max(1, min(a.n_jobs, 6))
    from sentinel.models.bioassess import evaluate as E

    df = current_entries()
    if df.empty:
        print("no registry entries yet")
        return 1
    # latest entry per (model, split, target, metric)
    df = df.sort_values("timestamp").groupby(["model", "split", "target", "metric"], as_index=False).last()
    wide = df.pivot_table(index=["split", "model"], columns=["metric", "target"], values="value")
    lines = []
    for split in sorted(df["split"].unique()):
        lines.append(f"\n### {split}\n")
        hdr = "| model | " + " | ".join(f"{SHORT[t]} R² / anom / reg-med" for t in TARGETS) + " | mean R² | mean anom |"
        lines += [hdr, "|" + "---|" * (len(TARGETS) + 3)]
        sub = wide.loc[split]
        sub = sub.loc[sub["r2"][list(TARGETS)].mean(1).sort_values(ascending=False).index]
        for model, r in sub.iterrows():
            cells = [f"{r[('r2', t)]:.3f} / {r[('r2_anomaly', t)]:.3f} / {r[('r2_region_median', t)]:.3f}"
                     for t in TARGETS]
            m_r2 = np.mean([r[("r2", t)] for t in TARGETS])
            m_an = np.mean([r[("r2_anomaly", t)] for t in TARGETS])
            lines.append(f"| {model} | " + " | ".join(cells) + f" | {m_r2:.3f} | {m_an:.3f} |")
    print("\n".join(lines))

    sel_rows = []
    out = {"comparisons": [], "selection": None}
    for split in a.schemes:
        d = df[df["split"] == split]
        models = set(d["model"])
        oofs: dict = {}

        def get(m):
            if m not in oofs:
                e = d[(d["model"] == m) & (d["metric"] == "r2")].iloc[0]
                oofs[m] = (e, pd.read_parquet(oof_path(e)))
            return oofs[m]

        pairs = []
        for model in sorted(models):
            if model.startswith("Ensemble[BioSet["):
                pairs += [(model, REF), (model, TAB_REF), (model, ENS_REF), (model, ENS_GEO_REF), (model, ENS3_REF)]
            elif model.startswith("BioSet[") and model.endswith("+context"):
                pairs.append((model, CTX_REF))              # like-for-like inputs
            elif model.startswith("BioSet[") and model.endswith("+geography"):
                pairs.append((model, GEO_REF))          # like-for-like inputs
            elif model.startswith("BioSet["):
                pairs += [(model, REF), (model, GBM_REF), (model, TAB_REF)]
            elif model in (GBM_REF, TAB_REF):
                pairs.append((model, REF))
        jobs = []
        if a.only:
            pairs = [pr for pr in pairs if pr[0] in set(a.only)]
        for model, refm in pairs:
            if refm not in models:
                continue
            e, o = get(model)
            ref_e, ref = get(refm)
            if not (o["visit_id"].to_numpy() == ref["visit_id"].to_numpy()).all():
                raise AssertionError("OOF rows differ between model and reference")
            for t in TARGETS:
                for metric in ("r2", "r2_anomaly"):
                    jobs.append((model, refm, t, metric, o[f"obs|{t}"].to_numpy(), o[f"pred|{t}"].to_numpy(),
                                 ref[f"pred|{t}"].to_numpy(), o["ag_eco9"].to_numpy(), o["location_id"].to_numpy()))
        from joblib import Parallel, delayed

        res = Parallel(n_jobs=a.n_jobs)(delayed(E.paired_vs)(y, pm, pr, rg, cl, metric=mt, n_boot=a.n_boot)
                                        for (_, _, _, mt, y, pm, pr, rg, cl) in jobs)
        for (model, refm, t, metric, y, *_), pv in zip(jobs, res):
            e, o = get(model)
            ref_e, _ = get(refm)
            rec = {"split": split, "model": model, "reference": refm, "target": t, "metric": metric, **pv}
            if a.log and (model.startswith("BioSet[") or model.startswith("Ensemble[BioSet[")):
                en = log_result(workstream="bioassess", experiment="bioassess_phase1_selection", model=model,
                                metric=f"delta_{metric}_vs_{REF_SLUG[refm]}", value=pv["diff"], split=split,
                                n=int(np.isfinite(y).sum()), ci_low=pv["ci_low"], ci_high=pv["ci_high"],
                                ci_method=f"paired percentile cluster bootstrap, unit = location cluster, B={a.n_boot}",
                                target=t, is_baseline=False, data_manifests=["nars_streams"],
                                artifacts=[oof_path(e), oof_path(ref_e)],
                                notes=f"Phase-1 CV (development data): {model} minus {refm}, same OOF rows.",
                                extra={"p_value": pv["p_value"], "reference_model": refm,
                                       "model_entry": e["id"], "reference_entry": ref_e["id"]})
                rec["registry_id"] = en["id"]
            out["comparisons"].append(rec)
        if split == "nrsa0814/loeo_cv":
            # selection candidates: community-only BioSet models (no '+geography' / '+context' suffix)
            cands = d[(d["metric"] == "r2") & d["model"].str.startswith("BioSet[") & d["model"].str.endswith("]")]
            for model, g in cands.groupby("model"):
                cfg = g.iloc[0]["config"] or {}
                oofp = Path(oof_path(g.iloc[0]))
                diag = oofp.with_name(oofp.name.replace("oof_", "diag_").replace(".parquet", ".csv"))
                n_params = int(pd.read_csv(diag)["n_params"].median()) if diag.exists() else None
                sel_rows.append({"model": model, "mean_r2": float(g.set_index("target")["value"].reindex(TARGETS).mean()),
                                 "pretrained": bool((cfg.get("bioset") or {}).get("pretrained")),
                                 "n_params": n_params})
    if sel_rows:
        s = pd.DataFrame(sel_rows).sort_values("mean_r2", ascending=False).reset_index(drop=True)
        best = s.iloc[0]["mean_r2"]
        # tie-break: no pretraining, then fewer parameters, then the simpler training objective
        # (no auxiliary outputs / extra loss terms; 'simpler' added 2026-10-24 when two candidates
        # tied with identical parameter counts), then higher mean R²
        s["extra_objective"] = s["model"].str.contains("_aux|_rloss")
        tied = s[s["mean_r2"] >= best - 0.005].sort_values(["pretrained", "n_params", "extra_objective", "mean_r2"],
                                                           ascending=[True, True, True, False])
        chosen = tied.iloc[0]["model"]
        out["selection"] = {"rule": ("highest mean pooled LOEO R2 over 6 targets; ties within 0.005 -> no pretraining, then fewer "
                                     "parameters, then simpler training objective"),
                            "n_configurations_tried": int(len(s)), "candidates": s.to_dict("records"),
                            "selected": chosen}
        print("\nSelection (LOEO, development data):")
        print(s.to_string(index=False))
        print(f"selected: {chosen}  (configurations tried: {len(s)})")
    comp = pd.DataFrame(out["comparisons"])
    if not comp.empty:
        show = comp[comp["model"].str.startswith("BioSet[") | comp["model"].str.startswith("Ensemble[BioSet[") | comp["model"].isin(["LightGBM[community]",
                                                                                    "TabPFNv2[community]"])]
        print("\nPaired differences, model minus reference (diff [95% CI]):")
        for (split, model, refm, metric), g in show.groupby(["split", "model", "reference", "metric"]):
            cells = ", ".join(f"{SHORT[r.target]} {r.diff:+.3f} [{r.ci_low:+.3f},{r.ci_high:+.3f}]"
                              for r in g.itertuples())
            print(f"{split} {model} - {refm} {metric}: {cells}")
    dest = paths.RUNS_ROOT / "bioassess" / "phase1_selection.json"
    if a.log:
        dest.write_text(json.dumps(out, indent=1, default=float))
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
