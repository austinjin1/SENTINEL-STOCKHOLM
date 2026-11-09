#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Plan §1.3c (W1) and §4.5 (W2) post-pass analyses, exactly as pre-registered before the pass.

Uses the primary pass's saved predictions and its protected report; fits nothing. Writes
``prospective_posthoc.json`` next to them and logs the W2 intervals to the registry.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sentinel.core.registry import log_result  # noqa: E402
from sentinel.eval.bootstrap import bootstrap_ci, holm_bonferroni, paired_bootstrap  # noqa: E402

RUN = Path("/home/bcheng/STOCKHOLM/sentinel_runs/chemtox/prospective_test")
DATA = Path("/home/bcheng/STOCKHOLM/sentinel_data/processed/chemtox_prospective/prospective.parquet")
SIDE = Path(__file__).resolve().parents[2] / "docs/preregistration/chemtox_adore_prefilter_seen.json"
TRACKS = ("fish_novel", "fish_novel_strict")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def main() -> int:
    rep = json.loads((RUN / "prospective_results_primary.json").read_text())
    side = json.loads(SIDE.read_text())
    spec = __import__("importlib.util").util.spec_from_file_location(
        "sc", Path(__file__).resolve().parent / "chemtox_score_prospective.py")
    sc = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    pro = pd.read_parquet(DATA)
    y_all = pd.to_numeric(pro["result_conc1_mean_mol_log"], errors="coerce").to_numpy(float)
    cas_all = pro["test_cas"].astype(str).to_numpy()
    par_all = pro["parent_key"].astype(str).to_numpy()
    rid_all = pd.to_numeric(pro["result_id"], errors="coerce").astype("Int64")
    preds = {f.stem[5:]: np.load(f).mean(axis=0) for f in RUN.glob("pred_*.npy")
             if not f.stem.endswith("_trainpart")}
    cands = rep["candidate_models"]
    out = {"source_report": "prospective_results_primary.json", "tracks": {}}
    for t in TRACKS:
        T = rep["tracks"][t]
        m = sc.track_mask(pro, t)
        y, cas, par = y_all[m], cas_all[m], par_all[m]
        refs = {"published_reference": T["published_reference"]["model"],
                "best_baseline": T["best_baseline"]["model"]}
        tr = {"references": refs, "w2_parent_clustered": {}, "w1_prefilter_breakdown": {}}
        for rname, ref in refs.items():
            res, pv = {}, {}
            for c in cands:
                d = paired_bootstrap(rmse, y, preds[c][m], preds[ref][m], groups=par, n_boot=2000,
                                     seed=0, higher_is_better=False)
                cas_d = T["models"][c][f"vs_{rname}"]
                d["cas_ci"] = [cas_d["ci_low"], cas_d["ci_high"]]
                res[c] = d
                if np.isfinite(d["p_value"]):
                    pv[c] = d["p_value"]
            rej = holm_bonferroni(pv)
            for c in res:
                res[c]["holm_rejected_parent"] = bool(rej.get(c, False)) and res[c]["diff"] < 0
                cas_claim = bool(T.get(f"holm_rejected_vs_{rname}", {}).get(c)) and \
                    T["models"][c][f"vs_{rname}"]["diff"] < 0
                res[c]["qualified_not_robust_to_parent_clustering"] = bool(
                    cas_claim and res[c]["ci_low"] <= 0 <= res[c]["ci_high"])
                log_result(workstream="chemtox", experiment="chemtox_prospective_test_posthoc",
                           model=c, metric=f"rmse_diff_vs_{rname}_parent_clustered",
                           value=res[c]["diff"], ci_low=res[c]["ci_low"],
                           ci_high=res[c]["ci_high"],
                           ci_method="paired cluster bootstrap by parent_key, 2000, seed 0",
                           split=f"prospective/{t}/test", n=int(m.sum()), is_baseline=False,
                           notes="plan §4.5 (W2) pre-registered sensitivity; primary CI is CAS",
                           extra={"reference_model": ref, "p_value": res[c]["p_value"],
                                  "holm_rejected_parent": res[c]["holm_rejected_parent"]})
            tr["w2_parent_clustered"][rname] = res
        seen_ids = set(side["tracks"][t]["seen_result_ids"])
        seen = rid_all[m].isin(seen_ids).fillna(False).to_numpy()
        for name, p in preds.items():
            cells = {}
            for lab, mm in (("seen", seen), ("unseen", ~seen)):
                if mm.sum() >= 10:
                    ci = bootstrap_ci(rmse, y[mm], p[m][mm], groups=cas[mm], n_boot=2000, seed=0)
                    cells[lab] = {"n_rows": int(mm.sum()), "n_chemicals": int(len(set(cas[mm]))),
                                  "rmse": rmse(y[mm], p[m][mm]), "ci": ci}
            tr["w1_prefilter_breakdown"][name] = cells
        out["tracks"][t] = tr
    (RUN / "prospective_posthoc.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", RUN / "prospective_posthoc.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
