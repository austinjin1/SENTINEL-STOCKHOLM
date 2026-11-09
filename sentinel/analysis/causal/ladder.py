# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Summaries for the causal claim ladder steps 2–3 (+ evidence cards) on real data (protocol §7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import controls as C
from .prep import RIVER, WEATHER


def _load(p: Path):
    z = np.load(p, allow_pickle=False)
    return z["graphs"], z["vals"].astype("float32"), [str(n) for n in z["names"]]


def step2(out: Path, d=None) -> dict:
    pos, neg, xpos, xneg, asym = [], [], [], [], []
    rc = json.loads((out / "run_config.json").read_text()) if (out / "run_config.json").exists() else {}
    x_alpha = rc.get("fdr_q", 0.01)
    for f in sorted((out / "controls").glob("*.json")):
        r = json.loads(f.read_text())
        s = r["site"]
        g, v, names = _load(out / "controls" / f"{s}_primary.npz")
        for row in C.positive_controls(g[1:], v[1:], names, dilution=r["dilution"]):
            pos.append({"site": s, **row})
        gu, vu, names = _load(out / "controls" / f"{s}_unconstrained.npz")
        neg += [{"site": s, "class": "N1", **x} for x in C.negative_n1(gu[1:], vu[1:], names)]
        # N2 (review A2): reported as a forward-vs-reversed asymmetry diagnostic, NOT pooled into the FPR
        g, v, names_r = _load(out / "controls" / f"{s}_reversed.npz")
        asym += [{"site": s, **x} for x in C.reversal_asymmetry(gu[1:], vu[1:], g[1:], v[1:], names)]
        g, v, names = _load(out / "controls" / f"{s}_augmented.npz")
        far = [n for n in names if n.startswith("far_")]
        neg += [{"site": s, "class": "N3", **x} for x in
                C.negative_lagged_into(g[1:], v[1:], names, far, [n for n in names if n in RIVER], "N3", lags=(0, 7))]
        if d is not None:   # cross-correlation baseline on the same matrices
            from .discovery import xcorr_baseline
            from .prep import site_matrix
            from ...models.sim import data as D

            i = d.sites.index(s)
            m = site_matrix(d, i, D.block_mask(d.dates, "train"))
            gx, rx = xcorr_baseline(m.to_numpy(), list(m.columns), alpha=x_alpha)
            for row in C.positive_controls(gx[None], rx[None], list(m.columns), dilution=r["dilution"]):
                xpos.append({"site": s, **row})
            xneg += [{"site": s, "class": "N1", **x} for x in C.negative_n1(gx[None], rx[None], list(m.columns))]
    P, N, A = pd.DataFrame(pos), pd.DataFrame(neg), pd.DataFrame(asym)
    rec = P.groupby("control")["detected"].agg(["mean", "size"]).rename(columns={"mean": "recall", "size": "n_sites"})
    fpr = N.groupby("class")["detected"].agg(["mean", "size"]).rename(columns={"mean": "fpr", "size": "n_tests"})
    out_d = {"recall_by_control": rec.reset_index().to_dict("records"),
             "mean_recall": float(rec["recall"].mean()), "min_recall": float(rec["recall"].min()),
             "fpr_by_class": fpr.reset_index().to_dict("records"), "fpr_pooled": float(N["detected"].mean()),
             "n_sites": int(P["site"].nunique()),
             "N2_reversal_asymmetry": ({"n_pairs": int(len(A)), "mean_rev_minus_fwd": float(A["rev_minus_fwd"].mean()),
                                        "share_rev_ge_fwd": float((A["rev_minus_fwd"] >= 0).mean()),
                                        "share_rev_ge_fwd_among_fwd_detected": float(
                                            (A.loc[A["fwd_detected"], "rev_minus_fwd"] >= 0).mean())
                                        if A["fwd_detected"].any() else None,
                                        "note": "diagnostic only (review A2): not part of the pooled FPR"}
                                       if len(A) else None)}
    out_d["pass"] = bool(out_d["mean_recall"] >= 0.8 and out_d["min_recall"] >= 0.6 and out_d["fpr_pooled"] <= 0.05)
    if xpos:
        XP, XN = pd.DataFrame(xpos), pd.DataFrame(xneg)
        out_d["xcorr_baseline"] = {"recall_by_control": XP.groupby("control")["detected"].mean().to_dict(),
                                   "mean_recall": float(XP.groupby("control")["detected"].mean().mean()),
                                   "fpr_N1": float(XN["detected"].mean())}
    P.to_csv(out / "step2_positive.csv", index=False)
    N.to_csv(out / "step2_negative.csv", index=False)
    if len(A):
        A.to_csv(out / "step2_reversal_asymmetry.csv", index=False)
    return out_d


def step3(out: Path, d, huc2: pd.Series) -> pd.DataFrame:
    tr, va = [], []
    for f in sorted((out / "discovery").glob("*.npz")):
        s = f.stem
        g, v, names = _load(f)
        L = C.site_links(g[1:], v[1:], names)
        L["site"] = s
        L["huc2"] = huc2.get(s, "unk")
        role = d.role[d.sites.index(s)]
        (va if role == "val" else tr).append(L)
    T = pd.concat(tr) if tr else pd.DataFrame()
    V = pd.concat(va) if va else None
    T.to_csv(out / "step3_links_train.csv", index=False)
    if V is not None:
        V.to_csv(out / "step3_links_val.csv", index=False)
    return C.replication(T, val_links=V)


# Step-1 candidate configurations (protocol §7 step 1, declared 2026-11-08) and the selection rule.
STEP1_CANDIDATES = ("v2_a0.01_q0.01", "v2_a0.05_q0.05", "v2_a0.05_q0.01")


def step1_selection(syn: Path, candidates=STEP1_CANDIDATES) -> dict:
    """Winner on the DEV sites: highest Bagged-PCMCI+ recall among candidates with FPR ≤ 0.05, else the
    lowest FPR. Every candidate must have a complete dev result (20 sites, 50 bootstraps)."""
    res = {}
    for t in candidates:
        f = syn / t / "dev" / "step1_results.json"
        if not f.exists():
            raise SystemExit(f"dev result missing for candidate {t}")
        s = json.loads(f.read_text())["summary"]
        b = s["bagged_pcmciplus"]
        if b["n_sites"] < 20 or s["config"]["boot"] != 50:
            raise SystemExit(f"candidate {t}: incomplete dev run ({b['n_sites']} sites, boot {s['config']['boot']})")
        res[t] = (b["recall"], b["fpr"])
    ok = {t: v for t, v in res.items() if v[1] <= 0.05}
    win = max(ok, key=lambda t: ok[t][0]) if ok else min(res, key=lambda t: res[t][1])
    return {"winner": win, "dev": res}


def diel_summary(results: list[dict]) -> dict:
    """Diel controls (protocol §7): per-control detection rate over sites; same thresholds as daily
    (mean positive recall ≥ 0.8, each ≥ 0.6; pooled negative FPR ≤ 0.05) — reported, not a gate."""
    pos = pd.DataFrame([{"site": r["site"], **x} for r in results for x in r["positive"]])
    neg = pd.DataFrame([{"site": r["site"], **x} for r in results for x in r["negative"]])
    if not len(pos):
        return {"n_sites": 0, "positive": [], "negative": []}
    P = [{"kind": "positive", "control": c, "rate": float(g["detected"].mean()), "n_sites": int(len(g))}
         for c, g in pos.groupby("control")]
    N = [{"kind": "negative", "control": c, "rate": float(g["detected"].mean()), "n_sites": int(len(g))}
         for c, g in neg.groupby("control")]
    mr = float(np.mean([x["rate"] for x in P]))
    fpr = float(neg["detected"].mean())
    return {"n_sites": int(pos["site"].nunique()), "positive": P, "negative": N, "mean_recall": mr,
            "min_recall": float(min(x["rate"] for x in P)), "fpr_pooled": fpr,
            "meets_thresholds": bool(mr >= 0.8 and min(x["rate"] for x in P) >= 0.6 and fpr <= 0.05)}


def run_step4(out: Path, d, val_sites, *, register: bool = False, n_boot: int = 2000) -> dict:
    """Step 4 on the step-3 passers (rung-3 links) of ``ladder_summary.json``; updates the cards."""
    from .step4 import evaluate_links

    f = out / "ladder_summary.json"
    L = json.loads(f.read_text())
    cards = pd.DataFrame(L["cards"])
    passers = cards[cards["rung"] >= 3] if len(cards) else cards
    if not len(passers):
        L["summary"]["step4"] = {"n_links": 0, "note": "no rung-3 link to test"}
        f.write_text(json.dumps(L, indent=1, default=str))
        return L["summary"]["step4"]
    parents_of = passers.groupby("effect")["cause"].apply(list).to_dict()
    res = evaluate_links(d, passers, parents_of, val_sites, n_boot=n_boot)
    res.to_csv(out / "step4_results.csv", index=False)
    ok = {(r["cause"], r["effect"]): r for _, r in res.iterrows()}
    for c in L["cards"]:
        r = ok.get((c["cause"], c["effect"]))
        if r is None:
            continue
        c["step4"] = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.to_dict().items()}
        if c["rung"] >= 3 and bool(r["step4_pass"]):
            c["rung"] = 4
            c["label"] = "causal evidence (with stated assumptions)"
    L["summary"]["causal_evidence_links"] = [f"{c['cause']}->{c['effect']}" for c in L["cards"] if c["rung"] == 4]
    L["summary"]["step4"] = {"n_links": int(len(res)), "n_pass": int(res["step4_pass"].sum())}
    L["summary"]["note"] = "rung 4 = passed steps 1-4"
    f.write_text(json.dumps(L, indent=1, default=str))
    if register:
        from ...core.registry import log_result

        for _, r in res.iterrows():
            for k in ("gain", "event"):
                if np.isfinite(r.get(f"{k}_mean", np.nan)):
                    log_result(workstream="sim", experiment="causal_step4", model="ridge lagged-parent gain" if k == "gain"
                               else "event study (matched non-event days)", metric=f"{k}_mean", value=r[f"{k}_mean"],
                               ci_low=r.get(f"{k}_lo"), ci_high=r.get(f"{k}_hi"),
                               ci_method=f"site bootstrap B={n_boot}", n=int(r[f"{k}_n"]),
                               target=f"{r['cause']}->{r['effect']}", split="val",
                               artifacts=[str(out / "step4_results.csv")], data_manifests=["sonde_daily", "forcings_daily"],
                               notes=f"step4_pass={bool(r['step4_pass'])}")
    return L["summary"]["step4"]


def summarise(out: Path, d, huc2: pd.Series, *, register: bool = False, step1_pass: bool | None = None,
              step1_tag: str | None = None) -> dict:
    s1f = out.parent / "causal_synthetic" / (step1_tag or "") / "step1_results.json"
    if step1_pass is None and s1f.exists():
        step1_pass = json.loads(s1f.read_text())["summary"]["gate_pass"]
    s2 = step2(out, d)
    rep = step3(out, d, huc2) if any((out / "discovery").glob("*.npz")) else pd.DataFrame()
    cards = []
    for _, r in rep.iterrows():
        rung = 2 if r["share_robust"] > 0 else (1 if r["n_sites"] else 0)
        if r["step3_pass"] and step1_pass and s2["pass"]:
            rung = 3
        cards.append({**r.to_dict(), "step1_gate": step1_pass, "step2_gate": s2["pass"], "step4": "not run",
                      "rung": rung,
                      "label": {3: "robust candidate causal link", 2: "conditionally dependent (rung 2)",
                                1: "associated", 0: "untested"}[rung]})
    summ = {"step1_gate_pass": step1_pass, "step1_tag": step1_tag, "step2": s2, "n_links_tested": int(len(rep)),
            "n_links_step3_pass": int(rep["step3_pass"].sum()) if len(rep) else 0,
            "causal_evidence_links": [], "note": "no link is 'causal evidence' until step 4 passes"}
    (out / "ladder_summary.json").write_text(json.dumps({"summary": summ, "cards": cards}, indent=1, default=str))
    if register:
        from ...core.registry import log_result

        art = [str(out / "ladder_summary.json")]
        rc = json.loads((out / "run_config.json").read_text()) if (out / "run_config.json").exists() else {}
        cfg = {"tau_max": 7, "pc_alpha": rc.get("pc_alpha"), "fdr_q": rc.get("fdr_q"), "boot": 50, "block": 10,
               "fdr": "fdr_bh applied to graphs", "detect": C.DETECT, "robust": C.ROBUST, "step1_tag": step1_tag}
        for r in s2["recall_by_control"]:
            log_result(workstream="sim", experiment="causal_step2_controls", model="Bagged-PCMCI+ (ParCorr)",
                       metric="recall", value=r["recall"], n=int(r["n_sites"]), target=r["control"],
                       split="train_sites_train_block", config=cfg, artifacts=art, data_manifests=["sonde_daily", "forcings_daily"])
        for r in s2["fpr_by_class"]:
            log_result(workstream="sim", experiment="causal_step2_controls", model="Bagged-PCMCI+ (ParCorr)",
                       metric="fpr", value=r["fpr"], n=int(r["n_tests"]), target=r["class"],
                       split="train_sites_train_block", config=cfg, artifacts=art, data_manifests=["sonde_daily", "forcings_daily"])
        if "xcorr_baseline" in s2:
            log_result(workstream="sim", experiment="causal_step2_controls", model="lagged cross-correlation baseline",
                       metric="mean_recall", value=s2["xcorr_baseline"]["mean_recall"], is_baseline=True,
                       split="train_sites_train_block", config=cfg, artifacts=art, data_manifests=["sonde_daily", "forcings_daily"])
            log_result(workstream="sim", experiment="causal_step2_controls", model="lagged cross-correlation baseline",
                       metric="fpr", value=s2["xcorr_baseline"]["fpr_N1"], target="N1", is_baseline=True,
                       split="train_sites_train_block", config=cfg, artifacts=art, data_manifests=["sonde_daily", "forcings_daily"])
        for c in cards:
            log_result(workstream="sim", experiment="causal_step3_replication", model="Bagged-PCMCI+ (ParCorr)",
                       metric="share_sites_robust", value=c["share_robust"], n=int(c["n_sites"]),
                       target=f"{c['cause']}->{c['effect']}", split="train_sites_train_block", config=cfg,
                       artifacts=art, data_manifests=["sonde_daily", "forcings_daily"],
                       notes=f"rung {c['rung']}: {c['label']}", extra={k: c[k] for k in
                                                                      ("n_regions_ok", "modal_sign", "modal_lag",
                                                                       "val_share_detected", "step3_pass")})
    return summ
