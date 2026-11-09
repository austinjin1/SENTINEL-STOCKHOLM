# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Latent validation of Sim GPP / ER / K600 against independent metabolism estimates (protocol §6).

Powell Center (Appling et al. 2018) daily estimates, confidence H/M, plausible days (GPP ≥ −0.5,
ER ≤ 0.5) — loaded with the D8 RiverPulse loader (imported, not modified). Per-site Spearman over
overlapping days (≥ 100), median across sites with a basin-group bootstrap CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ...core import paths
from . import data as D

MIN_DAYS = 100


def load_powell() -> pd.DataFrame:
    from ..downstream.riverpulse import load_labels

    lab = load_labels(confidence=("H", "M"), plausible=True)
    k = pd.read_parquet(paths.PROCESSED / "labels_powell_metabolism" / "daily_metabolism.parquet",
                        columns=["site_id", "date", "k600"])
    k["date"] = pd.to_datetime(k["date"])
    lab = lab.merge(k, on=["site_id", "date"], how="left")
    lab["nwis_id"] = lab["nwis_id"].astype(str)
    return lab


def load_streampulse() -> pd.DataFrame:
    """StreamPULSE public daily metabolism (secondary comparator, protocol §6): rows with a USGS site
    number, ``valid_day``, not flagged (GPP < 0 / ER > 0) and the same plausibility filter as Powell."""
    sp = pd.read_parquet(paths.PROCESSED / "labels_streampulse" / "daily_metabolism.parquet",
                         columns=["usgs_site_no", "date", "gpp", "er", "k600", "valid_day", "flag_gpp_negative",
                                  "flag_er_positive", "dup_site_date_ci"])
    sp = sp[sp["usgs_site_no"].notna() & sp["valid_day"].fillna(False).astype(bool)
            & ~sp["flag_gpp_negative"].fillna(False).astype(bool) & ~sp["flag_er_positive"].fillna(False).astype(bool)
            & ~sp["dup_site_date_ci"].fillna(False).astype(bool)]
    sp = sp[(sp["gpp"] >= -0.5) & (sp["er"] <= 0.5)]
    sp = sp.assign(nwis_id=sp["usgs_site_no"].astype(str), date=pd.to_datetime(sp["date"]))
    return sp.drop_duplicates(["nwis_id", "date"])


def compare(d: D.SimData, sim: dict, roles=("val",), labels: pd.DataFrame | None = None) -> dict:
    lab = load_powell() if labels is None else labels
    pos = {s: i for i, s in enumerate(d.sites)}
    t_index = {t: j for j, t in enumerate(d.dates)}
    rows = []
    for s, g in lab.groupby("nwis_id"):
        i = pos.get(s)
        if i is None or d.role[i] not in roles:
            continue
        j = np.array([t_index.get(t, -1) for t in g["date"]])
        ok = j >= 0
        if ok.sum() < MIN_DAYS:
            continue
        g = g[ok]
        j = j[ok]
        rec = {"site": s, "basin_group": d.basin_group[i], "n_days": int(len(g))}
        for v in ("gpp", "er", "k600"):
            a = sim[v][i, j]
            b = g[v].to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b)
            rec[f"n_{v}"] = int(m.sum())
            if m.sum() >= MIN_DAYS:
                rec[f"rho_{v}"] = float(spearmanr(a[m], b[m]).statistic)
                rec[f"mean_sim_{v}"] = float(np.mean(a[m]))
                rec[f"mean_ref_{v}"] = float(np.mean(b[m]))
                if v != "er":
                    rec[f"med_logratio_{v}"] = float(np.median(np.log(np.maximum(a[m], 1e-3) / np.maximum(b[m], 1e-3))))
        rows.append(rec)
    df = pd.DataFrame(rows)
    out = {"n_sites": int(len(df)), "per_site": df}
    rs = np.random.default_rng(0)
    for v in ("gpp", "er", "k600"):
        c = f"rho_{v}"
        if c not in df or df[c].notna().sum() < 3:
            continue
        x = df[[c, "basin_group"]].dropna()
        g, inv = np.unique(x["basin_group"], return_inverse=True)
        members = [np.flatnonzero(inv == k) for k in range(len(g))]
        bs = []
        for _ in range(2000):
            pick = rs.integers(0, len(g), len(g))
            idx = np.concatenate([members[k] for k in pick])
            bs.append(np.median(x[c].to_numpy()[idx]))
        out[v] = {"median_rho": float(x[c].median()), "ci_low": float(np.quantile(bs, 0.025)),
                  "ci_high": float(np.quantile(bs, 0.975)), "n_sites": int(len(x)), "n_groups": int(len(g))}
        ms, mr = df.get(f"mean_sim_{v}"), df.get(f"mean_ref_{v}")
        if ms is not None:
            mm = ms.notna() & mr.notna()
            if mm.sum() >= 5:
                out[v]["between_site_rho"] = float(spearmanr(ms[mm], mr[mm]).statistic)
    out["S_H3_pass"] = bool(any(out.get(v, {}).get("median_rho", -1) >= 0.5 and out[v]["ci_low"] > 0
                                for v in ("gpp", "er")))
    return out
