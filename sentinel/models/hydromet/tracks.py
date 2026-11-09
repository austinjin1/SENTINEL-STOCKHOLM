# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Harder M6 benchmark tracks (protocol §10–§11; RESEARCH_PLAN §15.5 amendment of 2026-11-08).

* **Track R — leave-region-out.** Basin groups of the FINAL split are assigned to one of six fixed macro-regions
  (unions of whole HUC2 water-resource regions, :data:`REGION_FOLDS`, chosen by geography before any result).
  For fold k, models are fitted on train- AND val-role sites outside region k (train block only) and scored on
  the train/val-role sites of region k × the val block (cross-validation cell). Test-role sites are never fitted
  on; in the single test look each is scored by the model of its own region's fold, on the test block.
* **Track S — a-priori difficulty strata** (:func:`strata_masks`), thresholds from train-role sites × train
  block only.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

REGION_FOLDS = {   # fold -> HUC2 regions (USGS water-resource regions)
    "R1_northeast": ("01", "02"),
    "R2_southeast": ("03", "06", "08"),
    "R3_midwest_greatlakes": ("04", "05", "07", "09"),
    "R4_plains_texas": ("10", "11", "12"),
    "R5_southwest_basin": ("13", "14", "15", "16"),
    "R6_pacific": ("17", "18"),
}
HUC2_TO_FOLD = {h: f for f, hs in REGION_FOLDS.items() for h in hs}


def assign_folds(sites: list[str], huc8: dict, basin_group: dict) -> dict[str, str]:
    """Site -> fold. A basin group goes whole to the fold of the HUC2 holding most of its sites (ties:
    lowest HUC2 code), so flow-connected stations never straddle folds."""
    df = pd.DataFrame({"site": sites, "bg": [basin_group[s] for s in sites],
                       "h2": [str(huc8[s])[:2] for s in sites]})
    bad = sorted(set(df["h2"]) - set(HUC2_TO_FOLD))
    if bad:
        raise ValueError(f"HUC2 codes without a fold: {bad}")
    maj = (df.groupby(["bg", "h2"]).size().reset_index(name="n").sort_values(["bg", "n", "h2"],
                                                                                 ascending=[True, False, True])
           .drop_duplicates("bg").set_index("bg")["h2"])
    return {s: HUC2_TO_FOLD[maj[g]] for s, g in zip(df["site"], df["bg"])}


def folds_payload(fold_of: dict, split_sha256: str, roles: dict) -> dict:
    sites = sorted(fold_of)
    body = {"track": "M6 track R (leave-region-out)", "split_sha256": split_sha256,
            "region_folds": {k: list(v) for k, v in REGION_FOLDS.items()},
            "fold_of": {s: fold_of[s] for s in sites}, "role_of": {s: roles[s] for s in sites}}
    body["sha256"] = hashlib.sha256(json.dumps({k: body[k] for k in ("split_sha256", "region_folds", "fold_of",
                                                                     "role_of")}, sort_keys=True).encode()).hexdigest()
    return body


TRACK_R_FILE = "docs/preregistration/hydromet_track_r_folds.json"
TRACK_R_SHA256 = "142341a9d0379f2b50c90bdc72508cce8aa24d78d79db855aba9f1065f92a6c8"   # protocol §10 (frozen)


def load_folds(path=None) -> dict:
    from ...core import paths

    p = paths.REPO_ROOT / (path or TRACK_R_FILE)
    body = json.loads(p.read_text())
    re = folds_payload(body["fold_of"], body["split_sha256"], body["role_of"])
    if re["sha256"] != body["sha256"] or body["sha256"] != TRACK_R_SHA256:
        raise ValueError(f"{p}: sha256 mismatch with the frozen track-R folds")
    return body


def fold_roles(fold_of: dict, role_of: dict, fold: str) -> dict[str, str]:
    """Track-R roles for one fold: 'train' = train/val-role sites outside the fold, 'val' = train/val-role
    sites inside it, 'test' = FINAL test-role sites (never fitted; scored only in the frozen test look)."""
    out = {}
    for s, f in fold_of.items():
        r = role_of[s]
        out[s] = "test" if r == "test" else ("val" if f == fold else "train")
    return out


# ----------------------------------------------------------------------------- Track S strata

STRATA = {
    "warm_p90": ("temp_c_mean", "temp_c_7dadm"),   # observed value > pooled train P90 of that target
    "heatwave": ("temp_c_mean", "temp_c_7dadm"),   # >= 3 consecutive days of site Tmax > site train P95, + 3 days
    "low_do_summer": ("do_mgl_mean", "do_mgl_min"),  # Jul–Sep and observed value < pooled train P10
}
DO_CRITERION_MGL = 5.0      # exceedance (CSI) threshold for do_mgl_min (EPA warm-water minimum)


def train_thresholds(y: np.ndarray, train_sites: np.ndarray, train_days: np.ndarray, targets: list[str]) -> dict:
    """Pooled P90 (temperature) / P10 (DO) over train-role sites × train block."""
    out = {}
    for j, v in enumerate(targets):
        x = y[np.ix_(np.asarray(train_sites), np.flatnonzero(train_days))][:, :, j]
        x = x[np.isfinite(x)]
        if v.startswith("temp"):
            out[v] = float(np.percentile(x, 90))
        elif v.startswith("do_"):
            out[v] = float(np.percentile(x, 10))
    return out


def heatwave_mask(tmax: np.ndarray, train_days: np.ndarray, *, q: float = 95, run: int = 3, after: int = 3) -> np.ndarray:
    """(S, T) bool: days in runs of >= ``run`` consecutive days with site Tmax > the site's own train-block
    P``q``, plus ``after`` days following each run (forcing-defined; no target is used)."""
    S, T = tmax.shape
    thr = np.nanpercentile(np.where(train_days[None, :], tmax, np.nan), q, axis=1)
    hot = tmax > thr[:, None]
    out = np.zeros((S, T), bool)
    for s in range(S):
        h = hot[s]
        t = 0
        while t < T:
            if h[t]:
                u = t
                while u < T and h[u]:
                    u += 1
                if u - t >= run:
                    out[s, t:min(u + after, T)] = True
                t = u
            else:
                t += 1
    return out


def strata_masks(y: np.ndarray, dates: pd.DatetimeIndex, tmax: np.ndarray, thresholds: dict, train_days: np.ndarray,
                 targets: list[str]) -> dict[tuple[str, str], np.ndarray]:
    """{(stratum, target): (S, T) bool} on the observed-target cells (scoring masks are intersected later)."""
    out = {}
    hw = heatwave_mask(tmax, train_days)
    summer = np.isin(pd.DatetimeIndex(dates).month, [7, 8, 9])[None, :]
    for j, v in enumerate(targets):
        obs = np.isfinite(y[:, :, j])
        if v in STRATA["warm_p90"]:
            out[("warm_p90", v)] = obs & (y[:, :, j] > thresholds[v])
            out[("heatwave", v)] = obs & hw
        if v in STRATA["low_do_summer"]:
            out[("low_do_summer", v)] = obs & summer & (y[:, :, j] < thresholds[v])
    return out


def csi(obs_exceed: np.ndarray, pred_exceed: np.ndarray) -> float:
    """Critical success index hits / (hits + misses + false alarms)."""
    h = int((obs_exceed & pred_exceed).sum())
    m = int((obs_exceed & ~pred_exceed).sum())
    f = int((~obs_exceed & pred_exceed).sum())
    return h / (h + m + f) if h + m + f else float("nan")
