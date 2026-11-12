"""Scoring rules of the look ``M2S/test``: pure functions over the sealed long tables (no file, no registry, no
row loader, no protected-row access).

Sources: ``docs/preregistration/orbit_arch_v2_amendment.md`` §1.6, §1.7, §2.6, §6; lead rulings R-10 (the MDN_B +
GBM average is a claim comparator), R-15 (small lakes: both arms together, every comparator hypothesis of a target
rejected), R-17 (per-seed-unit rule; v1: every frozen single network), R-17a (+1-corrected paired role-block
bootstrap p; a hypothesis's p is the largest over its claim comparators and counts only if every point difference
favours the arm; step-down Holm with monotone adjusted p), R-18 (H2-S statuses, divisor 6 either way, NTU-class
turbidity rows), R-18b (label wording); clarification 01 K-1 (small-lake family of 32 elements).

Sealed table (written by ``scripts/eval/orbitv2_test_s.py``, one per cell): one row per (row, target) with a
label; columns ``target, row_id, block, y`` (log10; NaN for a left-censored label, whose detection limit is in
``y_dl``), ``small_lake, ntu_class, scene_disjoint`` and one column per sealed model (``looks.sealed_columns``;
log10 predictions).

Conservative readings made here (each can only remove a label, never add one; reported to the lead):
- Holm uses the family's fixed divisor (2, 6, 3, 32) whatever the number of hypotheses still confirmatory;
- a label needs every labelled row of the endpoint predicted by the arm, its members and every claim comparator
  (rows with a non-finite prediction are dropped from the comparison, counted, and block the label).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..orbit import evaluate as E
from . import looks as LK
from . import train as TR
from . import units as U

N_BOOT, SEED, ALPHA = 2000, 0, 0.05                       # R-17a
BASE_COMPARATORS = ("mdn_b", "gbm", "knn", "mdn_b+gbm")   # §6.2 + R-10
TRACK_N_COMPARATORS = ("mdn_b", "gbm", "knn", "season_loc_gbm", "trophic_ols", "mdn_b+gbm")       # §2.5 + R-10
ARMS = (LK.V1_MODEL, LK.V2_MODEL)
# "iu": one hypothesis per (target, arm), intersection-union over the comparators (R-17a);
# "element": one hypothesis per (target, comparator, arm); a label needs all of a target's elements (R-15, K-1)
FAMILIES = {
    "H1-S": {"divisor": 2, "targets": ("chla",), "stratum": "all", "comparators": BASE_COMPARATORS, "arms": ARMS,
             "form": "iu"},
    "H2-S": {"divisor": 6, "targets": ("secchi", "tss", "turbidity"), "stratum": "all",
             "comparators": BASE_COMPARATORS, "arms": ARMS, "form": "iu"},
    "TrackN": {"divisor": 3, "targets": ("tp", "tn", "doc"), "stratum": "all", "comparators": TRACK_N_COMPARATORS,
               "arms": (LK.TRACK_N_MODEL,), "form": "iu"},
    "small_lakes": {"divisor": 32, "targets": ("chla", "secchi", "tss", "turbidity"), "stratum": "small_lake",
                    "comparators": BASE_COMPARATORS, "arms": ARMS, "form": "element"},
}
CONFIRMATORY = ("primary", "confirmatory")                # statuses of scripts/train/orbitv2_power.py
STATUSES = (*CONFIRMATORY, "descriptive", "no SENTINEL claim")
SCOPE = "unseen lakes, archive years 2015–2023"           # R-18b
SCENE_WORDS = {"scene_disjoint": "scene-disjoint", "all_rows": "scenes shared with training"}
V1_SENTENCE = "the per-seed rule was applied to each of the {n} frozen single networks; {k} of {n} individually beat every comparator"
FAR_KM = 10.0                                             # §6.4
EARLY_MAX_YEAR = 2021                                     # §6.4: <= 2021 versus 2022-23


class ClaimError(RuntimeError):
    pass


def cmp_col(name: str) -> str:
    return f"{LK.COMPARATOR_PREFIX}{name}"


# ---------------------------------------------------------------------------------------------- statistics

def paired(y_log, pa_log, pb_log, groups, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """MdSA(a) - MdSA(b) on identical rows (negative: a better) with the role-block bootstrap of
    ``evaluate.paired_mdsa`` (the same draws) and the +1-corrected two-sided p-value of R-17a."""
    y, pa, pb = (np.asarray(v, float) for v in (y_log, pa_log, pb_log))
    groups = np.asarray(groups)
    if not len(y) or not (np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)).all():
        raise ClaimError("paired comparison needs at least one row and finite labels and predictions")
    ea, eb = pa - y, pb - y
    d = np.array([E.mdsa_log10(ea[ix]) - E.mdsa_log10(eb[ix]) for ix in E._boot_indices(groups, n_boot, seed)])
    lo, hi = np.quantile(d, [0.025, 0.975])
    p = min(1.0, 2 * min((int((d <= 0).sum()) + 1) / (n_boot + 1), (int((d >= 0).sum()) + 1) / (n_boot + 1)))
    return {"diff": float(E.mdsa_log10(ea) - E.mdsa_log10(eb)), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(p), "n_clusters": int(len(np.unique(groups))), "n": int(len(y)), "n_boot": int(n_boot)}


def holm(pvalues: dict, divisor: int, alpha: float = ALPHA) -> dict:
    """Step-down Holm with monotone adjusted p-values over a family of fixed size ``divisor``: the i-th smallest
    p (i = 0, 1, ...) is multiplied by ``divisor - i``; a hypothesis is rejected if its adjusted p < alpha."""
    if len(pvalues) > divisor:
        raise ClaimError(f"{len(pvalues)} hypotheses in a family of {divisor}")
    if any(not (0.0 <= float(p) <= 1.0) for p in pvalues.values()):
        raise ClaimError("a p-value is missing or outside [0, 1]")
    out, run = {}, 0.0
    for i, k in enumerate(sorted(pvalues, key=lambda k: (float(pvalues[k]), str(k)))):
        run = max(run, min(1.0, (divisor - i) * float(pvalues[k])))
        out[k] = {"p": float(pvalues[k]), "p_holm": float(run), "rank": i + 1, "alpha_at_rank": alpha / (divisor - i),
                  "rejected": bool(run < alpha)}
    return out


# -------------------------------------------------------------------------------------------- endpoint rows

def endpoint_rows(tab: pd.DataFrame, target: str, stratum: str = "all") -> pd.DataFrame:
    """Rows of one endpoint in a sealed cell table: the target's quantified rows (a left-censored row is sealed
    with ``y`` = NaN for the descriptive censored-row table and never enters an endpoint); NTU-class rows only for
    turbidity (R-18); the small-lake stratum of §1.7."""
    g = tab[(tab["target"] == target) & np.isfinite(tab["y"].to_numpy(float))]
    if target == "turbidity":
        g = g[g["ntu_class"].astype(bool)]
    if stratum == "small_lake":
        g = g[g["small_lake"].astype(bool)]
    elif stratum != "all":
        raise ClaimError(f"unknown stratum {stratum!r}")
    return g.reset_index(drop=True)


def primary_rows(tab: pd.DataFrame, primary_cell: str, *, sensitivity: bool = False) -> pd.DataFrame:
    """§1.6: the primary cell is the scene-disjoint subset or all rows, as fixed in the freeze; the other one is
    the sensitivity analysis."""
    if primary_cell not in LK.PRIMARY_CELLS:
        raise ClaimError(f"primary cell must be one of {LK.PRIMARY_CELLS}")
    disjoint = (primary_cell == "scene_disjoint") != sensitivity
    return tab[tab["scene_disjoint"].astype(bool)].reset_index(drop=True) if disjoint else tab


def member_names(arm: str, n_v1: int) -> dict:
    """{"units": [...], "networks": [...]} column names of the arm's members (R-17)."""
    if arm == LK.V1_MODEL:
        return {"units": [], "networks": [f"{arm}@s{i}" for i in range(int(n_v1))]}
    return {"units": [f"{arm}@unit{u}" for u in TR.UNITS],
            "networks": [f"{arm}@u{u}i{i}" for u in TR.UNITS for i in TR.INITS]}


def per_seed(scores: dict, arm: str, comparators, n_v1: int) -> dict:
    """R-17 on one endpoint from ``scores`` = {column: MdSA}. v2 / Track N arm: each of the 3 seed units alone is
    lower than every claim comparator (``units.per_seed_rule``). v1: every frozen single network is."""
    comp = {c: scores[cmp_col(c)] for c in comparators}
    names = member_names(arm, n_v1)
    if arm != LK.V1_MODEL:
        return U.per_seed_rule([scores[c] for c in names["units"]], [scores[c] for c in names["networks"]], comp)
    nets = [float(scores[c]) for c in names["networks"]]
    if not nets or not np.isfinite(nets).all() or not np.isfinite(list(comp.values())).all():
        raise ClaimError("the per-seed rule needs every frozen single network's score and every comparator score")
    worst = min(comp.values())
    k = int(sum(x < worst for x in nets))
    return {"passed": k == len(nets), "k_networks": k, "n_networks": len(nets), "network_scores": nets,
            "best_comparator": min(comp, key=comp.get), "comparators": comp,
            "sentence": V1_SENTENCE.format(n=len(nets), k=k)}


def endpoint(g: pd.DataFrame, arm: str, comparators, n_v1: int, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """One arm on one endpoint's rows ``g``: MdSA of the arm, its members and each claim comparator on the common
    rows; the paired comparison with each comparator; the intersection-union p (R-17a); the per-seed rule."""
    names = member_names(arm, n_v1)
    cols = [arm, *names["units"], *names["networks"], *[cmp_col(c) for c in comparators]]
    missing = [c for c in cols if c not in g.columns]
    if missing:
        raise ClaimError(f"sealed table lacks {missing[:4]} ({len(missing)} columns)")
    y = g["y"].to_numpy(float)
    if not np.isfinite(y).all():
        raise ClaimError("a sealed endpoint row has no quantified label")
    P = g[cols].to_numpy(float)
    ok = np.isfinite(P).all(axis=1)
    out = {"arm": arm, "n_labelled": int(len(g)), "n": int(ok.sum()), "n_dropped_nonfinite": int((~ok).sum()),
           "n_blocks": int(g["block"][ok].nunique()), "comparators": list(comparators), "evaluable": bool(ok.any())}
    if not ok.any():
        return {**out, "mdsa": {}, "pairs": {}, "p_iu": 1.0, "all_favour": False, "p_counted": 1.0,
                "per_seed": {"passed": False}}
    yy, blocks = y[ok], g["block"].to_numpy()[ok]
    scores = {c: E.mdsa_log10(P[ok, j] - yy) for j, c in enumerate(cols)}
    pairs = {c: paired(yy, P[ok, 0], P[ok, cols.index(cmp_col(c))], blocks, n_boot, seed) for c in comparators}
    favour = all(p["diff"] < 0 for p in pairs.values())
    p_iu = max(p["p_value"] for p in pairs.values())
    return {**out, "mdsa": scores, "pairs": pairs, "p_iu": float(p_iu), "all_favour": bool(favour),
            "p_counted": float(p_iu) if favour else 1.0, "per_seed": per_seed(scores, arm, comparators, n_v1)}


# -------------------------------------------------------------------------------------------------- families

def target_status(power_status: dict, family: str, target: str) -> str:
    st = (power_status.get(family) or {}).get(target, {}).get("status")
    if st not in STATUSES:
        raise ClaimError(f"the freeze gives no status for {family} / {target}")
    if family == "H1-S" and st != "primary":
        raise ClaimError("H1-S chl-a is the primary endpoint (amendment §6.2)")
    return st


def family(name: str, tab: pd.DataFrame, power_status: dict, n_v1: int, *, has_track_n: bool = True,
           n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """One family on the rows of one cell (the primary cell for the claim; the same call on the sensitivity cell
    is descriptive). Every endpoint is scored and reported; Holm runs over the confirmatory hypotheses with the
    family's fixed divisor; ``labels`` lists the (target, arm) pairs that meet every condition of §6.2:
    adjusted p < 0.05, every point difference in the arm's favour, the per-seed rule, and no dropped row."""
    fam = FAMILIES[name]
    arms = [a for a in fam["arms"] if a != LK.TRACK_N_MODEL or has_track_n]
    ends, pvals = {}, {}
    for t in fam["targets"]:
        st = target_status(power_status, name, t)
        g = endpoint_rows(tab, t, fam["stratum"])
        for arm in arms:
            e = endpoint(g, arm, fam["comparators"], n_v1, n_boot=n_boot, seed=seed)
            e.update(target=t, status=st, confirmatory=st in CONFIRMATORY)
            ends[t, arm] = e
            if not e["confirmatory"]:
                continue
            if fam["form"] == "iu":
                pvals[t, arm] = e["p_counted"]
            else:                                                   # K-1: element = (target, comparator, arm)
                for c in fam["comparators"]:
                    pr = e["pairs"].get(c)
                    pvals[t, c, arm] = pr["p_value"] if pr and pr["diff"] < 0 else 1.0
    hm = holm(pvals, fam["divisor"]) if pvals else {}
    labels = []
    for (t, arm), e in ends.items():
        if not e["confirmatory"]:
            e.update(holm=None, label=False)
            continue
        keys = [(t, arm)] if fam["form"] == "iu" else [(t, c, arm) for c in fam["comparators"]]
        e["holm"] = {("|".join(k[1:-1]) or "iu"): hm[k] for k in keys}
        e["label"] = bool(all(hm[k]["rejected"] for k in keys) and e["all_favour"] and e["per_seed"]["passed"]
                          and e["n_dropped_nonfinite"] == 0)
        if e["label"]:
            labels.append((t, arm))
    return {"family": name, "divisor": fam["divisor"], "form": fam["form"], "stratum": fam["stratum"],
            "n_hypotheses": len(pvals), "endpoints": ends, "labels": labels}


def all_families(tab: pd.DataFrame, power_status: dict, n_v1: int, *, has_track_n: bool, n_boot: int = N_BOOT,
                 seed: int = SEED) -> dict:
    out = {}
    for name in FAMILIES:
        if name == "TrackN" and not has_track_n:
            for t in FAMILIES[name]["targets"]:
                if target_status(power_status, name, t) in CONFIRMATORY:
                    raise ClaimError("Track N has a confirmatory target but no Track N arm is sealed (§2.7)")
        out[name] = family(name, tab, power_status, n_v1, has_track_n=has_track_n, n_boot=n_boot, seed=seed)
    return out


def counts_sentence(families: dict) -> str:
    """R-18b: every report states the number of confirmatory families and hypotheses in the look."""
    conf = {k: f["n_hypotheses"] for k, f in families.items() if f["n_hypotheses"]}
    return (f"confirmatory families in this look: {len(conf)} ({', '.join(conf) or 'none'}); "
            f"confirmatory hypotheses: {sum(conf.values())}")


def label_scope(primary_cell: str) -> str:
    """R-18b: the scope every Test-S label is written with."""
    if primary_cell not in SCENE_WORDS:
        raise ClaimError(f"primary cell must be one of {tuple(SCENE_WORDS)}")
    return f"{SCOPE}, {SCENE_WORDS[primary_cell]}"


# ---------------------------------------------------------------------------------------- descriptive strata

def strata(g: pd.DataFrame) -> dict:
    """§6.4 strata of sealed rows: {name: mask}. Needs ``year, source, n_context, dist_train_km``."""
    yr, src = g["year"].to_numpy(int), g["source"].astype(str).str.lower().to_numpy()
    ctx, far = g["n_context"].to_numpy(float) >= 1, g["dist_train_km"].to_numpy(float) > FAR_KM
    aq = np.char.startswith(src.astype(str), "aquacorr")
    return {f"year<={EARLY_MAX_YEAR}": yr <= EARLY_MAX_YEAR, f"year>{EARLY_MAX_YEAR}": yr > EARLY_MAX_YEAR,
            "source=aquacorr": aq, "source=wqp": ~aq, "context>=1": ctx, "context=0": ~ctx,
            f">{FAR_KM:g}km_from_train": far}


def descriptive(g: pd.DataFrame, models, *, n_boot: int = N_BOOT, seed: int = SEED, min_rows: int = 20,
                min_blocks: int = 5) -> pd.DataFrame:
    """MdSA with a role-block bootstrap interval of every model in ``models`` on the rows of ``g`` (one target)
    and on each §6.4 stratum; strata below ``min_rows`` rows or ``min_blocks`` blocks are listed with counts
    only."""
    rows = []
    y, blocks = g["y"].to_numpy(float), g["block"].to_numpy()
    cols = [m for m in models if m in g.columns and np.isfinite(g[m].to_numpy(float)).any()]
    for name, m in {"all": np.ones(len(g), bool), **strata(g)}.items():
        base = {"stratum": name, "n": int(m.sum()), "n_blocks": int(pd.Series(blocks[m]).nunique())}
        if base["n"] < min_rows or base["n_blocks"] < min_blocks or not cols:
            rows.append({**base, "model": None, "note": "counts only (too few rows or blocks)"})
            continue
        sc, ok = E.score_models(y[m], {c: g[c].to_numpy(float)[m] for c in cols}, blocks[m], n_boot=n_boot, seed=seed)
        rows += [{**base, **r, "n_common": int(ok.sum())} for r in sc.to_dict("records")]
    return pd.DataFrame(rows)


def interval_table(g: pd.DataFrame, lo: str, hi: str, offset: float | None = None) -> dict:
    """Coverage and median width (log10) of a sealed interval; ``offset``: split-conformal widening fitted on
    validation (protocol §5b)."""
    y, a, b = (g[c].to_numpy(float) for c in ("y", lo, hi))
    if offset is not None:
        a, b = a - offset, b + offset
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return {"n": 0, "coverage": None, "median_width_log10": None}
    return {"n": int(ok.sum()), "coverage": float(np.mean((y[ok] >= a[ok]) & (y[ok] <= b[ok]))),
            "median_width_log10": float(np.median(b[ok] - a[ok]))}
