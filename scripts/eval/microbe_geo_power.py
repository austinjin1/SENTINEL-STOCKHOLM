#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""GEO power file (amendment microbe_arch_v2 §7.3; lead ruling B9; ``microbe_arch_v2_deviations_01.md`` P1 as
changed by ``microbe_arch_v2_dev01_lead_rulings.md`` D2-D4).

On the family-G ``lodo`` out-of-fold predictions of the development events:

* Pairs (D3): each finished entry among G1, G3, G7, G8, G9 against every comparator amendment §7.4 names for it:
  ``tier0_best``, ``rf_clr``, ``xgb_clr``, ``rf_clr_tuned``, ``xgb_clr_tuned``, and for the entries that read a
  context column (G1, G8) also ``rf_clr_context`` and ``xgb_clr_context``. Stochastic models use their seed-0
  predictions.
* SE of a pair (D4): paired site-cluster bootstrap of the difference of R², B = 1,000, bootstrap seed 0; width of
  the 95% percentile interval of ``evaluate.paired_vs`` divided by 2·z_0.975, as for
  ``microbe_split_lodo_x001_report.json`` (``scripts/train/microbe_freeze_track.py``).
* The power figure of a target uses the LARGEST SE over its pairs.
* Development MDD = (z_(1-alpha/2) + z_0.8) * SE with alpha = 0.05/8.
* Holdout-scaled MDD (D2): SE * sqrt(1,215 / 214) in place of SE, for every target.
* A target is "underpowered" if its holdout-scaled MDD exceeds 0.30. It stays in the family; the Holm
  denominator stays 8.
* An entry reported as not run (``--not-run G3=<registry row id>`` or ``--not-run G3=log:E12``, checked against
  the registry or the headings of ``microbe_arch_v2_log.md``) contributes no pair and does not block the file. An
  entry that is neither finished nor reported as not run blocks it, and so does a missing comparator.

Development predictions and development labels only: no protected block is opened. The file is written ONCE
(the script refuses if it exists), before G10 is chosen. It needs the stamped deviations file that releases the
key ``GEO_POWER``.

    PYTHONPATH=. python -B scripts/eval/microbe_geo_power.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.registry import log_result

ENTRIES = ("G1", "G3", "G7", "G8", "G9")
COMPARATORS = ("tier0_best", "rf_clr", "xgb_clr", "rf_clr_tuned", "xgb_clr_tuned")      # amendment §7.4
CONTEXT_COMPARATORS = ("rf_clr_context", "xgb_clr_context")                           # for entries reading context
CONTEXT_ENTRIES = ("G1", "G8")
DEV_EVENTS = 1215
N_BOOT = 1000
BOOT_SEED = 0
ALPHA = 0.05 / 8
POWER = 0.80
HOLDOUT_EVENTS = 214
THRESHOLD = 0.30
EXPERIMENT = "microbe_arch_v2_geo_power"
OUT_REL = "docs/preregistration/microbe_geo_power.json"
LOG_REL = "docs/preregistration/microbe_arch_v2_log.md"


def paired_se(y, p_a, p_b, region, site, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """SE of R²(a) − R²(b) as ``scripts/train/microbe_freeze_track.py`` computed it for the reference report: the
    width of the 95% percentile interval of ``evaluate.paired_vs`` (sites resampled) divided by 2·z_0.975."""
    from sentinel.models.microbe import evaluate as E

    y, a, b = (np.asarray(v, float) for v in (y, p_a, p_b))
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    out = {"n": int(ok.sum()), "n_sites": int(len(np.unique(np.asarray(site)[ok]))) if ok.any() else 0,
           "diff": float("nan"), "se": float("nan")}
    if out["n"] < 3 or out["n_sites"] < 2:
        return out
    o = E.paired_vs(y, a, b, region, site, metric="r2", n_boot=n_boot, seed=seed)
    out["diff"] = float(o["diff"])
    out["se"] = float((o["ci_high"] - o["ci_low"]) / (2 * norm.ppf(0.975)))
    return out


def mdd(se: float, alpha: float = ALPHA, power: float = POWER) -> float:
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def comparators_of(entry: str) -> tuple:
    return COMPARATORS + (CONTEXT_COMPARATORS if entry in CONTEXT_ENTRIES else ())


def power_table(y: pd.DataFrame, region, site, pairs: dict, targets) -> dict:
    """``pairs``: {label: (model frame, comparator frame)}, frames indexed like ``y``. Per target: every pair's SE,
    the largest, and the MDDs."""
    scale = float(np.sqrt(DEV_EVENTS / HOLDOUT_EVENTS))
    res = {}
    for t in targets:
        pr = {k: paired_se(y[t].to_numpy(float), a[t].to_numpy(float), b[t].to_numpy(float), region, site)
              for k, (a, b) in pairs.items()}
        ses = {k: v["se"] for k, v in pr.items() if np.isfinite(v["se"])}
        if not ses:
            res[t] = {"pairs": pr, "paired_se_site_max": float("nan"), "underpowered": None}
            continue
        worst = max(ses, key=ses.get)
        se = ses[worst]
        m_hold = mdd(se * scale)
        res[t] = {"pairs": pr, "paired_se_site_max": se, "pair_of_max": worst, "n_dev": pr[worst]["n"],
                  "mdd_dev_alpha_0.05/8": mdd(se), "holdout_scale": scale, "mdd_holdout_alpha_0.05/8": m_hold,
                  "underpowered": bool(m_hold > THRESHOLD)}
    return res


def not_run_reference(ref: str, registry_path: Path, log_path: Path) -> bool:
    """A registry row id that exists, or ``log:<heading id>`` naming a ``### <id>`` heading of the execution log."""
    if ref.startswith("log:"):
        key = ref[4:].strip()
        if not key or not log_path.is_file():
            return False
        return any(line.startswith(f"### {key} ") or line.strip() == f"### {key}"
                   for line in log_path.read_text().splitlines())
    if not ref or not registry_path.is_file():
        return False
    with registry_path.open() as fh:
        for line in fh:
            if ref in line:
                try:
                    if json.loads(line).get("id") == ref:
                        return True
                except json.JSONDecodeError:
                    continue
    return False


def seed0(d: Path, name: str) -> tuple[pd.DataFrame, str]:
    """Seed-0 predictions of a stochastic model if stored per seed, else the model's single file."""
    for f in (d / f"{name}_seed0.parquet", d / f"{name}.parquet"):
        if f.is_file():
            return pd.read_parquet(f), f.name
    raise FileNotFoundError(f"{name}: no out-of-fold file in {d}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scheme", default="lodo")
    ap.add_argument("--tag", default="")
    ap.add_argument("--not-run", nargs="*", default=[], metavar="ENTRY=REF",
                    help="an entry reported as not run, with its registry row id or log:<heading id>")
    a = ap.parse_args(argv)

    from sentinel.models.microbe import archv2 as AV
    from sentinel.models.microbe import stamp as ST

    rel = ST.require_release("GEO power file (amendment §7.3)", "GEO_POWER")
    out_file = paths.REPO_ROOT / OUT_REL
    if out_file.exists():
        raise AV.GateError(f"{OUT_REL} exists; the GEO power file is written once")
    if a.scheme not in AV.RELEASED_SCHEMES:
        raise AV.GateError(f"scheme {a.scheme} is not released")
    not_run = {}
    for item in a.not_run:
        g, _, ref = item.partition("=")
        if g not in ENTRIES:
            raise AV.GateError(f"--not-run {item}: {g!r} is not one of {ENTRIES}")
        if not not_run_reference(ref, paths.REGISTRY_PATH, paths.REPO_ROOT / LOG_REL):
            raise AV.GateError(f"--not-run {item}: {ref!r} is neither a registry row id nor a heading of {LOG_REL}")
        not_run[g] = ref
    d = paths.RUNS_ROOT / "microbe" / f"cv_archv2{a.tag}" / a.scheme / "G"
    names = {g: AV.GEO_ENTRIES[g] for g in ENTRIES}
    finished = {g: (d / f"{n}_scores.json").is_file() for g, n in names.items()}
    both = [g for g in not_run if finished[g]]
    if both:
        raise AV.GateError(f"reported as not run but finished on {a.scheme}/G: {', '.join(both)}")
    entries = [g for g in ENTRIES if finished[g]]
    need = sorted({c for g in entries for c in comparators_of(g)}, key=(COMPARATORS + CONTEXT_COMPARATORS).index)
    missing = [f"{g} ({names[g]})" for g in ENTRIES if not finished[g] and g not in not_run]
    missing += [c for c in (need or COMPARATORS) if not (d / f"{c}_scores.json").is_file()]
    if missing or not entries:
        raise AV.GateError(f"not finished on {a.scheme}/G: {', '.join(missing) or 'every entry'}; the power file "
                           f"needs each of {', '.join(ENTRIES)} finished or reported as not run, and every "
                           "comparator of amendment §7.4")

    from sentinel.models.microbe.design import load_design

    design = AV.attach_master(load_design())
    S = design.samples
    targets = list(AV.FAMILIES["G"])
    files, frames = {}, {}
    for key, n in [(f"{g}:{names[g]}", names[g]) for g in entries] + [(c, c) for c in need]:
        frames[key], files[key] = seed0(d, n)
    idx = next(iter(frames.values())).index
    for k, f in frames.items():
        if not f.index.equals(idx):
            raise RuntimeError(f"{k}: out-of-fold rows differ from the other models'")
    pairs = {f"{g}:{names[g]} vs {c}": (frames[f"{g}:{names[g]}"], frames[c])
             for g in entries for c in comparators_of(g)}
    table = power_table(S.loc[idx, targets], S.loc[idx, "domain_id"].astype(str).to_numpy(),
                        S.loc[idx, "site_id"].astype(str).to_numpy(), pairs, targets)
    out = {"scheme": a.scheme, "family": "G", "release": {"file": rel["file"], "row_id": rel["row_id"]},
           "power_basis": f"paired site-cluster bootstrap SE of R2 differences on development {a.scheme} "
                          f"(B={N_BOOT}, seed {BOOT_SEED}; 95% percentile-interval width / (2 z_0.975)); largest SE "
                          f"over the pairs; MDD = (z_(1-alpha/2) + z_0.8) * SE, alpha = 0.05/8; holdout scaling "
                          f"sqrt({DEV_EVENTS} / {HOLDOUT_EVENTS}) for every target; underpowered if the "
                          f"holdout-scaled MDD > {THRESHOLD}",
           "entries": {g: names[g] for g in entries}, "not_run": not_run, "comparators": need,
           "prediction_files": files, "n_events": int(len(idx)), "n_pairs": len(pairs), "power": table,
           "underpowered_targets": [t for t in targets if table[t]["underpowered"]]}
    with out_file.open("x") as fh:
        json.dump(out, fh, indent=1, default=float)
        fh.write("\n")
    for t in targets:
        r = table[t]
        if r["underpowered"] is None:
            continue
        for metric, key in (("geo_power_paired_se_site_max", "paired_se_site_max"),
                            ("geo_power_mdd_dev", "mdd_dev_alpha_0.05/8"),
                            ("geo_power_mdd_holdout_scaled", "mdd_holdout_alpha_0.05/8")):
            log_result(workstream="M3-microbe", experiment=EXPERIMENT, model="GEO family comparators", metric=metric,
                       value=r[key], split=f"neon_dev/{a.scheme}_cv", target=t, n=r["n_dev"], is_baseline=False,
                       config={"pair_of_max": r["pair_of_max"], "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                               "alpha": ALPHA, "power": POWER, "underpowered": r["underpowered"],
                               "holdout_scale": r["holdout_scale"], "not_run": not_run, "release_row": rel["row_id"]},
                       data_manifests=AV.MANIFESTS, artifacts=[str(out_file)],
                       notes="GEO power file (amendment §7.3, ruling B9, deviations_01 P1, dev01 rulings D2-D4); "
                             "development rows only")
    print(json.dumps({t: {k: v for k, v in table[t].items() if k != "pairs"} for t in targets}, indent=1,
                     default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
