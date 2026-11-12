#!/usr/bin/env python
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Binding power calculation of the Test-S look (orbit_arch_v2_amendment.md §1.5, §1.7, §2.6;
orbit_arch_v2_lead_rulings.md R-15, R-16, R-18, R-18b).

    python -B scripts/train/orbitv2_power.py --val-rows <val_rows.parquet> --counts-json <split report> --register

A design calculation: validation per-row predictions plus COUNTS of the protected Test-S cell. No protected
label, prediction or statistic is read. Every status-setting MDE is computed once (the output folder is refused
if it holds a file), behind ``require_stamp``.

Method (R-16). ``orbit_power.block_draws`` of the v1 script is called unchanged: n_sim 2,000, seed 0 at the
validation block count and seed 1 at the block count of the primary Test-S cell fixed by amendment §1.6. The
standard error is the SD of the seed-1 draws. MDE80 = (z(1 - alpha/2) + 0.84) standard errors, with alpha the
smallest Holm alpha of the hypothesis's family (0.05 / divisor). For each target the largest MDE80 over every
scored arm and every claim comparator is the binding one (R-15).

Inputs.
* ``--val-rows``: one row per (target, cell, validation row) with a quantified label. Columns: ``target``,
  ``cell``, ``block`` (role block), ``y`` (log10 label), ``small_lake`` (bool), ``ntu_class`` (bool, read for
  turbidity only), ``arm:<name>`` for each scored arm (``arm:v1``, ``arm:v2``, ``arm:track_n``) and
  ``cmp:<name>`` for each claim comparator (log10 predictions). The rows of a target in its comparator
  selection cell are used: ``orbit_compare.PRIMARY_TARGET_CELL`` for the base targets; for TP, TN and DOC
  ``valxval_lake`` if it holds >= 200 quantified rows in >= 20 role blocks, otherwise ``valxany_lake``
  (amendment §2.5).
* ``--counts-json``: ``{"test_s_primary_cell": "scene_disjoint" | "all_rows", "test_s_primary_counts":
  {"all": {target: {"n", "n_waterbodies", "n_blocks"}}, "small_lake": {...}}}``. Turbidity counts are those of
  NTU-class rows (R-18).

Gates (applied by this code, no discretion):
* H1-S (chl-a; Holm over the 2 arms, divisor 2): no gate; MDE80 reported.
* H2-S (Secchi, TSS, turbidity-NTU; divisor 6 whatever is declared descriptive): a target is confirmatory only
  with >= 30 role blocks in its primary Test-S cell and a binding MDE80 <= 15 MdSA points (R-18).
* Track N (TP, TN, DOC; divisor 3 whatever is declared descriptive): a binding MDE80 > 15 points makes the
  target descriptive (§2.6).
* Small lakes (4 targets; divisor ``SMALL_LAKE_DIVISOR``): confirmatory only if the binding chl-a MDE80 of the
  stratum is <= 10 points (§1.7, R-15).
* R-18b: an endpoint whose MDE80 exceeds the arm's validation gain against any claim comparator is recorded as
  under-powered for that arm.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.models.orbit import evaluate as E  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402
from sentinel.models.orbit_v2 import stamp as ST  # noqa: E402

EXPERIMENT = "orbit_power_S_dev_v5m"            # amendment §1.5
V1_POWER_SCRIPT = "scripts/train/orbit_power.py"
ALPHA = 0.05
Z_POWER = 0.84                                  # R-16
N_SIM = 2000                                    # R-16
SEED_VAL, SEED_TEST = 0, 1                      # R-16
MIN_VAL_ROWS = 20                               # as scripts/train/orbit_compare.py
MIN_BLOCKS = 2                                  # as scripts/train/orbit_power.py

BASE_CELL = {"chla": PL.PRIMARY_CELL, "secchi": PL.PRIMARY_CELL, "tss": "valxany_lake", "turbidity": "valxany_lake"}
TRACK_N_CELLS = ("valxval_lake", "valxany_lake")        # §2.5 count rule, in this order
TRACK_N_MIN_ROWS, TRACK_N_MIN_BLOCKS = 200, 20

BASE_COMPARATORS = ("mdn_b", "gbm", "knn", "mdn_b+gbm")                                   # §6.2 + R-10
TRACK_N_COMPARATORS = ("mdn_b", "gbm", "knn", "season_loc_gbm", "trophic_ols", "mdn_b+gbm")  # §2.5 + R-10
BASE_ARMS = ("v1", "v2")                        # §6.1
TRACK_N_ARM = "track_n"                         # §2.7: the multi-target arm

# Small lakes: 4 targets x 4 claim comparators (the 3 of H1 and the MDN_B + GBM average, R-10) x 2 arms (R-15:
# Holm over both arms' hypotheses together).
SMALL_LAKE_DIVISOR = 32

FAMILIES = {
    "H1-S": {"divisor": 2, "targets": ("chla",), "stratum": "all", "comparators": BASE_COMPARATORS,
             "arms": BASE_ARMS},
    "H2-S": {"divisor": 6, "targets": ("secchi", "tss", "turbidity"), "stratum": "all",
             "comparators": BASE_COMPARATORS, "arms": BASE_ARMS, "max_mde": 15.0, "min_blocks": 30},
    "TrackN": {"divisor": 3, "targets": ("tp", "tn", "doc"), "stratum": "all",
               "comparators": TRACK_N_COMPARATORS, "arms": (TRACK_N_ARM,), "max_mde": 15.0},
    "small_lakes": {"divisor": SMALL_LAKE_DIVISOR, "targets": ("chla", "secchi", "tss", "turbidity"),
                    "stratum": "small_lake", "comparators": BASE_COMPARATORS, "arms": BASE_ARMS,
                    "max_mde": 10.0, "gate_targets": ("chla",)},
}
REQUIRED = ("target", "cell", "block", "y", "small_lake", "ntu_class")


class PowerError(RuntimeError):
    pass


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_block_draws():
    """``block_draws`` of the v1 power script, imported from its file and not copied (R-16: "unchanged")."""
    spec = importlib.util.spec_from_file_location("orbit_power_v1", REPO / V1_POWER_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.block_draws


def holm_alpha(family: str) -> float:
    """Smallest Holm alpha of the family: 0.05 / divisor. The divisors do not change with any status."""
    return ALPHA / FAMILIES[family]["divisor"]


def mde_factor(alpha: float) -> float:
    return float(norm.ppf(1.0 - alpha / 2.0) + Z_POWER)


def selection_cell(vp: pd.DataFrame, target: str) -> str:
    if target in BASE_CELL:
        return BASE_CELL[target]
    d = vp[(vp.target == target) & (vp.cell == TRACK_N_CELLS[0])]
    ok = len(d) >= TRACK_N_MIN_ROWS and d.block.nunique() >= TRACK_N_MIN_BLOCKS
    return TRACK_N_CELLS[0] if ok else TRACK_N_CELLS[1]


def cell_rows(vp: pd.DataFrame, target: str, stratum: str) -> pd.DataFrame:
    """Validation rows of the target and stratum in the target's comparator selection cell."""
    d = vp[(vp.target == target) & (vp.cell == selection_cell(vp, target))]
    if stratum == "small_lake":
        d = d[d.small_lake.astype(bool)]
    elif stratum != "all":
        raise PowerError(f"unknown stratum {stratum!r}")
    if target == "turbidity":
        d = d[d.ntu_class.astype(bool)]             # R-18: NTU-class rows only
    return d.reset_index(drop=True)


def check_table(vp: pd.DataFrame) -> None:
    miss = [c for c in REQUIRED if c not in vp.columns]
    if miss:
        raise PowerError(f"validation-row table lacks columns {miss}")
    if not np.all(np.isfinite(vp.y.to_numpy(float))):
        raise PowerError("validation-row table holds non-finite labels; quantified rows only")
    if "row_id" in vp.columns and vp.duplicated(["target", "cell", "row_id"]).any():
        raise PowerError("validation-row table repeats a (target, cell, row)")


def target_power(vp: pd.DataFrame, counts: dict, family: str, target: str, arms, block_draws,
                 n_sim: int = N_SIM) -> list[dict]:
    """One record per (arm, comparator) for a target of a family."""
    fam = FAMILIES[family]
    alpha = holm_alpha(family)
    k = mde_factor(alpha)
    cnt = (counts.get(fam["stratum"]) or {}).get(target)
    if cnt is None:
        raise PowerError(f"no protected count for target {target!r}, stratum {fam['stratum']!r}")
    bt = int(cnt["n_blocks"])
    d = cell_rows(vp, target, fam["stratum"])
    comps = [f"cmp:{c}" for c in fam["comparators"]]
    acols = [f"arm:{a}" for a in arms]
    miss = [c for c in comps + acols if c not in d.columns]
    if miss:
        raise PowerError(f"{family} {target}: validation-row table lacks {miss}")
    cell = selection_cell(vp, target)
    base = {"family": family, "target": target, "stratum": fam["stratum"], "val_cell": cell, "alpha_min": alpha,
            "mde_factor": k, "n_val_rows": int(len(d)), "n_blocks_val": int(d.block.nunique()),
            "n_test_rows": int(cnt["n"]), "n_waterbodies_test": int(cnt["n_waterbodies"]), "n_blocks_test": bt}
    note = None
    if len(d) < MIN_VAL_ROWS or d.block.nunique() < MIN_BLOCKS:
        note = "validation cell too small"
    elif bt < MIN_BLOCKS:
        note = "test cell too small"
    if note:
        return [{**base, "arm": a, "comparator": c, "note": note} for a in arms for c in fam["comparators"]]
    bad = int((~np.isfinite(d[comps + acols].to_numpy(float))).sum())
    if bad:
        raise PowerError(f"{family} {target}: {bad} non-finite predictions on the selection-cell rows")
    y = d.y.to_numpy(float)
    out = []
    for a in arms:
        dd = d.drop(columns=[c for c in d.columns if c == "SENTINEL"]).rename(columns={f"arm:{a}": "SENTINEL"})
        dv = block_draws(dd, comps, base["n_blocks_val"], n_sim, SEED_VAL)
        dt = block_draws(dd, comps, bt, n_sim, SEED_TEST)
        se_v, se_t = dv.std(0, ddof=1), dt.std(0, ddof=1)
        ms = E.mdsa_log10(dd["SENTINEL"].to_numpy(float) - y)
        for j, c in enumerate(fam["comparators"]):
            diff = ms - E.mdsa_log10(dd[comps[j]].to_numpy(float) - y)
            mde = float(k * se_t[j])
            out.append({**base, "arm": a, "comparator": c, "val_diff": float(diff), "se_val": float(se_v[j]),
                        "se_test": float(se_t[j]), "mde80_test": mde,
                        "underpowered_at_val_gain": bool(mde > -diff), "note": ""})
    return out


def family_status(rows: list[dict], family: str, arms_by_target: dict) -> dict:
    """Status of each target of a family from its records. Binding MDE80 = largest over arms and comparators."""
    fam = FAMILIES[family]
    tab = pd.DataFrame(rows)
    out = {}
    for t in fam["targets"]:
        arms = tuple(arms_by_target[t])
        rec = {"alpha_min": holm_alpha(family), "holm_divisor": fam["divisor"], "arms": list(arms)}
        if not arms:
            out[t] = {**rec, "status": "no SENTINEL claim", "reason": "no arm is scored on this target"}
            continue
        r = tab[(tab.family == family) & (tab.target == t)] if len(tab) else tab
        if not len(r):
            raise PowerError(f"{family} {t}: no record")
        want = len(arms) * len(fam["comparators"])
        mde = r["mde80_test"].to_numpy(float) if "mde80_test" in r.columns else np.full(len(r), np.nan)
        rec.update(n_blocks_test=int(r.n_blocks_test.iloc[0]), n_test_rows=int(r.n_test_rows.iloc[0]),
                   n_val_rows=int(r.n_val_rows.iloc[0]), val_cell=str(r.val_cell.iloc[0]))
        if len(r) != want or not np.all(np.isfinite(mde)):
            notes = sorted({str(x) for x in r["note"].dropna() if str(x)}) if "note" in r.columns else []
            out[t] = {**rec, "binding_mde80": None, "status": "descriptive",
                      "reason": "; ".join(notes) or "an MDE80 is missing"}
            continue
        i = int(np.argmax(mde))
        rec.update(binding_mde80=float(mde[i]), binding_arm=str(r.arm.iloc[i]),
                   binding_comparator=str(r.comparator.iloc[i]),
                   underpowered_arms=sorted(r.arm[r.underpowered_at_val_gain.astype(bool)].unique().tolist()))
        if "max_mde" not in fam:
            out[t] = {**rec, "status": "primary", "reason": "no gate (amendment §6.2)"}
            continue
        ok = rec["binding_mde80"] <= fam["max_mde"]
        why = f"binding MDE80 {'<=' if ok else '>'} {fam['max_mde']:g} MdSA points"
        if "min_blocks" in fam and rec["n_blocks_test"] < fam["min_blocks"]:
            ok, why = False, f"fewer than {fam['min_blocks']} role blocks in the primary Test-S cell; " + why
        out[t] = {**rec, "status": "confirmatory" if ok else "descriptive", "reason": why}
    gate = fam.get("gate_targets")
    if gate:                                    # one status for the whole family, set by the gate targets
        ok = all(out[t]["status"] == "confirmatory" for t in gate)
        why = f"family status set by {', '.join(gate)} (amendment §1.7): " + out[gate[0]]["reason"]
        for t in fam["targets"]:
            out[t].update(own_gate=out[t]["status"], status="confirmatory" if ok else "descriptive", reason=why)
    return out


def run(vp: pd.DataFrame, counts: dict, *, track_n_arm: bool = True, dropped_heads=(), block_draws=None,
        n_sim: int = N_SIM) -> tuple[pd.DataFrame, dict]:
    """All families. ``track_n_arm`` False: A1 failed its kill criterion (§2.7), Track N has no SENTINEL claim.
    ``dropped_heads``: Track N targets whose head was dropped by the C-4 count rule (no SENTINEL result)."""
    check_table(vp)
    block_draws = block_draws or load_block_draws()
    rows, status = [], {}
    for family, fam in FAMILIES.items():
        frows, arms_by_target = [], {}
        for t in fam["targets"]:
            none = family == "TrackN" and (not track_n_arm or t in dropped_heads)
            arms_by_target[t] = () if none else fam["arms"]
            if arms_by_target[t]:
                frows += target_power(vp, counts, family, t, arms_by_target[t], block_draws, n_sim)
        st = family_status(frows, family, arms_by_target)
        if family == "TrackN":
            for t in fam["targets"]:
                if not track_n_arm:
                    st[t]["reason"] = "A1 failed its kill criterion: comparators only (amendment §2.7)"
                elif t in dropped_heads:
                    st[t]["reason"] = "head dropped by the count rule: counts only (C-4)"
        rows += frows
        status[family] = st
    return pd.DataFrame(rows), status


def read_counts(path: Path) -> tuple[str, dict]:
    rp = json.loads(Path(path).read_text())
    cell = rp.get("test_s_primary_cell")
    if cell not in ("scene_disjoint", "all_rows"):
        raise PowerError("counts file does not state the primary Test-S cell fixed by amendment §1.6")
    counts = rp.get("test_s_primary_counts")
    if not isinstance(counts, dict) or "all" not in counts or "small_lake" not in counts:
        raise PowerError("counts file lacks test_s_primary_counts for the strata 'all' and 'small_lake'")
    return cell, counts


def register(res: pd.DataFrame, status: dict, out_dir: Path, primary: str, stamp_row: str) -> int:
    from sentinel.core.registry import log_result

    common = {"workstream": PL.WORKSTREAM, "experiment": EXPERIMENT, "is_baseline": False,
              "data_manifests": ["orbit_dev_v5m"], "artifacts": [str(out_dir)]}
    note = ("design calculation: validation per-row predictions + Test-S primary-cell COUNTS only (no protected "
            f"label or prediction); orbit_power.block_draws unchanged; amendment stamp row {stamp_row}")
    n = 0
    for r in res.dropna(subset=["mde80_test"]).itertuples() if "mde80_test" in res.columns else []:
        split = f"M2S primary cell ({primary}, {r.stratum}; counts only) / {r.val_cell} (val predictions)"
        for met in ("mde80_test", "se_test"):
            log_result(model=r.arm, metric=f"{r.family}:{met}_vs_{r.comparator}", value=float(getattr(r, met)),
                       target=r.target, split=split, n=int(r.n_test_rows), notes=note,
                       extra={"val_diff": r.val_diff, "n_blocks_val": int(r.n_blocks_val),
                              "n_blocks_test": int(r.n_blocks_test), "alpha_min": r.alpha_min,
                              "mde_factor": r.mde_factor, "underpowered_at_val_gain": bool(r.underpowered_at_val_gain)},
                       **common)
            n += 1
    for family, st in status.items():
        for t, s in st.items():
            log_result(model="all scored arms", metric=f"{family}:confirmatory",
                       value=float(s["status"] in ("confirmatory", "primary")), target=t,
                       split=f"M2S primary cell ({primary}; counts only)", n=s.get("n_test_rows"),
                       notes=f"status {s['status']}: {s['reason']}. " + note,
                       extra={k: v for k, v in s.items() if k not in ("status", "reason")}, **common)
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--val-rows", required=True, help="validation-row table (see the module text)")
    ap.add_argument("--counts-json", required=True, help="dev_v5m split report with the Test-S primary-cell counts")
    ap.add_argument("--out", default=None, help=f"output folder (default <runs>/orbit/v2/{EXPERIMENT})")
    ap.add_argument("--no-track-n-arm", action="store_true", help="A1 failed its kill criterion (amendment §2.7)")
    ap.add_argument("--dropped-heads", nargs="*", default=[], choices=list(FAMILIES["TrackN"]["targets"]))
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args(argv)
    st = ST.require_stamp("orbitv2_power: binding MDE80 of the Test-S look")
    out = Path(a.out) if a.out else paths.RUNS_ROOT / "orbit" / "v2" / EXPERIMENT
    if out.exists() and any(out.iterdir()):
        raise PowerError(f"{out} holds files: each status-setting MDE is computed once (R-16)")
    primary, counts = read_counts(Path(a.counts_json))
    vp = pd.read_parquet(a.val_rows)
    res, status = run(vp, counts, track_n_arm=not a.no_track_n_arm, dropped_heads=tuple(a.dropped_heads))
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "power_S.csv", index=False)
    rec = {"experiment": EXPERIMENT, "test_s_primary_cell": primary, "status": status,
           "n_sim": N_SIM, "seeds": [SEED_VAL, SEED_TEST], "z_power": Z_POWER, "alpha": ALPHA,
           "holm_divisors": {f: v["divisor"] for f, v in FAMILIES.items()},
           "sha256": {"scripts/train/orbitv2_power.py": sha256(Path(__file__)),
                      V1_POWER_SCRIPT: sha256(REPO / V1_POWER_SCRIPT),
                      "val_rows": sha256(Path(a.val_rows)), "counts_json": sha256(Path(a.counts_json))},
           "stamp_row": st.get("row_id"), "track_n_arm": not a.no_track_n_arm, "dropped_heads": list(a.dropped_heads)}
    (out / "status.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    pd.set_option("display.width", 250)
    cols = [c for c in ("family", "target", "arm", "comparator", "n_blocks_val", "n_blocks_test", "val_diff",
                        "se_test", "mde80_test", "underpowered_at_val_gain", "note") if c in res.columns]
    print(res[cols].round(3).to_string(index=False))
    for family, s in status.items():
        for t, v in s.items():
            print(f"{family:12s} {t:10s} {v['status']:13s} binding MDE80 {v.get('binding_mde80')}  ({v['reason']})")
    if a.register:
        print(f"registered {register(res, status, out, primary, str(st.get('row_id')))} rows under {EXPERIMENT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
