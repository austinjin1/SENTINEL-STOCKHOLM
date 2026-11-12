# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Seed units and arms of architecture round v2 (orbit_arch_v2_amendment.md §3.2, §3.6, §3.7).

* :func:`assemble` reads the 9 networks of one arm configuration (written by ``scripts/train/orbitv2_arm.py``) and
  refuses an incomplete or inconsistent set.
* :func:`members` builds the reported model (equal-weight average of the 3 unit mixtures), the 3 seed units and the
  9 single networks; the point prediction is the median of the averaged mixture.
* :func:`score` scores them on v1's evaluation cells of the four base targets (v1 masks, v1 metric code).
* :func:`a1_best`, :func:`kill_criterion`, :func:`select_v2` and :func:`units_all_better` are the amendment's
  rules as pure functions of validation scores; :func:`kill_stricter` (R-18a), :func:`per_seed_rule` (R-17) and
  :func:`attribution` (C-5, C-6, C-7) are the lead's rulings on them.

This module computes scores from whatever rows it is given. The stamp gate and the registry are in
``scripts/train/orbitv2_units.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..orbit import data as D
from ..orbit import evaluate as E
from ..orbit import pipeline as PL
from . import train as TR

ORDER = ("a0", "a1_c3", "a1_c2", "a1_c1")      # simpler arm first (§3.6)
A1_CONFIGS = ("a1_c3", "a1_c2", "a1_c1")
TIE_POINTS = 0.05                               # §3.6
KILL_MARGIN_POINTS = 2.0                        # §3.7
KILL_TARGETS = ("chla", "secchi")               # §3.7
SELECT_TARGET = "chla"
SELECT_CELL = PL.PRIMARY_CELL                   # valxval_lake
MIN_ROWS = 20                                   # as scripts/train/orbit_compare.py
SAME = ("targets", "arm_config_hash", "arm_file_sha256", "data", "prediction_rows", "smoke")


class UnitError(RuntimeError):
    pass


def network_dir(root: Path, arm: str, unit: int, init: int) -> Path:
    return Path(root) / arm / f"u{int(unit)}i{int(init)}"


def load_network(run_dir: Path, kind: str = "preds", allow_smoke: bool = False) -> tuple[dict, dict]:
    run_dir = Path(run_dir)
    if not (run_dir / "DONE").is_file():
        raise UnitError(f"{run_dir}: the network is not complete (no DONE file)")
    meta = json.loads((run_dir / "network.json").read_text())
    if meta.get("smoke") and not allow_smoke:
        raise UnitError(f"{run_dir}: smoke-test network; it is not part of a seed unit")
    with np.load(run_dir / f"{kind}.npz") as z:
        out = {k: z[k] for k in ("row_id", "w", "mu", "sg")}
    out["row_id"] = out["row_id"].astype(str)
    return meta, out


def assemble(root: Path, arm: str, kind: str = "preds", allow_smoke: bool = False) -> dict:
    """The 9 networks of ``arm`` under ``root`` (``<root>/<arm>/u<unit>i<init>``), rows in one common order.

    Refused: a missing or unfinished network; a network whose arm, unit, init or seed does not match its folder;
    networks that differ in configuration, data version, targets or prediction rows."""
    ref, rid, nets = None, None, {}
    for u in TR.UNITS:
        for i in TR.INITS:
            d = network_dir(root, arm, u, i)
            meta, z = load_network(d, kind, allow_smoke)
            if (meta.get("arm"), meta.get("unit"), meta.get("init"), meta.get("seed")) != (arm, u, i, TR.network_seed(u, i)):
                raise UnitError(f"{d}: network.json does not describe arm {arm}, unit {u}, init {i}")
            same = {k: meta.get(k) for k in SAME}
            if same["arm_config_hash"] is None or same["data"] is None:
                raise UnitError(f"{d}: network.json lacks the configuration hash or the data record")
            if ref is None:
                ref, rid = same, z["row_id"]
                if len(set(rid)) != len(rid):
                    raise UnitError(f"{d}: duplicate row ids in {kind}.npz")
            else:
                bad = [k for k in SAME if same[k] != ref[k]]
                if bad:
                    raise UnitError(f"{d}: differs from the first network of the arm in {bad}")
                if len(z["row_id"]) != len(rid) or set(z["row_id"]) != set(rid):
                    raise UnitError(f"{d}: prediction rows differ from the first network of the arm")
            ix = pd.Series(np.arange(len(z["row_id"])), index=z["row_id"]).loc[rid].to_numpy()
            mix = tuple(np.asarray(z[k][ix], np.float64) for k in ("w", "mu", "sg"))
            if mix[0].shape[:2] != (len(rid), len(ref["targets"])):
                raise UnitError(f"{d}: mixture shape {mix[0].shape} does not match rows x targets")
            nets[(u, i)] = mix
    return {"arm": arm, "kind": kind, "targets": list(ref["targets"]), "row_id": rid, "nets": nets, "meta": ref}


def unit_mixture(nets: dict, unit: int):
    return TR.average_mixtures([nets[(unit, i)] for i in TR.INITS])


def arm_mixture(nets: dict):
    """Reported model: equal-weight average of the 3 unit mixtures (= equal weight on the 9 networks)."""
    return TR.average_mixtures([unit_mixture(nets, u) for u in TR.UNITS])


def members(asm: dict) -> dict:
    """{name: (level, mixture)}: 'arm', 'unit0'..'unit2', 'u0i0'..'u2i2'."""
    out = {"arm": ("arm", arm_mixture(asm["nets"]))}
    for u in TR.UNITS:
        out[f"unit{u}"] = ("unit", unit_mixture(asm["nets"], u))
    for (u, i), mix in asm["nets"].items():
        out[f"u{u}i{i}"] = ("network", mix)
    return out


def medians(mix) -> np.ndarray:
    """Point prediction [N, T] (log10): the median of the mixture."""
    return TR.mixture_median(mix)[:, :, 0]


def point_predictions(asm: dict, levels=("arm", "unit", "network")) -> dict:
    """{name: (level, [N, T] log10 medians)} over the arm's prediction rows."""
    return {k: (lev, medians(mix)) for k, (lev, mix) in members(asm).items() if lev in levels}


def _rows(asm: dict, row_ids) -> np.ndarray:
    pos = pd.Series(np.arange(len(asm["row_id"])), index=asm["row_id"])
    miss = pd.Index(row_ids).difference(pos.index)
    if len(miss):
        raise UnitError(f"arm {asm['arm']}: {len(miss)} rows of the cell have no prediction")
    return pos.loc[row_ids].to_numpy()


def score(asm: dict, base_df: pd.DataFrame, masks: dict, *, cells=PL.EVAL_CELLS, targets=D.TARGETS,
          levels=("arm", "unit", "network"), other: dict | None = None, n_boot: int = E.N_BOOT):
    """Score an arm on v1 evaluation cells. ``base_df`` and ``masks`` are those of ``pipeline.load_dev`` for the
    base data version, so the rows of a cell are exactly the rows v1 is scored on. ``other``: {name: {target:
    Series(row_id -> log10 prediction)}} of models scored on the same common rows (v1, comparators).

    Returns (table, rows): one line per cell, target and model (MdSA with a role-block cluster bootstrap CI, on the
    rows where every model of the call is finite), and the per-row values behind it."""
    pts = point_predictions(asm, levels)
    tabs, store = [], []
    for cell in [c for c in cells if c in masks]:
        m = np.asarray(masks[cell], bool)
        sub = base_df[m]
        ix = _rows(asm, sub["row_id"].to_numpy().astype(str))
        blocks = sub["block"].to_numpy()
        for t in targets:
            j = asm["targets"].index(t)
            y = np.log10(sub[t].to_numpy(float))
            if np.isfinite(y).sum() < MIN_ROWS:
                continue
            preds = {f"{asm['arm']}:{k}": p[ix, j] for k, (_, p) in pts.items()}
            for name, per_target in (other or {}).items():
                if t in per_target:
                    preds[name] = per_target[t].reindex(sub["row_id"].to_numpy()).to_numpy(float)
            tab, ok = E.score_models(y, preds, blocks, n_boot=n_boot)
            tab.insert(0, "target", t)
            tab.insert(0, "cell", cell)
            tab["level"] = tab["model"].map(lambda k: pts[k.split(":", 1)[1]][0] if k.startswith(asm["arm"] + ":")
                                            else "other")
            tabs.append(tab)
            store.append(pd.DataFrame({"cell": cell, "target": t, "row_id": sub["row_id"].to_numpy()[ok],
                                       "block": blocks[ok], "y": y[ok], **{k: v[ok] for k, v in preds.items()}}))
    if not tabs:
        raise UnitError(f"arm {asm['arm']}: no cell with at least {MIN_ROWS} labelled rows")
    return pd.concat(tabs, ignore_index=True), pd.concat(store, ignore_index=True)


def paired(rows: pd.DataFrame, cell: str, target: str, a: str, b: str, n_boot: int = E.N_BOOT) -> dict:
    """MdSA(a) - MdSA(b) on the common rows of one cell and target (negative = a better), with both MdSA values."""
    r = rows[(rows["cell"] == cell) & (rows["target"] == target)]
    y, pa, pb = (r[c].to_numpy(float) for c in ("y", a, b))
    if not (np.isfinite(pa).all() and np.isfinite(pb).all()) or len(r) < MIN_ROWS:
        raise UnitError(f"{cell}/{target}: {a} and {b} have no common scored rows")
    out = E.paired_mdsa(y, pa, pb, r["block"].to_numpy(), n_boot=n_boot)
    out.update({"mdsa_a": E.mdsa_log10(pa - y), "mdsa_b": E.mdsa_log10(pb - y), "a": a, "b": b})
    return out


# ----------------------------------------------------------------------------------------- rules
def _check_scores(scores: dict, allowed) -> None:
    bad = [a for a in scores if a not in allowed]
    if bad:
        raise UnitError(f"unknown arm configurations {bad}")
    for a, v in scores.items():
        if not np.isfinite(v):
            raise UnitError(f"{a}: validation score is not finite")


def a1_best(chla: dict) -> str:
    """§3.7 / §2.7: the A1 configuration with the lowest validation chl-a MdSA (an exact tie: the simpler one).
    All three A1 configurations must be given: the rule is not applied to a partial set."""
    _check_scores(chla, ORDER)
    miss = [a for a in A1_CONFIGS if a not in chla]
    if miss:
        raise UnitError(f"A1 configurations without a validation score: {miss}")
    return min(A1_CONFIGS, key=lambda a: (chla[a], A1_CONFIGS.index(a)))


def kill_criterion(a1: dict, v1: dict) -> dict:
    """§3.7. ``a1`` and ``v1``: {target: validation MdSA on the common rows of valxval_lake} of the lowest-chl-a A1
    configuration and of v1 as frozen for the headline. A1 passes only if it is >= 2 MdSA points better on both
    chl-a and Secchi."""
    gain = {}
    for t in KILL_TARGETS:
        if t not in a1 or t not in v1 or not (np.isfinite(a1[t]) and np.isfinite(v1[t])):
            raise UnitError(f"kill criterion: no finite validation MdSA of {t} for both A1 and v1")
        gain[t] = float(v1[t] - a1[t])
    return {"passed": bool(all(g >= KILL_MARGIN_POINTS for g in gain.values())), "gain_points": gain,
            "margin_points": KILL_MARGIN_POINTS, "a1": {t: float(a1[t]) for t in KILL_TARGETS},
            "v1": {t: float(v1[t]) for t in KILL_TARGETS}}


def select_v2(chla: dict, a1_passed: bool) -> dict:
    """§3.6: lowest validation chl-a MdSA (3-unit ensemble, valxval_lake) among A0 and, only if A1 passed its kill
    criterion, the A1 configurations. Configurations within 0.05 points of the lowest are tied; the simplest of
    them (order A0, c3, c2, c1) is selected."""
    _check_scores(chla, ORDER)
    if "a0" not in chla:
        raise UnitError("selection needs the A0 validation score")
    if a1_passed:
        miss = [a for a in A1_CONFIGS if a not in chla]
        if miss:
            raise UnitError(f"A1 passed its kill criterion but {miss} have no validation score")
    cands = [a for a in ORDER if a in chla and (a == "a0" or a1_passed)]
    low = min(chla[a] for a in cands)
    tied = [a for a in cands if chla[a] - low <= TIE_POINTS]
    return {"v2": tied[0], "candidates": cands, "tied": tied, "lowest": float(low), "tie_points": TIE_POINTS,
            "a1_passed": bool(a1_passed), "scores": {a: float(chla[a]) for a in cands}}


def arm_score(tab: pd.DataFrame, arm: str, target: str = SELECT_TARGET, cell: str = SELECT_CELL) -> float:
    r = tab[(tab["cell"] == cell) & (tab["target"] == target) & (tab["model"] == f"{arm}:arm")]
    if len(r) != 1:
        raise UnitError(f"{arm}: no {target} score of the 3-unit ensemble on {cell}")
    return float(r["mdsa"].iloc[0])


def evaluate_arms(asms: dict, base_df: pd.DataFrame, masks: dict, v1: dict | None = None, *,
                  n_boot: int = E.N_BOOT) -> dict:
    """Validation scores of the given arms, then the kill criterion and the selection rule when their inputs are
    complete. ``asms``: {arm: assemble(...)}; ``v1``: {target: Series(row_id -> log10 prediction)} of v1 as frozen
    for the headline.

    * Scores: every arm on all labelled rows of each v1 evaluation cell (3-unit ensemble, units, networks).
    * Kill criterion (needs the three A1 configurations and v1): the lowest-chl-a A1 configuration against v1 on
      the common rows of valxval_lake, chl-a and Secchi.
    * Selection (needs A0 and the kill outcome): :func:`select_v2`.
    A rule whose inputs are incomplete is not applied; 'notes' says why."""
    bad = [a for a in asms if a not in ORDER]
    if bad:
        raise UnitError(f"unknown arm configurations {bad}")
    tabs, rows, notes = [], {}, []
    for arm in [a for a in ORDER if a in asms]:
        tab, rows[arm] = score(asms[arm], base_df, masks, n_boot=n_boot)
        tab.insert(0, "arm", arm)
        tabs.append(tab)
    scores = pd.concat(tabs, ignore_index=True)
    chla = {arm: arm_score(scores, arm) for arm in rows}
    kill = selection = None
    if all(a in chla for a in A1_CONFIGS) and v1 is not None:
        miss = [t for t in KILL_TARGETS if t not in v1]
        if miss:
            raise UnitError(f"kill criterion: v1 has no predictions of {miss}")
        best = a1_best(chla)
        _, krows = score(asms[best], base_df, masks, cells=(SELECT_CELL,), targets=KILL_TARGETS, levels=("arm",),
                         other={"v1": {t: v1[t] for t in KILL_TARGETS}}, n_boot=n_boot)
        pr = {t: paired(krows, SELECT_CELL, t, f"{best}:arm", "v1", n_boot=n_boot) for t in KILL_TARGETS}
        kill = kill_criterion({t: pr[t]["mdsa_a"] for t in KILL_TARGETS}, {t: pr[t]["mdsa_b"] for t in KILL_TARGETS})
        kill.update({"a1_configuration": best, "cell": SELECT_CELL, "paired": pr})
        rows["kill"] = krows
    else:
        notes.append("kill criterion not applied: it needs the three A1 configurations and the v1 predictions")
    if "a0" in chla and kill is not None:
        selection = select_v2(chla, kill["passed"])
    else:
        notes.append("selection rule not applied: it needs A0 and the outcome of the A1 kill criterion")
    return {"scores": scores, "rows": rows, "chla": chla, "kill": kill, "selection": selection, "notes": notes}


def units_all_better(unit_scores, comparator: float) -> bool:
    """§3.2 per-seed rule: each of the 3 seed units is individually better than the comparator on the point
    estimate (lower MdSA)."""
    u = [float(x) for x in unit_scores]
    if len(u) != len(TR.UNITS) or not np.isfinite(u).all() or not np.isfinite(comparator):
        raise UnitError("the per-seed rule needs the 3 unit scores and the comparator score")
    return bool(all(x < comparator for x in u))


# ----------------------------------------------------------------------------------------- lead rulings
V1_REFERENCES = ("candidate", "frozen")
PER_SEED_SENTENCE = ("the per-seed rule was applied to 3-network seed units; {k} of 9 single networks individually "
                     "beat every comparator")


def kill_stricter(first: dict, frozen: dict) -> dict:
    """R-18a: the kill criterion first computed against a v1 freeze candidate (``first``) and recomputed against the
    reported v1 model of ``FREEZE_orbit.json`` (``frozen``): the stricter outcome stands, i.e. A1 passes only if it
    passed both times. The returned record is the frozen-reference one with both outcomes named."""
    for k in (first, frozen):
        if not isinstance(k, dict) or "passed" not in k or "gain_points" not in k:
            raise UnitError("R-18a needs the two kill-criterion records")
    if first.get("a1_configuration") != frozen.get("a1_configuration"):
        raise UnitError("R-18a: the two kill-criterion records are of different A1 configurations")
    return {**frozen, "passed": bool(first["passed"] and frozen["passed"]),
            "r18a": {"passed_vs_candidate": bool(first["passed"]), "passed_vs_frozen": bool(frozen["passed"]),
                     "gain_points_vs_candidate": dict(first["gain_points"]),
                     "gain_points_vs_frozen": dict(frozen["gain_points"])}}


def per_seed_rule(unit_scores, network_scores, comparators: dict) -> dict:
    """R-17 for one endpoint. ``unit_scores``: MdSA of the 3 seed units; ``network_scores``: MdSA of the 9 single
    networks; ``comparators``: {name: MdSA} of every claim comparator in its full form, on the same cell. The rule
    holds if each seed unit alone is lower than every comparator. The sentence every v2 label carries is returned
    with ``k`` = the number of single networks that are lower than every comparator."""
    n_net = len(TR.UNITS) * len(TR.INITS)
    nets = [float(x) for x in network_scores]
    comp = {str(k): float(v) for k, v in comparators.items()}
    if len(nets) != n_net or not np.isfinite(nets).all() or not comp or not np.isfinite(list(comp.values())).all():
        raise UnitError(f"the per-seed rule needs the {n_net} single-network scores and every claim comparator score")
    worst = min(comp.values())
    k = int(sum(x < worst for x in nets))
    return {"passed": bool(all(units_all_better(unit_scores, c) for c in comp.values())), "k_networks": k,
            "n_networks": n_net, "unit_scores": [float(x) for x in unit_scores], "best_comparator": worst,
            "comparators": comp, "sentence": PER_SEED_SENTENCE.format(k=k)}


def attribution(a: str, b: str, *, v1_ctx_off: bool, v1_amp: bool) -> list:
    """What a sentence that attributes the difference ``a - b`` must name (lead clarifications C-5, C-6, C-7).
    ``a``, ``b``: arm names of ORDER or ``"v1"``; ``v1_ctx_off`` / ``v1_amp``: the context setting and AMP use of
    the selected v1 configuration."""
    names = (*ORDER, "v1")
    if a not in names or b not in names or a == b:
        raise UnitError(f"attribution of {a!r} - {b!r}: two different models of {names}")
    lo, hi = sorted((a, b), key=names.index)            # "v1" sorts last; a1 after a0
    if lo == "a0" and hi == "v1":
        return ["seed units with EMA weights"] + ([] if v1_amp else ["fp16 AMP"])                     # C-7
    if lo == "a0":                                                                                     # A1 - A0
        out = ["land ring", "multi-target heads", "added rows"]                                        # C-6
        if hi != "a1_c3":
            out.insert(0, "context set")
        if hi == "a1_c1":
            out.append("latitude and longitude")
        if v1_ctx_off:
            out.append("day of year")                                                                  # C-5
        return out
    if hi == "v1":
        return attribution(lo, "a0", v1_ctx_off=v1_ctx_off, v1_amp=v1_amp) + attribution(
            "a0", "v1", v1_ctx_off=v1_ctx_off, v1_amp=v1_amp)
    pair = {("a1_c3", "a1_c2"): ["context set"], ("a1_c3", "a1_c1"): ["context set", "latitude and longitude"],
            ("a1_c2", "a1_c1"): ["latitude and longitude"]}
    return pair[(lo, hi)]
