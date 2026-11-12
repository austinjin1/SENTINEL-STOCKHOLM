# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Project-wide WQP label rule for SENTINEL-Lite round v2 (M7).

The rule is the lead's ("Shared WQP label defect", docs/plans/architecture_research/LEAD_DECISIONS_20261110.md):
labels come from the harmonised, QC-passed ``wqp_discrete`` tables only; a result is a surface result when
``depth_m`` <= 1.0 m or unknown and it is not flagged "bottom"; Secchi depth is exempt; a sample without a QC-passed
surface result is dropped and the reason recorded.

The rule is NOT implemented here. This module only calls M2's reader, :mod:`sentinel.models.orbit.wqp_surface`
(imported, never copied), which in turn calls the unchanged ``sentinel.data.matchups.wqp_surface_samples``:

* :func:`apply_rule` — a Lite matchup table (one row per WQP sample x NAIP chip; ``naip_matchups/national`` and
  ``national_pm7``, both built with ``load_wqp('processed')``) -> the same rows with rule-conforming values, plus the
  dropped rows as keys and reason. It runs ``wqp_surface.apply`` on M2's on-disk side table.
* :func:`check_built` — keys-only check that a built table is the depth-blind pass of that side table (every sample
  key found, same number of member results). No label value is compared, so it may run on every role.
* :func:`surface_samples` / :func:`sample_table` — result rows read from harmonised tables (with ``depth_m``) ->
  surface samples (or all samples with the drop reason) with their keys (``wqp_surface.surface_table`` in memory).
  Used by the P2 index, which reads ``variable=turbidity_fnu`` itself, and by the national-2b label build for the
  4-state chips; it works on a value-present flag as well as on values.

Nothing here builds a label table (see :mod:`.labels2b`). Protected rows: logs and records hold counts only.
"""

from __future__ import annotations

import json

import pandas as pd

from ...data import matchups as M
from ..orbit import wqp_surface as WS

RULE = WS.RULE
DROP_REASONS = WS.DROP_REASONS
KEY_COLS = ("site_id", "date", "target", "characteristic", "unit")
# Lite matchup tables whose WQP rows were read with load_wqp("processed"), i.e. with the two sidecar files and
# without depth. The 4-state table (naip_matchups/matchups.parquet) was built from raw/wqp with the WQX depth fields
# and raw units: its sample keys are not keys of the side table, and it is frozen.
PROCESSED_BUILDS = ("national", "national_pm7")


class LabelRuleError(RuntimeError):
    """The WQP label rule cannot be applied to a Lite table as ruled."""


def keyed(samples: pd.DataFrame, name: str) -> pd.DataFrame:
    """Copy of ``samples`` with ``sample_key`` (M2's expression) and ``_set`` = ``name``. A stored ``sample_key``
    that differs from the expression is refused (the key is the only link to the side table)."""
    miss = [c for c in KEY_COLS if c not in samples.columns]
    if miss:
        raise LabelRuleError(f"{name}: key column(s) {miss} missing")
    out = samples.copy()
    key = WS.sample_key(out.assign(unit=out["unit"].fillna("").astype(str)))
    if "sample_key" in out.columns and not (out["sample_key"].astype(str).to_numpy() == key.to_numpy()).all():
        n = int((out["sample_key"].astype(str).to_numpy() != key.to_numpy()).sum())
        raise LabelRuleError(f"{name}: {n} stored sample keys differ from the matchup builds' key expression")
    out["sample_key"] = key.to_numpy()
    out["_set"] = name
    return out


def load_side(keys=None) -> tuple[pd.DataFrame, str]:
    """(M2's on-disk side table, its sha256). ``wqp_surface.load_side`` checks the table against its meta file, the
    rule text and the input files; every rule must be ruled. ``keys``: keep these sample keys only."""
    WS.require_rulings()
    side = WS.load_side()
    sha = json.loads((WS.side_dir() / WS.META_FILE).read_text())["side_table_sha256"]
    if keys is not None:
        side = side[side["sample_key"].isin(pd.unique(pd.Series(keys)))].reset_index(drop=True)
    return side, sha


def check_built(samples: pd.DataFrame, name: str, side: pd.DataFrame) -> dict:
    """Keys only: is the built table the depth-blind pass of the side table? Counts of sample rows whose key is
    missing and whose ``n_results`` differs from the side table's ``n_results_all``. Raises on either."""
    k = keyed(samples, name)
    ref = side.drop_duplicates("sample_key").set_index("sample_key")
    found = k["sample_key"].isin(ref.index).to_numpy()
    n_all = ref["n_results_all"].reindex(k["sample_key"]).to_numpy(float)
    differs = found & (pd.to_numeric(k["n_results"], errors="coerce").to_numpy(float) != n_all)
    out = {"n_rows": int(len(k)), "n_keys": int(k["sample_key"].nunique()), "n_key_missing": int((~found).sum()),
           "n_results_differs": int(differs.sum())}
    if out["n_key_missing"] or out["n_results_differs"]:
        raise LabelRuleError(f"{name}: the built table is not the depth-blind pass of the side table: {out}")
    return out


def apply_rule(samples: pd.DataFrame, name: str, *, side: pd.DataFrame | None = None, side_sha256: str | None = None,
               log=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(kept rows with the side table's values, dropped rows as keys + reason, record).

    ``samples``: rows of a Lite matchup table of :data:`PROCESSED_BUILDS` (several rows may share a sample key: one
    per NAIP chip). ``side=None`` reads M2's on-disk side table. Call it on a whole build table, not on a role or a
    target subset: ``wqp_surface.apply`` refuses a table whose non-Secchi rows all lack a depth."""
    k = keyed(samples, name)
    if side is None:
        side, side_sha256 = load_side(k["sample_key"])
    check = check_built(samples, name, side)
    kept = WS.apply(k, log=log, side=side)
    applied = WS.last_applied()
    ref = side.drop_duplicates("sample_key").set_index("sample_key")
    reason = ref["drop_reason"].reindex(k["sample_key"]).to_numpy()
    dropped = k.loc[reason != "", [*KEY_COLS, "sample_key"]].copy()
    dropped["drop_reason"] = reason[reason != ""]
    fewer = (ref["n_results"].reindex(kept["sample_key"]).to_numpy() <
             ref["n_results_all"].reindex(kept["sample_key"]).to_numpy())
    by = {}
    for t, g in k.assign(_why=reason).groupby("target"):
        by[str(t)] = {"n_rows": int(len(g)), **{f"dropped_{w}": int((g["_why"] == w).sum()) for w in DROP_REASONS}}
    for t, n in kept.assign(_f=fewer).groupby("target")["_f"].sum().items():
        by[str(t)]["kept_with_fewer_results"] = int(n)
    record = {"table": name, "rule": RULE, "rulings": dict(WS.RULINGS), "side_table_sha256": side_sha256,
              "reader": "sentinel.models.orbit.wqp_surface.apply", "built_table_check": check,
              "n_rows": int(len(k)), "n_rows_kept": int(len(kept)), "n_rows_dropped": int(len(dropped)),
              "by_target": by, "orbit_record": applied}
    kept = kept.drop(columns=["_set"])
    return kept, dropped.reset_index(drop=True), record


def sample_table(results: pd.DataFrame, *, target: str | None = None) -> pd.DataFrame:
    """Harmonised, QC-passed result rows with ``depth_m`` -> one row per sample of those rows (``KEY_COLS``,
    ``sample_key``, ``surface``, ``drop_reason``, ``wqp_surface.VALUE_COLS``). ``surface`` is False and
    ``drop_reason`` is ``no_surface_result`` when the sample has QC-passed results but none at the surface. A
    sample that is absent from the output has no QC-passed result at all. Rows of the sidecar files must not be
    passed. ``target``: see :func:`surface_samples`."""
    if "depth_m" not in results.columns:
        raise LabelRuleError("result rows carry no depth_m column: the depth rule cannot be applied")
    res = results.copy()
    if target is not None:
        if target not in M.WQP_TARGETS:
            raise LabelRuleError(f"unknown target {target!r}")
        res["variable"] = target
    cols = [*KEY_COLS, "sample_key", "surface", "drop_reason", *WS.VALUE_COLS]
    if not len(res):
        return pd.DataFrame(columns=cols)
    side = WS.surface_table(res)
    # the side table carries the key as one string; the key columns come from the rows themselves
    r = res.copy()
    char = r["characteristic"].astype(str)
    r["target"] = char.map(M.NAME_TO_TARGET)
    if "variable" in r.columns:
        r["target"] = r["target"].fillna(r["variable"].where(r["variable"].isin(M.WQP_TARGETS)))
    r["date"] = pd.to_datetime(r["date"] if "date" in r.columns else r["activity_start_date"],
                               errors="coerce").dt.normalize()
    r["unit"] = r["unit"].fillna("").astype(str) if "unit" in r.columns else ""
    r["site_id"] = r["site_id"].astype(str)
    r = r[r["target"].notna() & r["date"].notna()][list(KEY_COLS)].drop_duplicates()
    r["sample_key"] = WS.sample_key(r).to_numpy()
    out = r.merge(side[["sample_key", "surface", "drop_reason", *WS.VALUE_COLS]], on="sample_key", how="inner")
    if len(out) != len(side):
        raise LabelRuleError(f"{len(side) - len(out)} samples could not be joined back to their key columns")
    return out[cols].sort_values("sample_key").reset_index(drop=True)


def surface_samples(results: pd.DataFrame, *, target: str | None = None) -> pd.DataFrame:
    """Harmonised, QC-passed result rows with ``depth_m`` -> one row per surface sample under the rule, with the key
    columns of :data:`KEY_COLS` and ``wqp_surface.VALUE_COLS``. Rows of the sidecar files must not be passed.

    ``target``: name the target when the rows carry neither a target characteristic nor a ``variable`` column value
    in ``matchups.WQP_TARGETS`` (the P2 index reads ``variable=turbidity_fnu`` and passes ``"turbidity"``).
    ``value`` may be a present/absent flag (1.0 / NaN): then ``value`` of the output is a flag as well."""
    t = sample_table(results, target=target)
    return t[t["surface"].astype(bool)][[*KEY_COLS, "sample_key", *WS.VALUE_COLS]].reset_index(drop=True)


def reason_counts(dropped: pd.DataFrame, by=("target",)) -> dict:
    """Counts of dropped rows by ``by`` and reason (for manifests and registry rows; no value)."""
    if not len(dropped):
        return {}
    g = dropped.groupby([*by, "drop_reason"]).size()
    return {"|".join(map(str, k if isinstance(k, tuple) else (k,))): int(n) for k, n in g.items()}


__all__ = ["RULE", "DROP_REASONS", "KEY_COLS", "PROCESSED_BUILDS", "LabelRuleError", "keyed", "load_side",
           "check_built", "apply_rule", "sample_table", "surface_samples", "reason_counts"]
