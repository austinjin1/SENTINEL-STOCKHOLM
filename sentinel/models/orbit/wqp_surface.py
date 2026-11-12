# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""WQP surface labels with the depth rule applied (lead ruling 2026-11-10; orbit_protocol.md change log).

Cause 1 (depth). ``sentinel.data.matchups.wqp_surface_samples`` documents "depth <= 1 m or unspecified (Secchi
exempt)" and reads the WQX depth fields ``result_depth`` / ``activity_depth`` / ``activity_top_depth``. The
processed ``wqp_discrete`` tables carry one harmonised column ``depth_m`` instead, and ``load_wqp('processed')``
does not keep it. Every depth was therefore "unspecified" when the Sentinel-2 matchup tables were built: a stored
``samples.parquet`` value is the median over all results that are not flagged "bottom", deep ones included.

Cause 2 (sidecar files). ``load_wqp('processed')`` recognises result tables by their columns, so it also reads
``wqp_discrete/_qc_rejects.parquet`` (rows REMOVED by the WQP QC: blanks, spikes, replicates, logger values and
aggregates, rejected status, out-of-range values, duplicates) and ``_unmapped.parquet`` (rows that could not be
mapped to a harmonised variable, e.g. the ambiguous characteristic "Chlorophyll"). Those rows have no ``depth_m``,
``qc_class`` or ``status_class`` column, so the matchup builds used them as ordinary surface results.

Fix (M2-owned code; the shared module is not edited).
* Depth-blind pass: the unchanged ``load_wqp`` + ``wqp_surface_samples``, i.e. what the matchup builds computed. It
  defines the key set and must reproduce every built ``samples.parquet`` exactly (:func:`check_reproduces`).
* Label pass: the harmonised, QC-passed tables only (``variable=*/``; the sidecar files in :data:`SIDECARS` are
  left out), with ``depth_m`` handed to the UNCHANGED ``wqp_surface_samples`` as ``result_depth`` in metres. The
  documented rule is applied by the documented function and the grouping keys are the matchup builds' own.
* The side table (:func:`build`) has one row per sample of the depth-blind pass. ``surface`` is False when the
  sample has no QC-passed surface result (``drop_reason`` says which).
* :func:`apply` replaces the values of a matchup sample table by the side table's and drops the samples without a
  label. It refuses to run unless EVERY sample key is found (nothing is defaulted to "unknown depth") and unless
  every rule in :data:`RULINGS` has been ruled by the lead.

Rule (one rule for every target, role and table): a result is a surface result when its depth is <= 1.0 m or
unknown, and it is not flagged "bottom"; Secchi depth is exempt. Sample label = median of the QC-passed surface
results over (site, date, target, characteristic, unit).

Protected rows: values are computed mechanically; logs and reports hold counts only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...data import matchups as M

SIDE_NAME = "orbit_wqp_surface"
SIDE_FILE = "surface_samples.parquet"
META_FILE = "surface_samples.meta.json"
VALUE_COLS = ("value", "value_min", "value_max", "n_results", "n_censored", "censored", "detection_limit",
              "sample_depth_m", "depth_rule")
# Files of processed/wqp_discrete that are NOT harmonised, QC-passed result tables (scripts/data/build_wqp_discrete.py:
# "rows removed by QC" and "rows that could not be mapped/converted"). They never contribute to a label.
SIDECARS = ("_qc_rejects.parquet", "_unmapped.parquet")
RULE = ("surface result: depth_m <= 1.0 m, or depth unknown; a result flagged 'bottom' is never a surface result, "
        "at any depth; Secchi exempt (sentinel.data.matchups.wqp_surface_samples, unchanged). Results come from the "
        "harmonised, QC-passed wqp_discrete tables only (variable=*/); rows of _qc_rejects.parquet and "
        "_unmapped.parquet never contribute. Sample label: median of those results over (site_id, date, target, "
        "characteristic, unit). A sample without such a result is dropped (reason no_qc_passed_result or "
        "no_surface_result).")
# Lead rulings this module implements. None = reported, not ruled yet: apply() refuses to run on the real tables,
# which holds the dev_v5 build (orbit_protocol.md change log).
RULINGS = {
    "depth_rule": "lead ruling 2026-11-10 on the WQP depth finding (option A)",
    "sidecar_exclusion": "lead ruling 2026-11-10 13:21 EST: approved as written (labels from the harmonised, "
                         "QC-passed tables only; the unchanged function's bottom-flag behaviour is the rule; the "
                         "ambiguous 'Chlorophyll' rows stay out)",
}
DROP_REASONS = ("no_qc_passed_result", "no_surface_result")

_LAST: dict = {}


def last_applied() -> dict:
    """Record of the last :func:`apply` call of this process (rule, side-table sha256, counts by table/target)."""
    return json.loads(json.dumps(_LAST))


class SurfaceRuleError(RuntimeError):
    """The label rule cannot be applied as ruled (missing side table, unmatched sample keys, depth never read,
    a rule the lead has not ruled on)."""


def side_dir() -> Path:
    return paths.PROCESSED / SIDE_NAME


def sample_key(df: pd.DataFrame) -> pd.Series:
    """The ``sample_key`` of scripts/data/build_s2_wqp_matchups.py (same expression)."""
    return (df["site_id"].astype(str) + "|" + pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d") + "|" +
            df["target"].astype(str) + "|" + df["characteristic"].astype(str) + "|" + df["unit"].astype(str))


def _rel(f) -> str:
    return Path(f).relative_to(paths.PROCESSED).as_posix()


def _read_filtered(f, extra=()) -> pd.DataFrame | None:
    """One result table read as ``load_wqp`` reads it (same columns plus ``extra``, same row filter); None when the
    file is not a result table."""
    import pyarrow.parquet as pq

    cols = set(pq.ParquetFile(f).schema_arrow.names)
    if not ({"site_id", "value"} <= cols and ({"characteristic", "variable"} & cols)):
        return None
    names = [n for t in M.WQP_TARGETS for n in M.WQP_TARGETS[t]]
    keep = [c for c in (*M.RESULT_KEEP, *extra) if c in cols]
    key, vals = ("characteristic", names) if "characteristic" in cols else ("variable", list(M.WQP_TARGETS))
    try:
        df = pq.read_table(f, columns=keep, filters=[(key, "in", vals)]).to_pandas()
    except Exception:  # noqa: BLE001 — same fallback as load_wqp
        df = pq.read_table(f, columns=keep).to_pandas()
        df = df[df[key].isin(vals)]
    return df


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(all rows as the matchup builds read them, harmonised QC-passed rows with ``depth_m``, rows by file)."""
    res_all, _stations, used = M.load_wqp("processed")
    harm, rows = [], {}
    for f in sorted(used):
        df = _read_filtered(f, extra=("depth_m",))
        if df is None or not len(df):
            continue
        rows[_rel(f)] = int(len(df))
        if Path(f).name in SIDECARS:
            continue
        if "depth_m" not in df.columns:
            raise SurfaceRuleError(f"{_rel(f)}: harmonised result table without a depth_m column")
        harm.append(df)
    if sum(rows.values()) != len(res_all):
        raise SurfaceRuleError(f"own read ({sum(rows.values())} rows) differs from load_wqp ({len(res_all)} rows)")
    if not harm:
        raise SurfaceRuleError("no harmonised result rows")
    return res_all, pd.concat(harm, ignore_index=True), rows


def _no_wqx_depth(res: pd.DataFrame, what: str) -> None:
    known = [c for c in ("result_depth", "activity_depth", "activity_top_depth") if c in res.columns
             and pd.to_numeric(res[c], errors="coerce").notna().any()]
    if known:
        raise SurfaceRuleError(f"unexpected schema ({what}): WQX depth fields {known} are populated")


def surface_table(harm: pd.DataFrame, res_all: pd.DataFrame | None = None) -> pd.DataFrame:
    """Side table (see the module docstring).

    ``harm``: harmonised, QC-passed result rows with ``depth_m``. ``res_all``: every row the matchup builds read
    (sidecar rows included, no depth); default: ``harm`` itself. Three passes of the unchanged
    ``wqp_surface_samples``: depth-blind over ``res_all`` (key set; what the builds computed), depth-blind over
    ``harm`` (which samples have a QC-passed result) and with ``depth_m`` as ``result_depth`` [m] (the labels)."""
    if "depth_m" not in harm.columns:
        raise SurfaceRuleError("result rows carry no depth_m column")
    plain = harm.drop(columns=["depth_m"])
    res_all = plain if res_all is None else res_all.drop(columns=["depth_m"], errors="ignore")
    _no_wqx_depth(plain, "harmonised rows")
    _no_wqx_depth(res_all, "all rows")
    blind = M.wqp_surface_samples(res_all, None)
    if blind["sample_depth_m"].notna().any():
        raise SurfaceRuleError("depth-blind pass returned depths")
    qc = blind if res_all is plain else M.wqp_surface_samples(plain, None)
    surf = M.wqp_surface_samples(plain.assign(result_depth=pd.to_numeric(harm["depth_m"], errors="coerce"),
                                              result_depth_unit="m"), None)
    tabs = {"depth-blind": blind, "QC-passed": qc, "surface": surf}
    for name, t in tabs.items():
        if "sample_key" not in t.columns:
            t["sample_key"] = sample_key(t)
        if t["sample_key"].duplicated().any():
            raise SurfaceRuleError(f"{name} pass: duplicate sample keys")
    if not surf["sample_key"].isin(qc["sample_key"]).all() or not qc["sample_key"].isin(blind["sample_key"]).all():
        raise SurfaceRuleError("a pass produced a sample key that the pass before it lacks")
    out = blind[["sample_key", "target", "date", "n_results", "value"]].rename(
        columns={"n_results": "n_results_all", "value": "_value_all"})
    out["n_results_qc"] = out["sample_key"].map(qc.set_index("sample_key")["n_results"]).fillna(0).astype("int32")
    out = out.merge(surf[["sample_key", *VALUE_COLS]], on="sample_key", how="left", indicator=True)
    out["surface"] = (out.pop("_merge") == "both").to_numpy()
    out["n_results"] = out["n_results"].fillna(0).astype("int32")
    out["n_results_all"] = out["n_results_all"].astype("int32")
    out["n_censored"] = out["n_censored"].fillna(0).astype("int32")
    out["censored"] = out["censored"].astype("boolean").fillna(False).astype(bool)
    out["drop_reason"] = np.where(out["surface"], "", np.where(out["n_results_qc"] == 0, *DROP_REASONS))
    out["depth_rule"] = out["depth_rule"].where(out["surface"], out["drop_reason"]).astype(str)
    return out.sort_values("sample_key").reset_index(drop=True)


def bottom_flag_counts(harm: pd.DataFrame) -> dict:
    """Result-row counts for the one place where the unchanged function is stricter than "depth <= 1 m": a result
    flagged "bottom" is never a surface result, whatever its depth. Depth-exempt targets are left out."""
    tgt = harm["characteristic"].map(M.NAME_TO_TARGET) if "characteristic" in harm.columns else harm["variable"]
    rel = harm.get("activity_relative_depth", pd.Series("", index=harm.index)).fillna("").astype(str).str.lower()
    bottom = rel.str.contains("bottom").to_numpy() & ~tgt.isin(M.DEPTH_EXEMPT).to_numpy()
    d = pd.to_numeric(harm["depth_m"], errors="coerce").to_numpy(float)
    out = {}
    for t in sorted(set(tgt[bottom])):
        m = bottom & (tgt == t).to_numpy()
        out[str(t)] = {"n_results_flagged_bottom": int(m.sum()),
                       "n_flagged_bottom_depth_le_1m": int((m & (d <= M.SURFACE_MAX_M)).sum()),
                       "n_flagged_bottom_depth_unknown": int((m & np.isnan(d)).sum())}
    return out


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def input_fingerprint(files) -> dict:
    """{relative path: [size, mtime_ns]} of the wqp_discrete files read (cheap staleness check)."""
    return {_rel(f): [Path(f).stat().st_size, Path(f).stat().st_mtime_ns] for f in sorted(files)}


def build(log=print) -> dict:
    """Build the side table from the processed wqp_discrete tables and write it with its meta file."""
    res_all, harm, rows = load_results()
    files = [paths.PROCESSED / f for f in rows]
    n_side = sum(n for f, n in rows.items() if Path(f).name in SIDECARS)
    log(f"[wqp_surface] {len(res_all)} result rows of the matchup targets in {len(rows)} files: {len(harm)} "
        f"harmonised QC-passed (depth_m known for {int(harm['depth_m'].notna().sum())}), {n_side} in the sidecar "
        f"files {list(SIDECARS)} (left out of the labels)")
    bottom = bottom_flag_counts(harm)
    side = surface_table(harm, res_all)
    del res_all, harm
    d = side_dir()
    d.mkdir(parents=True, exist_ok=True)
    reproduces = check_reproduces(side, log)
    side = side.drop(columns=["_value_all"])
    tmp = d / (SIDE_FILE + ".tmp")
    side.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(d / SIDE_FILE)
    non_secchi = side[~side["target"].isin(M.DEPTH_EXEMPT) & side["surface"]]

    def cnt(g):
        return {"n_samples": int(len(g)), "n_no_qc_passed_result": int((g["drop_reason"] == DROP_REASONS[0]).sum()),
                "n_no_surface_result": int((g["drop_reason"] == DROP_REASONS[1]).sum()),
                "n_kept": int(g["surface"].sum()),
                "n_kept_with_fewer_results": int((g["surface"] & (g["n_results"] < g["n_results_all"])).sum()),
                "n_kept_depth_known": int((g["surface"] & g["sample_depth_m"].notna()).sum())}

    meta = {"rule": RULE, "rulings": RULINGS, "surface_max_m": M.SURFACE_MAX_M, "depth_exempt": list(M.DEPTH_EXEMPT),
            "sidecars_left_out": list(SIDECARS), "result_rows_by_file": rows, "n_result_rows_sidecar": int(n_side),
            **cnt(side), "n_kept_non_secchi": int(len(non_secchi)),
            "n_kept_non_secchi_depth_known": int(non_secchi["sample_depth_m"].notna().sum()),
            "by_target": {t: cnt(g) for t, g in side.groupby("target")},
            "results_flagged_bottom": bottom,
            "reproduces_built_tables": reproduces,
            "input_files": input_fingerprint(files),
            "input_sha256": {_rel(f): _sha_file(f) for f in files},
            "side_table_sha256": _sha_file(d / SIDE_FILE),
            "matchups_py_sha256": _sha_file(Path(M.__file__))}
    (d / META_FILE).write_text(json.dumps(meta, indent=1) + "\n")
    log(f"[wqp_surface] wrote {d / SIDE_FILE}: {meta['n_samples']} samples; {meta['n_no_qc_passed_result']} without "
        f"a QC-passed result, {meta['n_no_surface_result']} without a surface result")
    return meta


def check_reproduces(side: pd.DataFrame, log=print) -> dict:
    """The depth-blind pass must reproduce every built ``samples.parquet`` (keys, n_results, value): then the only
    differences between a built table and the side table are the rules of this module. Counts only."""
    from .data import WQP_MATCHUP_SETS

    ref = side.set_index("sample_key")
    out = {}
    for name in WQP_MATCHUP_SETS:
        f = paths.PROCESSED / name / "samples.parquet"
        if not f.exists():
            continue
        s = pd.read_parquet(f, columns=["sample_key", "value", "n_results"])
        found = s["sample_key"].isin(ref.index)
        r = ref.reindex(s["sample_key"][found])
        v0, v1 = s["value"][found].to_numpy(float), r["_value_all"].to_numpy(float)
        same_v = np.isclose(v0, v1, rtol=1e-9, atol=0.0, equal_nan=True)
        same_n = s["n_results"][found].to_numpy() == r["n_results_all"].to_numpy()
        out[name] = {"n_samples": int(len(s)), "n_key_missing": int((~found).sum()),
                     "n_value_differs": int((~same_v).sum()), "n_results_differs": int((~same_n).sum())}
        log(f"[wqp_surface] {name}: depth-blind pass vs built table: {out[name]}")
        if out[name]["n_key_missing"] or out[name]["n_value_differs"] or out[name]["n_results_differs"]:
            raise SurfaceRuleError(f"{name}: the depth-blind pass does not reproduce the built table: {out[name]}")
    return out


def load_side(check_inputs: bool = True) -> pd.DataFrame:
    d = side_dir()
    if not ((d / SIDE_FILE).exists() and (d / META_FILE).exists()):
        raise SurfaceRuleError(f"side table {d / SIDE_FILE} missing: run scripts/train/orbit_build_wqp_surface.py")
    meta = json.loads((d / META_FILE).read_text())
    if _sha_file(d / SIDE_FILE) != meta["side_table_sha256"]:
        raise SurfaceRuleError("side table does not match its meta file")
    if meta.get("rule") != RULE or meta.get("sidecars_left_out") != list(SIDECARS):
        raise SurfaceRuleError("side table was built under a different rule: rebuild it")
    if check_inputs:
        now = input_fingerprint([paths.PROCESSED / f for f in meta["input_files"]
                                 if (paths.PROCESSED / f).exists()])
        if now != meta["input_files"]:
            raise SurfaceRuleError("processed wqp_discrete tables changed since the side table was built: rebuild it "
                                   "(scripts/train/orbit_build_wqp_surface.py)")
    return pd.read_parquet(d / SIDE_FILE)


def check_depth_read(smp: pd.DataFrame, what: str = "table") -> None:
    """Guard: a built table whose non-Secchi rows ALL lack a sample depth was built without reading depth."""
    sub = smp[~smp["target"].isin(M.DEPTH_EXEMPT)]
    by = sub["_set"].to_numpy() if "_set" in sub.columns else np.full(len(sub), what, dtype=object)
    for name, g in sub.groupby(by):
        if len(g) and g["sample_depth_m"].isna().all():
            raise SurfaceRuleError(f"{name}: sample_depth_m is null for all {len(g)} non-Secchi rows — the depth "
                                   "rule was not applied")


def require_rulings() -> None:
    open_ = [k for k, v in RULINGS.items() if not v]
    if open_:
        raise SurfaceRuleError(f"label rule(s) {open_} reported to the lead but not ruled yet: WQP labels are not "
                               "built until they are (the dev_v5 build is held)")


def apply(smp: pd.DataFrame, log=print, side: pd.DataFrame | None = None) -> pd.DataFrame:
    """Matchup samples (``samples.parquet`` rows) -> the same rows with rule-conforming values; samples without a
    QC-passed surface result are dropped. Raises unless every sample key is in the side table. With the on-disk
    side table (``side=None``) it also raises while a rule in :data:`RULINGS` is not ruled."""
    sha = None
    if side is None:
        require_rulings()
        side = load_side()
        sha = json.loads((side_dir() / META_FILE).read_text())["side_table_sha256"]
    if side["sample_key"].duplicated().any():
        raise SurfaceRuleError("side table has duplicate sample keys")
    ref = side.set_index("sample_key")
    by = smp["_set"].to_numpy() if "_set" in smp.columns else np.full(len(smp), "table", dtype=object)
    found = smp["sample_key"].isin(ref.index).to_numpy()
    if not found.all():
        miss = pd.Series(by[~found]).value_counts().to_dict()
        raise SurfaceRuleError(f"{int((~found).sum())} sample keys are not in the side table (by table: {miss}); "
                               "nothing is defaulted to unknown depth")
    r = ref.reindex(smp["sample_key"])
    out = smp.copy()
    for c in VALUE_COLS:
        out[c] = r[c].to_numpy()
    keep = r["surface"].to_numpy(bool)
    reason = r["drop_reason"].to_numpy() if "drop_reason" in r.columns else np.where(keep, "", DROP_REASONS[1])
    counts = {}
    for (name, t), g in pd.DataFrame({"s": by, "t": smp["target"].to_numpy(), "k": keep, "why": reason,
                                      "fewer": (r["n_results"] < r["n_results_all"]).to_numpy() & keep}
                                     ).groupby(["s", "t"]):
        counts[f"{name}/{t}"] = {"n": int(len(g)), **{f"dropped_{w}": int((g["why"] == w).sum())
                                                       for w in DROP_REASONS},
                                 "kept_with_fewer_results": int(g["fewer"].sum())}
    out = out[keep].copy()
    check_depth_read(out)
    log(f"[data] wqp label rule: {int(keep.sum())} of {len(keep)} samples have a QC-passed surface result; {counts}")
    _LAST.clear()
    _LAST.update({"rule": RULE, "side_table_sha256": sha, "n_samples": int(len(keep)),
                  "n_with_surface_result": int(keep.sum()), "by_table_target": counts})
    return out
