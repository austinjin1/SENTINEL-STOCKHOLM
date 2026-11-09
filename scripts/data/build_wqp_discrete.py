#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the analysis-ready WQP discrete dataset from the raw parsed chunks.

Input:  $SENTINEL_DATA/raw/wqp/{results,physchem}/... (scripts/data/download_wqp.py)
Output: $SENTINEL_DATA/processed/wqp_discrete/
          variable=<name>/data.parquet   harmonised + QC'd rows, one canonical unit per variable
          _stations.parquet              sites referenced by the processed rows
          _unmapped.parquet              rows that could not be mapped/converted (+ reason)
          _qc_rejects.parquet            rows removed by QC (+ first failing step)
          _qc_report.csv / .json         per-variable counts before/after every step
          _unmapped_summary.csv          unmapped counts by reason x characteristic x unit
          _coverage_by_variable.csv, _coverage_by_state.csv, _coverage_by_year.csv

Every target chunk is processed independently (a variable's site-dates never span two raw
chunks), then physchem rows are kept only where a QC-passed target result exists at the same
site on the same local date.

    python -B scripts/data/build_wqp_discrete.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import wqp_harmonize as H  # noqa: E402
from sentinel.data import wqp_qc as Q  # noqa: E402
from sentinel.data.sources import wqp as W  # noqa: E402

log = logging.getLogger("wqp_build")

OUT_SCHEMA = pa.schema([
    ("site_id", pa.string()), ("state", pa.string()), ("huc8", pa.string()),
    ("lat", pa.float64()), ("lon", pa.float64()), ("site_type", pa.string()),
    ("site_class", pa.string()),
    ("activity_start_date", pa.date32()), ("activity_start_time", pa.string()),
    ("activity_tz", pa.string()), ("time_utc", pa.timestamp("us", tz="UTC")),
    ("depth_m", pa.float32()), ("activity_relative_depth", pa.string()),
    ("value", pa.float64()), ("unit", pa.string()), ("censored", pa.bool_()),
    ("censor_type", pa.string()), ("detection_limit", pa.float64()),
    ("detection_limit_type", pa.string()),
    ("value_original", pa.float64()), ("unit_original", pa.string()),
    ("characteristic", pa.string()), ("sample_fraction", pa.string()),
    ("fraction_class", pa.string()), ("species_used", pa.string()),
    ("speciation_assumed", pa.bool_()),
    ("method_id", pa.string()), ("method_context", pa.string()),
    ("activity_type", pa.string()), ("media_subdivision", pa.string()),
    ("hydrologic_event", pa.string()), ("hydrologic_condition", pa.string()),
    ("provider", pa.string()), ("organization", pa.string()), ("project_id", pa.string()),
    ("activity_id", pa.string()), ("result_id", pa.string()), ("status", pa.string()),
    ("status_class", pa.string()), ("qualifier", pa.string()), ("value_type", pa.string()),
    ("statistical_base", pa.string()), ("result_comment", pa.string()),
    ("last_change_date", pa.string()), ("usgs_pcode", pa.string()),
])
UNMAPPED_COLS = ["unmapped_reason", "variable", "characteristic", "sample_fraction", "speciation",
                 "unit", "value", "value_text", "censored", "detection_limit",
                 "detection_limit_unit", "method_id", "method_name", "activity_type", "site_id",
                 "state", "activity_start_date", "organization", "provider", "result_id", "group"]
REJECT_COLS = ["qc_reason", "variable", "characteristic", "value", "unit", "unit_original",
               "censored", "censor_type", "detection_limit", "value_text", "value_flag",
               "activity_type", "statistical_base", "time_basis", "status", "site_id", "state",
               "activity_start_date", "activity_start_time", "organization", "provider",
               "result_id"]


def _str_table(df: pd.DataFrame, cols: list[str]) -> pa.Table:
    d = {}
    for c in cols:
        s = df[c] if c in df.columns else pd.Series([None] * len(df), index=df.index)
        if s.dtype == bool or c in ("censored",):
            d[c] = pa.array(s.fillna(False).astype(bool).to_numpy())
        elif pd.api.types.is_float_dtype(s):
            d[c] = pa.array(s.to_numpy(dtype="float64"))
        else:
            d[c] = pa.array(s.fillna("").astype(str).to_numpy(dtype=object), type=pa.string())
    return pa.table(d)


class Writers:
    """One ParquetWriter per variable (row group per chunk), plus side files."""

    def __init__(self, out: Path):
        self.out = out
        self.w: dict[str, pq.ParquetWriter] = {}
        self.side: dict[str, pq.ParquetWriter] = {}
        self.n: dict[str, int] = {}

    def write_var(self, var: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        t = to_out_table(df)
        if var not in self.w:
            d = paths.ensure(self.out / f"variable={var}")
            self.w[var] = pq.ParquetWriter(d / "data.parquet", OUT_SCHEMA, compression="zstd")
        self.w[var].write_table(t, row_group_size=1_000_000)
        self.n[var] = self.n.get(var, 0) + len(df)

    def write_side(self, name: str, df: pd.DataFrame, cols: list[str]) -> None:
        if df.empty:
            return
        t = _str_table(df, cols)
        if name not in self.side:
            self.side[name] = pq.ParquetWriter(self.out / f"_{name}.parquet", t.schema, compression="zstd")
        self.side[name].write_table(t.cast(self.side[name].schema))

    def close(self) -> None:
        for w in list(self.w.values()) + list(self.side.values()):
            w.close()


def to_out_table(df: pd.DataFrame) -> pa.Table:
    arrays = []
    for f in OUT_SCHEMA:
        c = f.name
        s = df[c] if c in df.columns else pd.Series([None] * len(df), index=df.index)
        if f.type == pa.string():
            arrays.append(pa.array(s.fillna("").astype(str).to_numpy(dtype=object), type=pa.string()))
        elif f.type == pa.bool_():
            arrays.append(pa.array(s.fillna(False).astype(bool).to_numpy()))
        elif f.type == pa.date32():
            d = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
            arrays.append(pa.array(d.dt.date.where(d.notna(), None).to_numpy(dtype=object), type=pa.date32()))
        elif pa.types.is_timestamp(f.type):
            ts = pd.to_datetime(s, utc=True, errors="coerce")
            arrays.append(pa.array(ts, type=f.type))
        else:
            arrays.append(pa.array(pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")).cast(f.type))
    return pa.Table.from_arrays(arrays, schema=OUT_SCHEMA)


def load_stations(raw: Path) -> pd.DataFrame:
    p = raw / "stations.parquet"
    if p.exists():
        st = pd.read_parquet(p)
    else:
        files = sorted((raw / "locations").glob("*.parquet"))
        st = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=["site_id"])
    st = st.drop_duplicates("site_id")
    # WQX3 leaves Location_StatePostalCode blank for some sites; Station/search was queried one
    # statecode at a time, so the file name gives the state.
    sdir = raw / "stations_by_state"
    if sdir.exists():
        req = pd.concat([pd.read_parquet(f, columns=["site_id"]).assign(req_state=f.stem)
                         for f in sorted(sdir.glob("*.parquet"))], ignore_index=True)
        req = req.drop_duplicates("site_id").set_index("site_id")["req_state"]
        if "state" not in st.columns:
            st["state"] = ""
        blank = st["state"].fillna("") == ""
        st.loc[blank, "state"] = st.loc[blank, "site_id"].map(req).fillna("")
    st["site_class"] = Q.site_class(st.get("site_type", pd.Series("", index=st.index)))
    return st


def apply_secchi_bottom(m: pd.DataFrame, u: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Secchi readings whose activity also reports 'disk visible at bottom = Yes' become
    right-censored (true depth >= reading)."""
    aux = u[u["unmapped_reason"] == "aux_secchi_visible_at_bottom"]
    if aux.empty or m.empty:
        return m, 0
    yes = aux["value_text"].fillna("").str.strip().str.lower().isin(["y", "yes", "true", "1"])
    acts = set(aux.loc[yes, "activity_id"])
    hit = (m["variable"] == "secchi") & m["activity_id"].isin(acts) & ~m["censored"] & m["value"].notna()
    if hit.any():
        m = m.copy()
        m.loc[hit, "detection_limit"] = m.loc[hit, "value"]
        m.loc[hit, "detection_limit_type"] = "Secchi disk visible at bottom"
        m.loc[hit, "value"] = np.nan
        m.loc[hit, "censored"] = True
        m.loc[hit, "censor_type"] = ">"
    return m, int(hit.sum())


def add_site_fields(df: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("site_id", "huc8", "lat", "lon", "site_type", "site_class") if c in st.columns]
    d = df.drop(columns=[c for c in cols if c != "site_id" and c in df.columns])
    d = d.merge(st[cols], on="site_id", how="left")
    if "state" in st.columns:
        miss = d["state"].fillna("") == ""
        if miss.any():
            d.loc[miss, "state"] = d.loc[miss, "site_id"].map(st.set_index("site_id")["state"])
    return d


def _md(df: pd.DataFrame, floatfmt: str = "{:,.3g}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, row):
            if "year" in c and isinstance(v, (int, np.integer, float, np.floating)) and v == v:
                cells.append(str(int(v)))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,}")
            elif isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    cells.append("")
                elif float(v).is_integer() and abs(v) >= 1000:
                    cells.append(f"{int(v):,}")
                else:
                    cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


KEY_VARS = ("chla_corrected", "secchi", "tss", "turbidity_ntu", "tp", "tn", "no3no2", "chloride",
            "spcond", "ecoli", "microcystins", "temp")


def write_tables_md(out: Path, cv: pd.DataFrame, cs: pd.DataFrame, qc: pd.DataFrame,
                    us: pd.DataFrame, comp: pd.DataFrame) -> None:
    """Markdown tables for docs/datasheets/wqp_discrete.md (regenerated by every build)."""
    parts = ["## Coverage by variable\n", _md(cv[["variable", "unit", "rows", "sites", "states",
                                                 "first_year", "last_year", "censored_frac",
                                                 "usgs_frac", "median_value"]])]
    if len(cs):
        tot = cs.groupby("state").agg(rows=("rows", "sum")).reset_index()
        sites = cs.groupby("state")["sites"].max().rename("max_sites_any_variable").reset_index()
        piv = cs[cs["variable"].isin(KEY_VARS)].pivot_table(index="state", columns="variable",
                                                            values="rows", aggfunc="sum", fill_value=0)
        piv = piv.reindex(columns=[v for v in KEY_VARS if v in piv.columns]).reset_index()
        st = tot.merge(sites, on="state").merge(piv, on="state", how="left").fillna(0)
        for c in st.columns[1:]:
            st[c] = st[c].astype("int64")
        parts += ["\n\n## Coverage by state (rows; key variables)\n", _md(st)]
    q = qc.reset_index()
    parts += ["\n\n## QC counts per step (rows removed)\n", _md(q)]
    if len(us):
        r = us.groupby("unmapped_reason")["rows"].sum().sort_values(ascending=False).reset_index()
        parts += ["\n\n## Unmapped rows by reason\n", _md(r)]
        top = us.head(40)
        parts += ["\n\n## Top unmapped (reason x characteristic x unit)\n", _md(top)]
    if len(comp):
        c = comp.copy()
        c["period"] = np.where(c["year"] >= 2024, "2024-present", "2000-2023")
        g = c.groupby(["group", "period"])[["wqx3_rows", "legacy_total"]].sum().reset_index()
        g["ratio"] = (g["wqx3_rows"] / g["legacy_total"]).round(3)
        parts += ["\n\n## Completeness: WQX3 rows vs legacy WQX 2.2 summary\n", _md(g)]
    (out / "_datasheet_tables.md").write_text("\n".join(parts) + "\n")


def completeness_table(raw: Path) -> pd.DataFrame:
    """WQX3 rows downloaded per group x year (ledger) vs the legacy WQX 2.2 summary counts
    (all media; lacks USGS data added after 2024-03-11). Physchem compares downloaded rows."""
    led = raw / "ledger.jsonl"
    if not led.exists():
        return pd.DataFrame()
    recs = [json.loads(x) for x in led.read_text().splitlines() if x.strip()]
    done = {}
    for r in recs:
        if r.get("status") == "done" and r.get("phase") in ("results", "physchem"):
            done[r["id"]] = r
    rows = []
    for r in done.values():
        g = r.get("group", "physchem")
        n = r.get("rows_downloaded", r.get("rows", 0))
        rows.append({"group": g, "year": int(r["start"][:4]), "wqx3_rows": n})
    got = pd.DataFrame(rows).groupby(["group", "year"], as_index=False)["wqx3_rows"].sum()
    files = sorted((raw / "summary").glob("*.parquet"))
    if not files:
        return got
    sm = pd.concat([pd.read_parquet(f, columns=["characteristic", "year", "provider", "n_results"])
                    for f in files], ignore_index=True)
    sm["group"] = sm["characteristic"].map(W.name_to_group())
    sm = sm.dropna(subset=["group", "year"])
    sm["year"] = sm["year"].astype(int)
    ref = sm.pivot_table(index=["group", "year"], columns="provider", values="n_results",
                         aggfunc="sum", fill_value=0).reset_index()
    ref.columns = [c if c in ("group", "year") else f"legacy_{c.lower()}" for c in ref.columns]
    ref["legacy_total"] = ref[[c for c in ref.columns if c.startswith("legacy_")]].sum(axis=1)
    t = ref.merge(got, on=["group", "year"], how="outer").fillna(0)
    t = t[t["year"] >= 2000].sort_values(["group", "year"])
    t["ratio_wqx3_to_legacy"] = (t["wqx3_rows"] / t["legacy_total"].where(t["legacy_total"] > 0)).round(3)
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit-chunks", type=int, default=0, help="debug: process only N chunks")
    a = ap.parse_args()
    raw = paths.raw("wqp")
    out = paths.processed("wqp_discrete")
    for old in list(out.glob("variable=*/data.parquet")) + list(out.glob("_*")):
        old.unlink()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out.parent / "wqp_discrete_build.log"),
                                  logging.StreamHandler()])
    t0 = time.time()
    st = load_stations(raw)
    log.info("stations: %d", len(st))
    state_of = st.set_index("site_id")["state"].replace("", np.nan).dropna()
    writers = Writers(out)
    report = Q.QCReport()
    states_ok = set(W.STATES)
    n_raw = n_unmapped = n_secchi_bottom = n_out_of_scope = n_state_unknown = 0
    unmapped_summary: dict[tuple, int] = {}
    target_sd: set[tuple[str, str]] = set()

    def process(files: list[Path], phase: str) -> None:
        nonlocal n_raw, n_unmapped, n_secchi_bottom, n_out_of_scope, n_state_unknown
        for i, f in enumerate(files, 1):
            df = pd.read_parquet(f)
            n_raw += len(df)
            blank = df["state"].fillna("") == ""
            if blank.any():
                df.loc[blank, "state"] = df.loc[blank, "site_id"].map(state_of).fillna("")
            # every request was filtered to the 52 statecodes: rows whose state is still unknown
            # are in scope and kept with state ""
            scope = df["state"].isin(states_ok) | (df["state"] == "")
            n_out_of_scope += int((~scope).sum())
            n_state_unknown += int((df["state"] == "").sum())
            df = df[scope]
            m, u = H.harmonize(df)
            m, k = apply_secchi_bottom(m, u)
            n_secchi_bottom += k
            n_unmapped += len(u)
            if len(u):
                g = u.groupby(["unmapped_reason", "characteristic", "unit"], dropna=False).size()
                for key, n in g.items():
                    unmapped_summary[key] = unmapped_summary.get(key, 0) + int(n)
                writers.write_side("unmapped", u, UNMAPPED_COLS)
            kept, rej = Q.qc_chunk(m, report, co_sampled=target_sd if phase == "physchem" else None)
            if len(rej):
                writers.write_side("qc_rejects", rej, REJECT_COLS)
            if phase == "results" and len(kept):
                target_sd.update(zip(kept["site_id"], kept["activity_start_date"]))
            kept = add_site_fields(kept, st)
            for var, part in kept.groupby("variable", sort=False):
                writers.write_var(var, part)
            if i % 50 == 0 or i == len(files):
                log.info("%s: %d/%d chunks, raw rows so far %d (%.1f min)", phase, i, len(files),
                         n_raw, (time.time() - t0) / 60)

    rfiles = sorted(raw.glob("results/group=*/*.parquet"))
    pfiles = sorted(raw.glob("physchem/*.parquet"))
    if a.limit_chunks:
        rfiles, pfiles = rfiles[: a.limit_chunks], pfiles[: a.limit_chunks]
    process(rfiles, "results")
    process(pfiles, "physchem")
    writers.close()

    # Sort each variable file by state/site/date for efficient filtering; collect coverage.
    cov_var, cov_state, cov_year = [], [], []
    used_sites: set[str] = set()
    for vdir in sorted(out.glob("variable=*")):
        var = vdir.name.split("=", 1)[1]
        p = vdir / "data.parquet"
        t = pq.read_table(p)
        t = t.sort_by([("state", "ascending"), ("site_id", "ascending"),
                       ("activity_start_date", "ascending")])
        pq.write_table(t, p, compression="zstd", row_group_size=1_000_000)
        d = t.select(["site_id", "state", "activity_start_date", "censored", "provider",
                      "value"]).to_pandas()
        used_sites.update(d["site_id"].unique())
        yr = pd.to_datetime(d["activity_start_date"]).dt.year
        cov_var.append({"variable": var, "unit": VARS_UNIT.get(var, ""), "rows": len(d),
                        "sites": d["site_id"].nunique(), "states": d["state"].nunique(),
                        "first_year": int(yr.min()), "last_year": int(yr.max()),
                        "years_with_data": int(yr.nunique()),
                        "censored_frac": round(float(d["censored"].mean()), 4),
                        "usgs_frac": round(float((d["provider"] == "USGS").mean()), 4),
                        "median_value": float(np.nanmedian(d["value"])) if d["value"].notna().any() else np.nan})
        g = d.groupby("state").agg(rows=("site_id", "size"), sites=("site_id", "nunique"))
        g["variable"] = var
        cov_state.append(g.reset_index())
        gy = d.assign(year=yr).groupby("year").agg(rows=("site_id", "size"), sites=("site_id", "nunique"))
        gy["variable"] = var
        cov_year.append(gy.reset_index())
        del t, d
    cv = pd.DataFrame(cov_var).sort_values("variable")
    cv.to_csv(out / "_coverage_by_variable.csv", index=False)
    cs = pd.concat(cov_state, ignore_index=True) if cov_state else pd.DataFrame()
    cs.to_csv(out / "_coverage_by_state.csv", index=False)
    cy = pd.concat(cov_year, ignore_index=True) if cov_year else pd.DataFrame()
    cy.to_csv(out / "_coverage_by_year.csv", index=False)

    st_used = st[st["site_id"].isin(used_sites)].copy()
    for c in st_used.columns:
        if st_used[c].dtype == object:
            st_used[c] = st_used[c].fillna("").astype(str)
    st_used.to_parquet(out / "_stations.parquet", index=False, compression="zstd")

    comp = completeness_table(raw)
    if len(comp):
        comp.to_csv(out / "_completeness_by_group_year.csv", index=False)
    qc = report.table()
    qc.to_csv(out / "_qc_report.csv")
    us = pd.DataFrame([{"unmapped_reason": k[0], "characteristic": k[1], "unit": k[2], "rows": n}
                       for k, n in unmapped_summary.items()]).sort_values("rows", ascending=False)
    us.to_csv(out / "_unmapped_summary.csv", index=False)
    write_tables_md(out, cv, cs, qc, us, comp)
    totals = {
        "raw_rows_read": n_raw, "rows_outside_52_jurisdictions": n_out_of_scope,
        "rows_state_unknown_kept": n_state_unknown,
        "unmapped_rows": n_unmapped, "secchi_set_right_censored_by_bottom_flag": n_secchi_bottom,
        "qc_input_rows": int(qc["input"].sum()), "processed_rows": int(cv["rows"].sum()) if len(cv) else 0,
        "qc_removed_by_step": {s: int(qc[s].sum()) for s in Q.QC_STEPS if s in qc},
        "physchem_not_co_sampled": int(qc["not_co_sampled"].sum()) if "not_co_sampled" in qc else 0,
        "unmapped_by_reason": us.groupby("unmapped_reason")["rows"].sum().sort_values(ascending=False).to_dict() if len(us) else {},
        "n_sites": len(used_sites), "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "plausible_ranges": {k: [v.lo, v.hi, v.unit] for k, v in H.VARIABLES.items()},
    }
    (out / "_qc_report.json").write_text(json.dumps({"totals": totals, "by_variable": report.to_dict()},
                                                   indent=1, default=str))
    files = sorted(out.glob("variable=*/data.parquet")) + sorted(out.glob("_*"))
    m = write_manifest(
        "wqp_discrete", files=files,
        sources=["manifest:wqp_raw", W.RESULT_URL, W.STATION_URL],
        description="Harmonised, QC'd WQP (WQX 3.0) discrete water-quality samples, 52 jurisdictions, "
                    "2000-present; one canonical unit per variable (see docs/datasheets/wqp_discrete.md); "
                    "partitioned by variable; physchem only where co-sampled with a target result.",
        license="US public domain / open government data (WQP; data owned by submitting organizations)",
        stats={**totals, "coverage_by_variable": cv.to_dict(orient="records")})
    log.info("done in %.1f min: %d processed rows, manifest %s", (time.time() - t0) / 60,
             totals["processed_rows"], m)
    return 0


VARS_UNIT = {k: v.unit for k, v in H.VARIABLES.items()}

if __name__ == "__main__":
    raise SystemExit(main())
