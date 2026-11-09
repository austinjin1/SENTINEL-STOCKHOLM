#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build processed NEON aquatic datasets (+ manifests + datasheets) from raw('neon').

    python -B scripts/data/build_neon.py sites microbe chem          # Priority 1 observational
    python -B scripts/data/build_neon.py sonde nitrate --workers 4    # Priority 1 sensors
    python -B scripts/data/build_neon.py tsw tsd eos discharge macroinv periphyton fish \
        zooplankton qpcr metagenome                                   # Priority 2
    python -B scripts/data/build_neon.py all

Targets write ``processed('neon_<name>')``, ``manifests/neon_<name>.json`` and
``docs/datasheets/neon_<name>.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Shared-host rule (docs/CONTRIBUTING_V2.md): one CPU-heavy job per workstream, <= 6 threads.
# Each sensor worker process is single-threaded (BLAS/OpenMP/Arrow = 1); <= 4 workers + main.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402

pa.set_cpu_count(1)
pa.set_io_thread_count(1)

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import neon_aquatic as na  # noqa: E402
from sentinel.data.sources import neon  # noqa: E402

DOCS = paths.REPO_ROOT / "docs" / "datasheets"
TODAY = dt.date.today().isoformat()
RAW_MANIFEST_SOURCES = "https://data.neonscience.org/api/v0/data/{code}/<site>/<month>"


def _src(*codes: str) -> list[str]:
    return [f"https://data.neonscience.org/data-products/{c}" for c in codes] + \
           [RAW_MANIFEST_SOURCES.format(code=c) for c in codes]


def _release_counts(df: pd.DataFrame) -> dict:
    if "release" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df["release"].value_counts(dropna=False).items()}


def _coverage(df: pd.DataFrame, tcol: str) -> dict:
    if df.empty:
        return {"rows": 0}
    t = pd.to_datetime(df[tcol])
    out = {"rows": int(len(df)), "first": str(t.min())[:10], "last": str(t.max())[:10]}
    if "site" in df.columns:
        out["sites"] = int(df["site"].nunique())
    elif "siteID" in df.columns:
        out["sites"] = int(df["siteID"].nunique())
    out["releases"] = _release_counts(df)
    return out


def release_doi(code: str) -> str:
    p = paths.RAW / "neon" / "_meta" / "products" / f"{code}.json"
    if not p.exists():
        return "DOI: see product page"
    for r in (json.loads(p.read_text()).get("data") or {}).get("releases") or []:
        if r.get("release") == neon.RELEASE:
            return (r.get("productDoi") or {}).get("url") or "DOI: see product page"
    return "DOI: see product page"


def datasheet(name: str, title: str, *, products: list[str], description: str,
              coverage: dict, tables: dict[str, str], qc: list[str], issues: list[str],
              rebuild: str, extra_md: str = "") -> Path:
    prod_lines = "\n".join(f"  - `{c}` — {neon.BY_CODE[c].name if c in neon.BY_CODE else c} "
                           f"(https://data.neonscience.org/data-products/{c}; {neon.RELEASE} "
                           f"{release_doi(c)})" for c in products)
    tab = "\n".join(f"| `{k}` | {v} |" for k, v in tables.items())
    md = f"""# Datasheet — {name} ({title})

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
{prod_lines}
- **Release:** `{neon.RELEASE}` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** {neon.LICENSE}. Attribution required: "{neon.CITATION}"
- **Retrieved:** {TODAY} (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/{name}/`

{description}

## Files

| file | content |
|---|---|
{tab}

## Coverage

```json
{json.dumps(coverage, indent=1, default=str)}
```

## QC applied

{chr(10).join('- ' + q for q in qc)}

## Known issues

{chr(10).join('- ' + q for q in issues)}
{extra_md}
## Rebuild

```bash
{rebuild}
```
"""
    DOCS.mkdir(parents=True, exist_ok=True)
    p = DOCS / f"{name}.md"
    p.write_text(md)
    return p


DL = "python -B scripts/data/download_neon.py"
BL = "python -B scripts/data/build_neon.py"


# --------------------------------------------------------------------------- targets
def t_sites(args) -> None:
    df = na.build_sites()
    out = paths.processed("neon_sites")
    f = na.write_parquet(df, out / "sites.parquet")
    files = [f]
    aop = paths.RAW / "neon" / "_meta" / "aop_availability.parquet"
    if aop.exists():
        a = pd.read_parquet(aop)
        files.append(na.write_parquet(a, out / "aop_availability.parquet"))
    stats = {"sites": int(len(df)), "domains": int(df["domain"].nunique()),
             "aquatic_class": df["aquatic_class"].value_counts().to_dict(),
             "huc12_missing": int(df.get("huc12", pd.Series(dtype=str)).isna().sum())}
    write_manifest("neon_sites", files=files, sources=["https://data.neonscience.org/api/v0/sites",
                                                       "https://data.neonscience.org/api/v0/locations"],
                   description="NEON aquatic site table (34 sites) with domain, aquatic type, "
                               "coordinates, time zone, HUC and per-product coverage; AOP "
                               "availability (not downloaded).", license=neon.LICENSE,
                   stats={**stats, "retrieved": TODAY})
    datasheet("neon_sites", "NEON aquatic site table", products=list(neon.AOP_PRODUCTS),
              description="One row per NEON aquatic site. Use `domain` for leave-one-domain-out "
                          "splits. `station_lat/lon` are the downstream S2 station (streams) or "
                          "buoy (lakes/rivers) from the SENTINEL site registry; `huc*`/`comid` "
                          "come from the registry (WBD point query / NLDI position).",
              coverage=stats,
              tables={"sites.parquet": "site, domain, aquatic_type/class, coordinates, tz, HUC2–12, "
                                       "COMID, n_months_/first_/last_<product>",
                      "aop_availability.parquet": "AOP site-months per product (availability only)"},
              qc=["HUC and COMID strings (leading zeros kept)."],
              issues=["HUC/COMID from point queries; lakes may sit on a reach boundary.",
                      "AOP imagery not downloaded (availability only)."],
              rebuild=f"{DL} --site-meta --aop\n{BL} sites")
    print("sites:", stats)


def t_microbe(args) -> None:
    res = na.build_microbe_16s()
    out = paths.processed("neon_microbe_16s")
    files = []
    for k, df in res.items():
        files.append(na.write_parquet(df, out / f"{k}.parquet"))
    samples = {k: v for k, v in res.items() if k.startswith("samples")}
    summ = na.primer_release_summary(samples, res.get("primers", pd.DataFrame()))
    files.append(na.write_parquet(summ, out / "primer_release_summary.parquet"))
    stats = {}
    for k, v in res.items():
        stats[k] = {"rows": int(len(v))}
        if k.startswith("samples"):
            stats[k].update(_coverage(v, "time_utc"))
            stats[k]["primer_sets"] = v["primer_set"].value_counts().to_dict()
            stats[k]["with_counts"] = int(v.get("n_taxa", pd.Series(dtype="Int64")).notna().sum())
            stats[k]["domains"] = int(v["domain"].nunique())
    write_manifest("neon_microbe_16s", files=files,
                   sources=_src("DP1.20141.002", "DP1.20141.001", "DP1.20282.001") +
                   ["https://storage.neonscience.org/neon-microbe-community/ (taxon tables)",
                    "https://storage.neonscience.org/neon-os-data/data-frame/ (v1 ASV tables)"],
                   description="NEON surface-water 16S: sample metadata + primer set, long count "
                               "tables, OTU/ASV dictionaries (sequence + taxonomy). RELEASE-2026 "
                               "OTUs (DP1.20141.002) and superseded 2014-2018 ASVs (DP1.20141.001) "
                               "kept separate.", license=neon.LICENSE,
                   stats={**stats, "retrieved": TODAY, "release": neon.RELEASE})
    summ_md = "\n## Primer sets and releases (from PcrAmplification records)\n\n" + \
              summ.to_markdown(index=False) + "\n"
    datasheet("neon_microbe_16s", "NEON surface-water 16S community tables",
              products=["DP1.20141.002", "DP1.20141.001", "DP1.20282.001"],
              description="Per-sample 16S community profiles co-collected with grab chemistry "
                          "(join on `parentSampleID`/`eventID`, see `neon_chem`). `primer_set` comes "
                          "from the DP1.20282.001 PcrAmplification record of each dnaSampleID "
                          "(never inferred from the date).",
              coverage=stats,
              tables={"samples.parquet": "DP1.20141.002 16S samples: dnaSampleID, site, domain, "
                                         "time_utc, date, parentSampleID, eventID, sequencerRunID, "
                                         "primer_set/forwardPrimer/reversePrimer, read numbers, "
                                         "field DO/SC/T, release, provisional",
                      "counts.parquet": "long counts (dnaSampleID, sequencerRunID, sequenceName, "
                                        "count, in_metadata) — RELEASE-2026 OTU ids (`ay26_16s_otu.*`)",
                      "otus.parquet": "sequenceName → sequence, domain…genus, scientificName, "
                                      "rank confidences",
                      "samples_v1/counts_v1/otus_v1.parquet": "DP1.20141.001 (2014–2018) ASV "
                                                              "tables (md5 sequence names)",
                      "primers.parquet": "all 16S PCR records with primer_set and "
                                         "primer_record_issue (no provisional PCR records yet)",
                      "primer_release_summary.parquet": "primer set × product summary"},
              qc=["Counts with individualCount <= 0 or missing dropped.",
                  "Samples linked to taxon tables by dnaSampleID; `in_metadata=False` rows are "
                  "samples in a run file whose metadata row is absent (other product/site).",
                  "Staff identifiers (`*By`) dropped."],
              issues=["OTU ids are NOT comparable across releases or between .002 and .001.",
                      "DP1.20141.002 (RELEASE-2026) starts 2018-08: legacy 2014–2018 samples "
                      "(older primer sets) exist only as DP1.20141.001 sequence variants, so a "
                      "cross-primer test also crosses pipeline/table version — compare on "
                      "sequences (`otus*.sequence`) or taxonomy, never on OTU ids.",
                      "The RELEASE-2026 taxonTableLabSummary still points to the AY25 reference "
                      "table (mct_sequence_reference_table_16S_20241111, `ay25_16S_otu.*`), whose "
                      "ids match none of the `ay26_16s_otu.*` ids in the taxon tables; it is "
                      "not used (sequences come from the taxon tables themselves).",
                      "Some taxon tables spell sample IDs in lower case (`WLOU.ss.`); IDs are "
                      "canonicalised case-insensitively to the metadata spelling "
                      "(`dnaSampleID_raw` keeps the original).",
                      "DP1.20141.001: RELEASE-2026 removed duplicate records with/without "
                      "sequencerRunID; rows without a run id are single-run samples.",
                      "Primer sets in surface water (PcrAmplification records): V3V4 "
                      "Pro341F/Pro805R 2014-08→2018-10 (all DP1.20141.001 samples, MiSeq); V4V5 "
                      "515F-Y/926R from 2018-08 (all DP1.20141.002 samples; MiSeq→NovaSeq). No "
                      "V4 515F/806R surface-water records were found. Primer set is confounded "
                      "with time, table version and pipeline; no sample was sequenced with both.",
                      "342 GMCF PCR records (2019-10→2021-10) list ITS1F/ITS2 primers on a 16S "
                      "v4v5 record — a metadata error (taxonomy 98% Bacteria, same amplicon length "
                      "as V4V5); labelled V4V5 with `primer_record_issue` set.",
                      "Sequencing platform changed (MiSeq → NovaSeq) — see `instrument_model`.",
                      "COVID-19 reduced sampling 2020–2022; YELL/BLDE flood gap 2022.",
                      "ITS tables not downloaded (16S only); FASTQ not downloaded."],
              rebuild=f"{DL} --products microbe_tax microbe_comp_v1 microbe_seq --externals\n"
                      f"{BL} microbe", extra_md=summ_md)
    # sequencing metadata stack (separate dataset)
    meta = na.build_microbe_seqmeta()
    out2 = paths.processed("neon_microbe_seqmeta")
    f2 = [na.write_parquet(df, out2 / f"{t}.parquet") for t, df in meta.items() if len(df)]
    st2 = {t: _coverage(df, "collectDate") if "collectDate" in df.columns else {"rows": len(df)}
           for t, df in meta.items()}
    write_manifest("neon_microbe_seqmeta", files=f2, sources=_src("DP1.20282.001"),
                   description="DP1.20282.001 marker-gene sequencing metadata tables (field, "
                               "extraction, PCR, sequencing, raw-file ids).",
                   license=neon.LICENSE, stats={**st2, "retrieved": TODAY})
    datasheet("neon_microbe_seqmeta", "NEON surface-water marker-gene sequencing metadata",
              products=["DP1.20282.001"],
              description="All DP1.20282.001 (expanded) tables stacked across sites and months.",
              coverage=st2, tables={f"{t}.parquet": "stacked table" for t in meta},
              qc=["Typed by the product variables file; staff identifiers dropped."],
              issues=["Raw FASTQ files are referenced (mmg_swRawDataFiles) but not downloaded."],
              rebuild=f"{DL} --products microbe_seq\n{BL} microbe")
    print("microbe:", json.dumps(stats, default=str)[:2000])


def t_chem(args) -> None:
    res = na.build_chem()
    out = paths.processed("neon_chem")
    files = [na.write_parquet(df, out / f"{k}.parquet") for k, df in res.items() if len(df)]
    long, wide = res["chem_long"], res["chem_event_wide"]
    top = long["var"].value_counts().head(40).to_dict()
    stats = {"long": _coverage(long, "time_utc"), "events": int(len(wide)),
             "event_sites": int(wide["site"].nunique()), "domains": int(long["domain"].nunique()),
             "censored_frac": float(long["censored"].mean()),
             "detection_limit_matched_frac": float(long["detection_limit"].notna().mean()),
             "vars_top40": top}
    write_manifest("neon_chem", files=files, sources=_src("DP1.20093.001"),
                   description="NEON surface-water grab chemistry (external lab analytes, field "
                               "in-situ values at sampling, domain-lab ANC/alkalinity).",
                   license=neon.LICENSE, stats={**stats, "retrieved": TODAY})
    datasheet("neon_chem", "NEON surface-water chemistry (grab samples)",
              products=["DP1.20093.001"],
              description="Lab chemistry 12–26×/yr per site, collected at the same events as the "
                          "16S samples (`parentSampleID` e.g. `ARIK.SS.20190806`, `eventID`).",
              coverage=stats,
              tables={"chem_long.parquet": "one row per sample × analyte: value, unit, censored, "
                                           "qualifier (BDL/ND), detection_limit, lab, "
                                           "sample_fraction (FIL/RAW/PCN/POM), var, release",
                      "chem_event_wide.parquet": "one row per parentSampleID: <var> values (mean "
                                                 "of duplicate analyses) + <var>__censored, field "
                                                 "DO/SC/T, lab pH/ANC",
                      "field.parquet": "swc_fieldSuperParent (coordinates, depth, field sonde)",
                      "field_samples.parquet": "swc_fieldData (sample ids, volumes, conditions)",
                      "domain_lab.parquet": "swc_domainLabData (pH, ANC, alkalinity)",
                      "lab_summary.parquet": "lab method detection limits per analyte/period",
                      "units.parquet": "var → analyte, unit"},
              qc=["`censored` = belowDetectionQF ∈ {BDL, ND}; `value` is the published "
                  "concentration (NaN for ND); no DL/2 substitution.",
                  "`detection_limit` = methodDetectionLimit of the same lab/analyte/unit whose "
                  "lab-specific date window contains the collection date.",
                  "`below_mdl` (derived) = value < detection_limit; early-year rows can be below "
                  "the MDL without a NEON flag — `censored` keeps NEON's flag only.",
                  "Units kept per row (`unit`); `var` names encode the unit (e.g. `tn_mgl`, "
                  "`tpc_ugl`).",
                  "POM isotope replicates kept only in the long table.",
                  "Missing units (EcoCore_CSU: Br/Cl/F/SO4 2022-04→2023-09, DIC/DOC/TDN/TN/TOC "
                  "2020-02→03, TDS 2022-12→2023-09, UV absorbance always) filled only from the "
                  "same lab's single unit for that analyte (UV → 'absorbance units'); "
                  "`unit_imputed=True` on every filled row."],
              issues=["Lab changes over time (EcoCore CSU, ISWS CAL, FIU, USGS NWQA) — check "
                      "`lab` before pooling.",
                      "`shipment_warm_qf`=1 rows (warm shipment) kept but flagged.",
                      "`externalLabDataQF` carries format changes and lab remarks."],
              rebuild=f"{DL} --products chem\n{BL} chem")
    print("chem:", json.dumps(stats, default=str)[:1500])


def t_links(args) -> None:
    mdir, cdir = paths.processed("neon_microbe_16s"), paths.processed("neon_chem")
    wide = pd.read_parquet(cdir / "chem_event_wide.parquet")
    parts = [pd.read_parquet(mdir / f) for f in ("samples.parquet", "samples_v1.parquet")
             if (mdir / f).exists()]
    links = na.link_microbe_chem(pd.concat(parts, ignore_index=True), wide)
    f = na.write_parquet(links, mdir / "chem_links.parquet")
    n = sum(len(p) for p in parts)
    stats = {"samples": n, "linked": int(links["dnaSampleID"].nunique()),
             "by_link": links["link"].value_counts().to_dict(),
             "by_product": links.groupby("product")["dnaSampleID"].nunique().to_dict()}
    print("links:", stats)
    m = json.loads((paths.MANIFESTS / "neon_microbe_16s.json").read_text())
    files = [r["path"] for r in m["files"] if not r["path"].endswith("chem_links.parquet")] + [str(f)]
    write_manifest("neon_microbe_16s", files=files, sources=m["sources"],
                   description=m["description"] + " Includes chem_links.parquet (16S sample -> "
                   "neon_chem event).", license=m["license"],
                   stats={**m["stats"], "chem_links": stats})
    ds = DOCS / "neon_microbe_16s.md"
    txt = ds.read_text().split("\n## Microbe ↔ chemistry links")[0]
    ds.write_text(txt.rstrip() + "\n\n## Microbe ↔ chemistry links\n\n`chem_links.parquet`: "
                  "dnaSampleID → chem_parentSampleID (`link='parentSampleID'` = same field event; "
                  "`'location_date'` = same named location, |Δdate| ≤ 1 day, nearest). "
                  f"Stats: `{json.dumps(stats, default=str)}`. Rebuild: `{BL} links` (after "
                  "`microbe` and `chem`).\n")


def _sensor_job(short: str, site: str, offset: float, keep_native: bool) -> tuple[str, dict]:
    res = na.build_sensor_site(short, site, offset, keep_native=keep_native)
    names = {"sonde": ("neon_sonde_hourly", "neon_sonde_daily"),
             "nitrate": ("neon_nitrate", "neon_nitrate"),
             "tsw": ("neon_tsw", "neon_tsw"), "tsd": ("neon_tsd", "neon_tsd"),
             "eos": ("neon_eos", "neon_eos"), "discharge": ("neon_discharge", "neon_discharge")}
    hname, dname = names[short]
    files = []
    if len(res["hourly"]):
        sub = "sites" if hname != dname else "hourly"
        files.append(na.write_parquet(res["hourly"], paths.processed(hname) / sub / f"{site}.parquet"))
    if len(res["daily"]):
        sub = "sites" if hname != dname else "daily"
        files.append(na.write_parquet(res["daily"], paths.processed(dname) / sub / f"{site}.parquet"))
    if "native" in res and len(res["native"]):
        files.append(na.write_parquet(res["native"], paths.processed(hname) / "native" /
                                      f"{site}.parquet"))
    idx = []
    h = res["hourly"]
    if len(h):
        variables, _ = na.SENSOR_PRODUCTS[short]
        outs = [v.out for v in variables] + (["no3n_mgl"] if short == "nitrate" else [])
        for (hor, ver), g in h.groupby(["hor", "ver"]):
            for v in outs:
                c = f"{v}_mean"
                if c not in g.columns:
                    continue
                ok = g[c].notna()
                if not ok.any():
                    continue
                idx.append({"site": site, "hor": hor, "ver": ver, "var": v,
                            "first": g.loc[ok, "time_utc"].min(), "last": g.loc[ok, "time_utc"].max(),
                            "n_hours": int(ok.sum()),
                            "prov_hours": int((ok & g["provisional"]).sum())})
    return site, {"files": [str(f) for f in files], "qc": res["qc"], "index": idx}


SENSOR_META = {
    "sonde": dict(title="NEON water-quality sonde (DP1.20288.001)", products=["DP1.20288.001"],
                  hourly="neon_sonde_hourly", daily="neon_sonde_daily"),
    "nitrate": dict(title="NEON surface-water nitrate (SUNA, DP1.20033.001)",
                    products=["DP1.20033.001"], hourly="neon_nitrate", daily="neon_nitrate"),
    "tsw": dict(title="NEON surface-water temperature (PRT, DP1.20053.001)",
                products=["DP1.20053.001"], hourly="neon_tsw", daily="neon_tsw"),
    "tsd": dict(title="NEON temperature at specific depth (lakes/rivers, DP1.20264.001)",
                products=["DP1.20264.001"], hourly="neon_tsd", daily="neon_tsd"),
    "eos": dict(title="NEON elevation of surface water (DP1.20016.001)",
                products=["DP1.20016.001"], hourly="neon_eos", daily="neon_eos"),
    "discharge": dict(title="NEON continuous discharge (DP4.00130.001)",
                      products=["DP4.00130.001"], hourly="neon_discharge", daily="neon_discharge"),
}


SENSOR_ISSUES = {
    "nitrate": ["SUNA readings near zero can be slightly negative (daily means down to about "
                "-1.7 µmol/L); kept as published, not clipped.",
                "`no3n_mgl` is derived (µmol/L × 0.0140067); SUNA measures nitrate (+ nitrite "
                "interference), compare with lab `no3_no2_n_mgl` accordingly."],
    "sonde": ["fDOM is often fully QF-failed for long periods at some sites (e.g. ARIK 2023-07).",
              "Early months (2016) at several sites are published with all values empty.",
              "`chl_ugl` (µg/L) and `chl_rfu` (RFU) come from the same sensor in different "
              "periods/sites; they are never merged."],
    "discharge": ["`q_ls` is NEON's maxpost discharge (L/s); lakes have no discharge.",
                  "Stage-discharge rating curves change by water year (curveID)."],
}


def t_sensor(short: str, args) -> None:
    offsets = na.site_offsets()
    meta = SENSOR_META[short]
    keep_native = short == "nitrate"
    results = {}
    with ProcessPoolExecutor(max_workers=max(1, min(4, args.workers))) as ex:
        futs = [ex.submit(_sensor_job, short, s, offsets.get(s, 0.0), keep_native)
                for s in neon.AQUATIC_SITES]
        for fu in as_completed(futs):
            site, r = fu.result()
            results[site] = r
            print(f"  {short} {site}: {len(r['index'])} series", flush=True)
    files = [Path(f) for r in results.values() for f in r["files"]]
    index = pd.DataFrame([i for r in results.values() for i in r["index"]])
    qc: dict = {}
    for r in results.values():
        for k, c in r["qc"].items():
            q = qc.setdefault(k, {"n": 0, "masked_qf": 0})
            for kk, vv in c.items():
                q[kk] = q.get(kk, 0) + vv
    hdir = paths.processed(meta["hourly"])
    pos = na.sensor_positions(neon.PRODUCTS[short].code)
    extra = []
    if len(index):
        extra.append(na.write_parquet(index, hdir / "index.parquet"))
    if len(pos):
        extra.append(na.write_parquet(pos, hdir / "sensor_positions.parquet"))
    units = na.sensor_units(short)
    (hdir / "units.json").write_text(json.dumps(units, indent=1))
    (hdir / "qc_counts.json").write_text(json.dumps(qc, indent=1))
    extra += [hdir / "units.json", hdir / "qc_counts.json"]
    cov = {}
    if len(index):
        for v, g in index.groupby("var"):
            cov[v] = {"sites": int(g["site"].nunique()), "series": int(len(g)),
                      "hours": int(g["n_hours"].sum()), "prov_hours": int(g["prov_hours"].sum()),
                      "first": str(g["first"].min())[:10], "last": str(g["last"].max())[:10]}
    stats = {"coverage": cov, "qc": qc, "units": units, "retrieved": TODAY}
    names = sorted({meta["hourly"], meta["daily"]})
    for nm in names:
        nf = [f for f in files + extra if f"/{nm}/" in str(f)]
        if not nf:
            continue
        write_manifest(nm, files=nf, sources=_src(*meta["products"]),
                       description=f"{meta['title']}: final-QF-masked, aggregated to UTC hours and "
                                   "local-standard-time days per site × horizontal (HOR) × "
                                   "vertical (VER) position.", license=neon.LICENSE, stats=stats)
    tables = {"hourly/<SITE>.parquet" if meta["hourly"] == meta["daily"] else "sites/<SITE>.parquet (hourly)":
              "site, time_utc (UTC hour start), hor, ver, <var>_mean/_min/_max/_n, release, provisional",
              "index.parquet": "per site × hor × ver × var: first, last, n_hours, prov_hours",
              "sensor_positions.parquet": "NEON sensor_positions (HOR/VER → location, offsets)",
              "units.json / qc_counts.json": "units per variable; values seen / masked by QF"}
    if meta["hourly"] == meta["daily"]:
        tables["daily/<SITE>.parquet"] = ("site, date (local standard time), hor, ver, "
                                          "<var>_mean/_min/_max/_hours, provisional")
    if keep_native:
        tables["native/<SITE>.parquet"] = "15-min QF-masked values (no3_umoll)"
    desc = ("HOR codes: 101 = upstream sensor set (S1), 102 = downstream (S2) in wadeable streams; "
            "103 = buoy (lakes/rivers); 110/111/112 = lake inlet/outlet/littoral where present "
            "(see sensor_positions). Raw native-resolution values with all basic QF columns are "
            "kept as zstd Parquet in raw/neon.")
    rebuild = f"{DL} --products {short}\n{BL} {short} --workers 4"
    for nm in names:
        datasheet(nm, meta["title"] + (" — daily" if nm.endswith("daily") else
                                       " — hourly" if nm.endswith("hourly") else ""),
                  products=meta["products"], description=desc, coverage=cov,
                  tables=(tables if meta["hourly"] == meta["daily"] else
                          ({"sites/<SITE>.parquet": "site, date (local standard time), hor, ver, "
                                                     "<var>_mean/_min/_max/_hours, provisional"}
                           if nm.endswith("daily") else tables)),
                  qc=[f"Values with the variable's final QF == 1 (or science-review flag == 2) "
                      f"set to NaN before aggregation. QC counts: {json.dumps(qc)}",
                      "Then the SENTINEL plausible-range screen of the USGS daily tier "
                      "(sonde_daily.PLAUSIBLE via the equivalent USGS code, e.g. pH 3–11.5, DO "
                      "0–25 mg/L) masks outliers and small negative turbidity/chlorophyll "
                      "readings are clipped to 0 (counts in qc_counts.json).",
                      "Hourly: mean/min/max/count of valid native values in each UTC hour.",
                      "Daily: local standard time (fixed UTC offset, no DST; `utc_offset_h` in "
                      "neon_sites); count-weighted mean of hourly means, min/max of hourly "
                      "extremes; NaN when < 18 hours have data (`<var>_hours` kept).",
                      f"Units: {json.dumps(units)} — RFU and µg/L chlorophyll are separate "
                      "variables; turbidity is FNU (YSI EXO)."],
                  issues=["Sensor drift/fouling corrections are NEON's; residual spikes remain "
                          "(apply SENTINEL plausibility/spike screens downstream).",
                          "Provisional months (2025-07 onward for most sites) may change.",
                          "Lake buoys are removed in winter (ice) — seasonal gaps by design."]
                         + SENSOR_ISSUES.get(short, []),
                  rebuild=rebuild)
    print(short, "coverage:", json.dumps(cov, default=str)[:1500])


OS_TARGETS = {
    "macroinv": ("macroinv", "neon_macroinv", "Macroinvertebrate collection"),
    "periphyton": ("periphyton", "neon_periphyton", "Periphyton, seston, phytoplankton chemistry"),
    "fish": ("fish", "neon_fish", "Fish electrofishing / gill / fyke netting counts"),
    "zooplankton": ("zooplankton", "neon_zooplankton", "Zooplankton collection"),
    "qpcr": ("microbe_qpcr", "neon_microbe_qpcr", "Surface-water microbe group abundances (qPCR)"),
    "metagenome": ("metagenome", "neon_metagenome_meta", "Surface-water metagenome metadata"),
}


def t_os(target: str, args) -> None:
    short, name, title = OS_TARGETS[target]
    spec = neon.PRODUCTS[short]
    written = na.stack_product(short, name)
    stats = {}
    for t, f in written.items():
        df = pd.read_parquet(f)
        tc = next((c for c in ("collectDate", "startDate", "eventDate", "passStartTime")
                   if c in df.columns), None)
        stats[t] = _coverage(df.rename(columns={"siteID": "site"}), tc) if tc else {"rows": len(df)}
    files = list(written.values())
    if target == "macroinv" and "inv_taxonomyProcessed" in written:
        tax = pd.read_parquet(written["inv_taxonomyProcessed"])
        fld = pd.read_parquet(written["inv_fieldData"]) if "inv_fieldData" in written else None
        den = na.macroinv_density(tax, None, fld)
        files.append(na.write_parquet(den, paths.processed(name) / "density.parquet"))
        stats["density"] = {"rows": int(len(den)),
                            "with_density": int(den.get("density_per_m2", pd.Series()).notna().sum())}
    if not files:
        print(f"{target}: nothing downloaded yet")
        return
    write_manifest(name, files=files, sources=_src(spec.code),
                   description=f"NEON {title} ({spec.code}): all {spec.package} tables stacked.",
                   license=neon.LICENSE, stats={**stats, "retrieved": TODAY})
    tables = {f"{t}.parquet": "stacked NEON table" for t in written}
    if target == "macroinv":
        tables["density.parquet"] = ("inv_taxonomyProcessed + field benthicArea → density_per_m2 "
                                     "(estimatedTotalCount / benthicArea)")
    issues = ["Staff identifiers (`*By`) dropped.", "Release/provisional flagged per row."]
    if target == "metagenome":
        issues.append("Sequence data not downloaded; use NMDC (nmdc:sty-11-hht5sb92, "
                      "PRJNA406977) via the raw-file identifiers here.")
    datasheet(name, title, products=[spec.code],
              description=f"All {spec.package}-package tables of {spec.code} for the 34 aquatic "
                          "sites, typed with the product variables file.",
              coverage=stats, tables=tables, qc=["Typed by variables file; IDs as strings."],
              issues=issues, rebuild=f"{DL} --products {short}\n{BL} {target}")
    print(target, json.dumps(stats, default=str)[:1200])


ALL = ["sites", "microbe", "chem", "links", "metagenome", "nitrate", "sonde", "tsw", "tsd", "eos",
       "discharge", "macroinv", "periphyton", "fish", "zooplankton", "qpcr"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help=f"any of {ALL} or 'all'")
    ap.add_argument("--workers", type=int, default=4,
                    help="single-threaded processes for sensor builds (<= 4; shared-host cap 6 threads)")
    a = ap.parse_args(argv)
    targets = ALL if a.targets == ["all"] else a.targets
    for t in targets:
        print(f"== {t}", flush=True)
        if t == "sites":
            t_sites(a)
        elif t == "microbe":
            t_microbe(a)
        elif t == "chem":
            t_chem(a)
        elif t == "links":
            t_links(a)
        elif t in SENSOR_META:
            t_sensor(t, a)
        elif t in OS_TARGETS:
            t_os(t, a)
        else:
            print(f"unknown target {t}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
