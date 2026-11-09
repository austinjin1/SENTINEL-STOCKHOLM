#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build processed/microbe_wq (+ microbe_wq_v1) — NEON 16S community events joined with co-located
grab chemistry and sonde daily means (docs/preregistration/microbe_protocol.md §2.2).

    python -B scripts/data/build_microbe_wq.py            # needs neon_microbe_16s, neon_chem, neon_sites
                                                         # (+ neon_sonde_daily when available)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.manifest import read_manifest, write_manifest
from sentinel.data import microbe_wq as M

DOCS = paths.REPO_ROOT / "docs" / "datasheets"


def offsets(sites: pd.DataFrame) -> dict:
    out = {}
    for r in sites.itertuples(index=False):
        v = getattr(r, "utc_offset_h", None)
        if v is None or pd.isna(v):
            v = pd.Timestamp("2021-01-15", tz=str(r.timezone)).utcoffset().total_seconds() / 3600
        out[r.site] = float(v)
    return out


def build(suffix: str, a, sites, chem_wide, chem_units, chem_long, sonde_dir) -> dict:
    mdir = paths.PROCESSED / "neon_microbe_16s"
    samples = pd.read_parquet(mdir / f"samples{suffix}.parquet")
    counts = pd.read_parquet(mdir / f"counts{suffix}.parquet")
    otus = pd.read_parquet(mdir / f"otus{suffix}.parquet")
    src = "DP1.20141.002" if suffix == "" else "DP1.20141.001"
    ev, tok, seqs, audit = M.build_events(samples, counts, otus, sites, offsets(sites), min_reads=a.min_reads,
                                          source=src)
    if chem_wide is not None:
        ch = M.match_chemistry(ev, chem_wide, chem_units, chem_long)
        ev = ev.merge(ch, on="sample_id", how="left")
    else:   # community-only build (used to freeze the split before any target is seen)
        ev["chem_match"] = "not_joined"
        for k in M.CHEM_ANALYTES:
            ev[k] = float("nan")
    have_sonde = sonde_dir is not None and Path(sonde_dir).exists()
    if have_sonde:
        ev = ev.merge(M.match_sonde(ev, sonde_dir), on="sample_id", how="left")
    else:
        for k in M.SONDE_VARS:
            ev[k] = float("nan")
            ev[k + "_pm1d"] = float("nan")
    name = "microbe_wq" + suffix
    out = paths.processed(name)
    files = []
    for fn, df in (("samples", ev), ("tokens", tok), ("sequences", seqs)):
        p = out / f"{fn}.parquet"
        df.to_parquet(p, compression="zstd", index=False)
        files.append(p)
    tcols = list(M.CHEM_ANALYTES) + list(M.SONDE_VARS)
    stats = {**audit, "n_events": len(ev), "n_tokens": len(tok), "n_sequences": len(seqs),
             "sites": int(ev["site_id"].nunique()), "domains": int(ev["domain_id"].nunique()),
             "first": str(ev["collect_date"].min())[:10], "last": str(ev["collect_date"].max())[:10],
             "primer_sets": ev["primer_set"].value_counts().to_dict(),
             "chem_match": ev["chem_match"].value_counts().to_dict(), "chem_joined": chem_wide is not None,
             "target_non_missing": {c: int(ev[c].notna().sum()) for c in tcols},
             "events_per_year": ev["collect_date"].dt.year.value_counts().sort_index().to_dict(),
             "sonde_joined": bool(have_sonde), "min_reads": a.min_reads,
             "retrieved_inputs": {m: read_manifest(m)["created"] for m in a.inputs if (paths.MANIFESTS / f"{m}.json").exists()}}
    write_manifest(name, files=files, sources=[f"manifest:{m}" for m in a.inputs],
                   description=f"NEON {src} 16S community events joined with co-located grab chemistry "
                               "(same parentSampleID, else ±24 h) and sonde daily means (S2/buoy, local date).",
                   license="CC BY 4.0 (NEON)", synthetic=False, stats=stats)
    return {"name": name, "stats": stats}


def datasheet(results: list[dict]) -> None:
    today = dt.date.today().isoformat()
    body = []
    for r in results:
        body.append(f"### `{r['name']}`\n\n```json\n{json.dumps(r['stats'], indent=1, default=str)}\n```\n")
    md = f"""# Datasheet — microbe_wq (NEON 16S events × co-located water quality)

- **Built:** {today} by `scripts/data/build_microbe_wq.py` from processed NEON datasets (manifests
  `neon_microbe_16s`, `neon_chem`, `neon_sonde_daily`, `neon_sites`). License CC BY 4.0 (NEON;
  attribution: NSF NEON, RELEASE-2026, https://data.neonscience.org).
- **Unit:** one row per community event = site × local collection date × primer set (16S
  dnaSampleIDs summed). `microbe_wq` = DP1.20141.002 (RELEASE-2026 OTUs); `microbe_wq_v1` =
  DP1.20141.001 (superseded 2014–2018 ASV pipeline; cross-release scheme only).
- **Files:** `samples.parquet` (event metadata, primer set, reads, context, raw matched targets
  `raw_*` + match diagnostics `chem_match`, `chem_dt_h`, `sonde_hor/ver`, `*_pm1d` ablation),
  `tokens.parquet` (sample_id, seq_id = sha1(sequence), count, RDP ranks), `sequences.parquet`.
- **Join rules:** docs/preregistration/microbe_protocol.md §2.2 (chemistry: same parentSampleID,
  else ±24 h at the same site; sonde: daily mean on the local-standard-time date at HOR 102 for
  wadeable streams / HOR 103 for lakes and rivers, shallowest VER).
- **Units:** TN/TP/DOC/NO3+NO2-N mg/L (lab, `milligramsPerLiter` only); SC µS/cm; DO mg/L;
  turbidity FNU; chlorophyll-a µg/L (never RFU).
- **QC:** provisional records dropped; events with < min_reads reads dropped; censored chemistry
  keeps the published value (else the detection limit), flagged `*_censored`.
- **Known issues:** primer set confounded with time; OTU ids release-specific (sequences used
  instead); COVID-19 sampling gaps 2020–2022; lab changes for chemistry.
- **Upstream data issue (workaround in place, fix requested from the NEON data workstream):**
  `neon_chem/chem_event_wide` has one row per *bottle*, keyed by the bottle's parentSampleID with
  fraction/replicate suffixes (`*.RAW.NUT`, `*.FIL.NUT`, `.2`/`.3`); from 2019 on TN/TP/NO3 appear only
  on those suffixed rows, not on the bare field-visit id that the 16S `parentSampleID` points to. A naive
  id join silently loses most post-2019 nutrient targets. Workaround: `sentinel.data.microbe_wq.visit_key`
  strips the suffixes and `collapse_visits` takes the mean over the visit's bottles that report each
  analyte (censored = any) before the parentSampleID match. Verified 2026-11-08: a rebuild against the
  2026-10-25 07:07 UTC upstream refresh is identical to the current files.
- **Rebuild:** `python -B scripts/data/build_microbe_wq.py`

## Coverage

{chr(10).join(body)}
"""
    (DOCS / "microbe_wq.md").write_text(md)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-reads", type=int, default=1000)
    ap.add_argument("--community-only", action="store_true", help="no target joins (split freezing)")
    ap.add_argument("--inputs", nargs="+", default=["neon_microbe_16s", "neon_chem", "neon_sonde_daily", "neon_sites"])
    a = ap.parse_args(argv)
    sites = pd.read_parquet(paths.PROCESSED / "neon_sites" / "sites.parquet")
    cdir = paths.PROCESSED / "neon_chem"
    chem_wide = chem_units = chem_long = None
    if (paths.MANIFESTS / "neon_chem.json").exists() and not a.community_only:
        chem_wide = pd.read_parquet(cdir / "chem_event_wide.parquet")
        chem_units = pd.read_parquet(cdir / "units.parquet")
        chem_long = pd.read_parquet(cdir / "chem_long.parquet", columns=["parentSampleID", "var", "detection_limit"])
    sonde_dir = paths.PROCESSED / "neon_sonde_daily" / "sites"
    sonde_dir = sonde_dir if (paths.MANIFESTS / "neon_sonde_daily.json").exists() and not a.community_only else None
    res = [build("", a, sites, chem_wide, chem_units, chem_long, sonde_dir)]
    if (paths.PROCESSED / "neon_microbe_16s" / "samples_v1.parquet").exists():
        res.append(build("_v1", a, sites, chem_wide, chem_units, chem_long, sonde_dir))
    datasheet(res)
    for r in res:
        print(r["name"], json.dumps(r["stats"], default=str)[:1500])


if __name__ == "__main__":
    main()
