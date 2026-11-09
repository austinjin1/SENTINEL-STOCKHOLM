#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download and build the Washington State Toxic Algae label tables (nwtoxicalgae.org).

Network mode (default) makes ~10 sequential requests (>= --sleep s apart): the Data.aspx page
(form state + map layer), ONE "Export Toxin Data" postback with a blank search (= whole
database), the LocationExtractor "Get EIM Locations" postback, the Year/State summary pages,
About.aspx (guideline values), and --verify-sites per-site pages used only as a check.

Raw files go to $SENTINEL_DATA/raw/wa_toxic_algae/<vintage>/ (vintage = retrieval date, UTC);
processed Parquet to $SENTINEL_DATA/processed/labels_wa_toxic_algae/; manifest
'labels_wa_toxic_algae'. Terms: King County website terms (no redistribution without written
permission) — keep the data internal.

    python -B scripts/data/download_events_wa_toxic_algae.py              # download + build
    python -B scripts/data/download_events_wa_toxic_algae.py --from-raw   # rebuild latest vintage offline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import events_common as C  # noqa: E402
from sentinel.data.sources import events_wa_toxic_algae as W  # noqa: E402

log = logging.getLogger("wa_toxic_algae")
NAME = "labels_wa_toxic_algae"


def download(vdir: Path, sleep: float, verify_sites: int) -> dict:
    """Fetch every raw file into ``vdir``; returns the request log."""
    reqlog: dict = {"retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "requests": []}

    def note(url, method, **kw):
        reqlog["requests"].append({"url": url, "method": method, **kw})

    page = W.fetch_data_page()
    W.write_gz(page, vdir / "data_page.html.gz")
    note(W.DATA_URL, "GET", bytes=len(page))
    log.info("Data.aspx: %d bytes", len(page))

    data, meta = W.fetch_export(page, sleep=sleep)
    (vdir / "toxin_export.csv").write_bytes(data)
    note(W.DATA_URL, "POST", form="Export Toxin Data, all search fields blank", **meta)
    log.info("toxin export: %d bytes (%s)", len(data), meta.get("content_disposition"))

    time.sleep(sleep)
    eim, meta = W.fetch_eim_locations(sleep=sleep)
    (vdir / "eim_locations.csv").write_bytes(eim)
    note(W.LOCATION_URL, "POST", form="Get EIM Locations", **meta)
    log.info("EIM locations: %d bytes", len(eim))

    for url, fn in [(W.YEAR_SUMMARY_URL, "year_summary.html.gz"), (W.STATE_SUMMARY_URL, "state_summary.html.gz"),
                    (W.ABOUT_URL, "about.html.gz")]:
        time.sleep(sleep)
        txt = C.get_text(url)
        W.write_gz(txt, vdir / fn)
        note(url, "GET", bytes=len(txt))
        log.info("%s: %d bytes", url, len(txt))

    if verify_sites > 0:
        raw, _ = W.parse_export(W.decode_export(data))
        counts = raw["site_id"].value_counts()
        # deterministic choice: the most-sampled site, a site whose Location text has commas,
        # and evenly spaced sites by sample count
        picks = [counts.index[0]]
        comma = raw.loc[raw["location"].str.contains(",", regex=False), "site_id"]
        if len(comma):
            picks.append(comma.value_counts().index[0])
        step = max(1, len(counts) // max(1, verify_sites))
        picks += [counts.index[k] for k in range(step // 2, len(counts), step)]
        picks = list(dict.fromkeys(picks))[:verify_sites]
        for sid in picks:
            time.sleep(sleep)
            url = W.SITE_PAGE_URL.format(site_id=sid)
            txt = C.get_text(url, timeout=600)
            W.write_gz(txt, vdir / "site_pages" / f"site_{sid}.html.gz")
            note(url, "GET", bytes=len(txt))
            log.info("site page %s: %d bytes", sid, len(txt))
    C.save_json(reqlog, vdir / "request_log.json")
    return reqlog


def build(vdir: Path) -> dict:
    out = paths.processed(NAME)
    page = W.read_gz_text(vdir / "data_page.html.gz")
    text = W.decode_export((vdir / "toxin_export.csv").read_bytes())
    raw, bad = W.parse_export(text)
    if bad:
        C.save_json([{"line_no": k, "line": ln} for k, ln in bad], vdir / "unparsed_lines.json")
        log.warning("%d export lines did not match the row pattern (see unparsed_lines.json)", len(bad))
    if len(bad) > 0.001 * max(1, len(raw)):
        raise RuntimeError(f"too many unparsed export lines: {len(bad)}")
    wn = W.wria_names(page)
    res = W.tidy_results(raw, wria_names=wn)
    unknown_tox = sorted(res.loc[res["toxin"].isna(), "toxin_raw"].unique())
    if unknown_tox:
        log.warning("unknown toxin names (kept with toxin=<NA>): %s", unknown_tox)
    eim = W.parse_eim_locations((vdir / "eim_locations.csv").read_bytes())
    mp = W.parse_map_sites(page)
    sites = W.build_sites(eim, mp, res)
    siteday = W.siteday_labels(res)
    ys = W.parse_summary_table(W.read_gz_text(vdir / "year_summary.html.gz"))
    ss = W.parse_summary_table(W.read_gz_text(vdir / "state_summary.html.gz"))
    cc = W.crosscheck_year_summary(res, ys)
    guide = W.parse_guidelines(W.read_gz_text(vdir / "about.html.gz"))
    if guide != W.WA_GUIDELINES_UGL:
        log.warning("About.aspx guideline values %s differ from WA_GUIDELINES_UGL %s — re-verify labels",
                    guide, W.WA_GUIDELINES_UGL)

    checks = []
    for f in sorted((vdir / "site_pages").glob("site_*.html.gz")) if (vdir / "site_pages").exists() else []:
        sid = f.name[len("site_"):-len(".html.gz")]
        checks.append(W.compare_site_page(W.parse_site_page(W.read_gz_text(f)), res, sid))
    for c in checks:
        log.info("site page check: %s", c)

    files = {
        "results": C.write_parquet(res, out / "wa_toxin_results.parquet"),
        "siteday": C.write_parquet(siteday, out / "wa_toxin_siteday.parquet"),
        "sites": C.write_parquet(sites, out / "wa_sites.parquet"),
        "crosscheck": C.write_parquet(cc, out / "wa_year_summary_crosscheck.parquet"),
    }
    state_tot = ss.set_index("toxin")[["n_samples", "n_exceedances"]].to_dict("index")
    exp_tot = res.groupby("toxin").agg(n=("record_id", "size"),
                                      exceed_published=("exceed_published", lambda x: int(x.fillna(False).sum())),
                                      exceed_wa=("exceed_wa_guideline", lambda x: int(x.fillna(False).sum())))
    reqlog = json.loads((vdir / "request_log.json").read_text()) if (vdir / "request_log.json").exists() else {}
    stats = {
        "vintage": vdir.name,
        "retrieved_utc": reqlog.get("retrieved_utc"),
        "request_params": "Data.aspx 'Export Toxin Data' postback with all search fields blank "
                          "(county='', WRIA=0, site=0, toxin='', dates/concentrations blank); "
                          "LocationExtractor.aspx 'Get EIM Locations' postback",
        "export_rows": len(raw), "unparsed_lines": len(bad),
        "results": C.coverage_stats(res, date_col="date", site_col="site_id"),
        "results_by_toxin": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in exp_tot.to_dict("index").items()},
        "state_summary_published": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in state_tot.items()},
        "censored": int(res["censored"].sum()), "right_censored": int((res["censor_type"] == ">").sum()),
        "result_missing": int(res["result_missing"].sum()),
        "exceed_wa_guideline": res["exceed_wa_guideline"].value_counts(dropna=False).rename(str).to_dict(),
        "exceed_epa_8ugL_microcystin": res.loc[res["toxin"] == "microcystin", "exceed_epa_8ugL"]
            .value_counts(dropna=False).rename(str).to_dict(),
        "published_flag_vs_computed_disagreements": int(
            ((res["exceed_published"] != res["exceed_wa_guideline"]).fillna(False)).sum()),
        "siteday_rows": len(siteday),
        "sites": {"n_sites_listed": len(sites), "n_with_coords": int(sites["latitude"].notna().sum()),
                  "n_with_results": int(sites["in_results"].sum())},
        "crosscheck_year_summary": {
            "cells": len(cc), "cells_samples_match": int((cc["d_samples"] == 0).sum()),
            "cells_exceed_match": int((cc["d_exceed"] == 0).sum()),
            "export_minus_summary_samples": int(cc["d_samples"].sum()),
            "mismatched_cells": cc.loc[(cc["d_samples"] != 0) | (cc["d_exceed"] != 0),
                                       ["year", "toxin", "summary_n", "export_n", "summary_exceed",
                                        "n_flag_and_gt"]].to_dict("records"),
        },
        "guidelines_ugL_about_page": guide,
        "guidelines_ugL_used": W.WA_GUIDELINES_UGL,
        "site_page_checks": checks,
    }
    raw_files = [p for p in sorted(vdir.rglob("*")) if p.is_file()]
    man = write_manifest(
        NAME, files=list(files.values()) + raw_files,
        sources=[W.DATA_URL, W.LOCATION_URL, W.YEAR_SUMMARY_URL, W.STATE_SUMMARY_URL, W.ABOUT_URL,
                 W.SITE_PAGE_URL, W.TERMS_URL],
        description="Washington State Toxic Algae database (WA Ecology Freshwater Algae Program; "
                    "nwtoxicalgae.org, hosted by King County): lab cyanotoxin results per sample x toxin "
                    "with censoring, WA DOH guideline and EPA 8 ug/L microcystin exceedance labels, "
                    "site-day labels, site table with coordinates, YearSummary cross-check.",
        license=W.LICENSE, synthetic=False, stats=stats)
    log.info("wrote %s", man)
    log.info("results %d rows, %d sites, %s..%s", len(res), res["site_id"].nunique(),
             stats["results"].get("date_min"), stats["results"].get("date_max"))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vintage", default=None, help="raw sub-directory (default: today UTC, or latest with --from-raw)")
    ap.add_argument("--from-raw", action="store_true", help="rebuild processed tables from saved raw files only")
    ap.add_argument("--sleep", type=float, default=1.5, help="seconds between requests (>= 1)")
    ap.add_argument("--verify-sites", type=int, default=4, help="per-site pages fetched to verify the export")
    a = ap.parse_args()
    root = paths.raw(W.SOURCE)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(root / "download.log"), logging.StreamHandler()])
    if a.from_raw:
        vintages = sorted(p for p in root.iterdir() if p.is_dir() and (p / "toxin_export.csv").exists())
        if not vintages:
            log.error("no raw vintage under %s", root)
            return 1
        vdir = root / a.vintage if a.vintage else vintages[-1]
    else:
        vdir = paths.ensure(root / (a.vintage or C.retrieval_date()))
        download(vdir, max(1.0, a.sleep), a.verify_sites)
    stats = build(vdir)
    log.info("crosscheck: %s", json.dumps(stats["crosscheck_year_summary"], default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
