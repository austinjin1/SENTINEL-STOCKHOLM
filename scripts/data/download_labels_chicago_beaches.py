#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download and tidy the City of Chicago beach datasets (SafeSwim labels + incumbent model).

Stage 1 (raw): every dataset in ``downstream_chicago_beaches.DATASETS`` as gzipped CSV plus its
Socrata /api/views metadata JSON → $SENTINEL_DATA/raw/chicago_beaches/. Existing raw files are
re-used unless ``--refresh``.
Stage 2 (processed): tidy Parquet tables → $SENTINEL_DATA/processed/labels_chicago_beaches/
  culture_ecoli.parquet, qpcr_enterococci.parquet (kept separate), predictions_official.parquet,
  advisories_2016.parquet, water_sensors_long.parquet, weather_sensors_long.parquet,
  sensor_locations.parquet, beaches.parquet, predictions_vs_qpcr_matched.parquet
Stage 3: manifest ``labels_chicago_beaches``.

    python -B scripts/data/download_labels_chicago_beaches.py            # full build
    python -B scripts/data/download_labels_chicago_beaches.py --no-weather
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import downstream_chicago_beaches as CB  # noqa: E402
from sentinel.data.sources import events_common as C  # noqa: E402

log = logging.getLogger("chicago_beaches")
NAME = "labels_chicago_beaches"


def raw_paths(out: Path, key: str) -> tuple[Path, Path]:
    ds = CB.DATASETS[key]
    return out / f"{key}_{ds}.csv.gz", out / f"{key}_{ds}_metadata.json"


def fetch_raw(out: Path, key: str, refresh: bool) -> tuple[Path, Path]:
    csv_p, meta_p = raw_paths(out, key)
    if csv_p.exists() and meta_p.exists() and not refresh:
        log.info("raw %s exists, re-using (%s)", key, csv_p.name)
        return csv_p, meta_p
    meta = CB.fetch_metadata(key)
    df = CB.fetch_dataset(key)
    n_api = C.socrata_count(CB.DOMAIN, CB.DATASETS[key])
    if len(df) != n_api:
        raise RuntimeError(f"{key}: downloaded {len(df)} rows but API count is {n_api}")
    tmp = csv_p.with_name(csv_p.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False)
    tmp.replace(csv_p)
    C.save_json(meta, meta_p)
    log.info("raw %s: %d rows (API count %d) → %s", key, len(df), n_api, csv_p.name)
    return csv_p, meta_p


def read_raw(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return pd.read_csv(io.StringIO(fh.read()), dtype=str, keep_default_na=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download raw files")
    ap.add_argument("--no-weather", action="store_true", help="skip the weather-station dataset")
    a = ap.parse_args()
    out = paths.raw("chicago_beaches")
    proc = paths.processed(NAME)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "download.log"), logging.StreamHandler()])
    keys = [k for k in CB.DATASETS if not (a.no_weather and k == "weather_sensors")]
    raw_files, meta = [], {}
    for k in keys:
        csv_p, meta_p = fetch_raw(out, k, a.refresh)
        raw_files += [csv_p, meta_p]
        meta[k] = C.socrata_metadata_summary(json.loads(meta_p.read_text()))
    # completeness check of the lab parent table against its filtered views
    view_counts_p = out / "lab_view_counts.json"
    if a.refresh or not view_counts_p.exists():
        vc = {k: C.socrata_count(CB.DOMAIN, v) for k, v in CB.LAB_VIEWS.items()}
        vc["parent"] = C.socrata_count(CB.DOMAIN, CB.DATASETS["lab"])
        vc["retrieved"] = C.retrieval_date()
        C.save_json(vc, view_counts_p)
    vc = json.loads(view_counts_p.read_text())
    raw_files.append(view_counts_p)

    raw = {k: read_raw(raw_paths(out, k)[0]) for k in keys}
    culture, dna = CB.parse_lab(raw["lab"])
    pred = CB.parse_predictions(raw["predictions"])
    adv = CB.parse_advisories(raw["advisories"])
    water = CB.parse_water_sensors(raw["water_sensors"])
    locs = CB.parse_sensor_locations(raw["sensor_locations"])
    beaches = CB.beach_lookup(culture, dna, pred, adv, water, locs)
    culture = CB.fill_coords_from_lookup(culture, beaches)
    dna = CB.fill_coords_from_lookup(dna, beaches)
    matched_all = CB.match_predictions_to_qpcr(pred, dna)
    matched_prev3 = matched_all[matched_all["prev_gap_days"] <= 3]
    tables = {
        "culture_ecoli": culture, "qpcr_enterococci": dna, "predictions_official": pred,
        "advisories_2016": adv, "water_sensors_long": water, "sensor_locations": locs,
        "beaches": beaches, "predictions_vs_qpcr_matched": matched_all,
    }
    if "weather_sensors" in raw:
        tables["weather_sensors_long"] = CB.parse_weather_sensors(raw["weather_sensors"])
    proc_files = []
    for nm, df in tables.items():
        proc_files.append(C.write_parquet(df, proc / f"{nm}.parquet"))
        log.info("processed %s: %d rows", nm, len(df))

    def _years(df, col="date"):
        d = pd.to_datetime(df[col], errors="coerce")
        return {int(k): int(v) for k, v in d.dt.year.value_counts().sort_index().items()}

    cexc, dexc = f"exceeds_{int(CB.CULTURE_BAV_CFU)}_cfu", f"exceeds_{int(CB.QPCR_ADVISORY_CCE)}_cce"

    def _lab_stats(df, exc, thr):
        pub_flag = (df["reading_mean_published"] > thr).where(df["reading_mean_published"].notna())
        both = df[exc].notna() & pub_flag.notna()
        return {
            **C.coverage_stats(df, date_col="date", site_col="beach_key", lat_col="latitude"),
            f"n_{exc}": int(df[exc].sum()), "n_result": int(df["result"].notna().sum()),
            "date_flags": df["date_flag"].value_counts().to_dict(),
            "coord_flags": df["coord_flag"].value_counts().to_dict(),
            "mean_check": df["mean_check"].value_counts().to_dict(),
            "censor_side": df["censor_side"].value_counts().to_dict(),
            "n_time_known": int(df["time_known"].sum()),
            "n_label_differs_from_published_mean": int((df.loc[both, exc].astype(bool)
                                                        != pub_flag[both].astype(bool)).sum()),
            "n_missing_beach": int(df["beach_key"].isna().sum()),
        }

    stats = {
        "retrieval_date": C.retrieval_date(),
        "request": {"domain": CB.DOMAIN, "datasets": {k: CB.DATASETS[k] for k in keys},
                    "paging": "SODA 2.1 CSV, $order=:id, $limit=50000"},
        "socrata_metadata": meta,
        "lab_view_counts": vc,
        "culture": _lab_stats(culture, cexc, CB.CULTURE_BAV_CFU),
        "qpcr": _lab_stats(dna, dexc, CB.QPCR_ADVISORY_CCE),
        "predictions": {**C.coverage_stats(pred, date_col="date", site_col="beach_key"),
                        "sources": pred["prediction_source"].value_counts().to_dict()},
        "advisories": {**C.coverage_stats(adv, date_col="date", site_col="beach_key"),
                       "n_advisory": int(adv["swim_advisory"].sum())},
        "water_sensors": {"n_measurements": int(water["measurement_id"].nunique()),
                          "n_rows_long": int(len(water)),
                          "n_masked": int((water["value"].isna()).sum()),
                          "qualifier_kinds": water["qualifier"].str.split(":").str[0].value_counts().to_dict(),
                          "measurements_by_year": _years(water.drop_duplicates("measurement_id"))},
        "matched_predictions_vs_same_day_qpcr": {
            "n_beach_days": int(len(matched_all)),
            "n_exceed_1000_cce": int(matched_all[dexc].sum()),
            "years": _years(matched_all),
            "n_beach_days_prev_qpcr_within_3d": int(len(matched_prev3)),
            "n_exceed_prev_within_3d": int(matched_prev3[dexc].sum()),
            "n_beaches": int(matched_all["beach_key"].nunique()),
        },
    }
    if "weather_sensors_long" in tables:
        w = tables["weather_sensors_long"]
        stats["weather_sensors"] = {"n_measurements": int(w["measurement_id"].nunique()),
                                    "n_rows_long": int(len(w)), "n_masked": int(w["value"].isna().sum()),
                                    "qualifier_kinds": w["qualifier"].str.split(":").str[0].value_counts().to_dict(),
                                    "measurements_by_year": _years(w.drop_duplicates("measurement_id"))}
    C.save_json(stats, proc / "build_stats.json")
    proc_files.append(proc / "build_stats.json")
    write_manifest(
        NAME, files=raw_files + proc_files,
        sources=[CB.resource_url(CB.DATASETS[k]) for k in keys] + [CB.TERMS_URL],
        description=("City of Chicago / Chicago Park District beach data: culture E. coli (CFU/100 mL) "
                     "and qPCR Enterococci (CCE/100 mL) lab results kept separate, official "
                     "predictions (incumbent), 2016 advisories, automated water/weather sensors."),
        license="City of Chicago Data Portal Terms of Use (as-is; derivative apps must carry the "
                "City disclaimer); no open licence declared on the datasets",
        synthetic=False, stats=stats)
    log.info("matched prediction/qPCR beach-days: %d (prev ≤3 d: %d)", len(matched_all), len(matched_prev3))
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
