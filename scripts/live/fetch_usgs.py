#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Fetch provisional USGS daily values for the live-trial sites (USGS Water Data OGC API, collection ``daily``).

One fetch per issue day D: every value dated [D - lookback, D - 1] at the configured sites (20 sites per request,
all parameters/statistics), stored as an immutable parquet + manifest (SHA-256s) under
``$SENTINEL_RUNS/live/ew_v2[/dryrun]/inputs/usgs/<D>/``. The API key is read only by the existing loader
(``sentinel.platform.live.read_api_key``), sent as a header, never logged.

    python -B scripts/live/fetch_usgs.py --date 2026-12-01 [--mode dryrun]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time

import pandas as pd
import requests

from live_common import DRYRUN_ROOT, LIVE_BLOCK, LIVE_ROOT, excl_write, load_config, sha256_file
from sentinel.platform.config import USER_AGENT, USGS_OGC_BASE
from sentinel.platform.live import read_api_key

PROPS = ("monitoring_location_id", "parameter_code", "statistic_id", "time", "value", "unit_of_measure",
         "approval_status", "qualifier", "time_series_id", "last_modified")


def fetch_batch(sess, headers, sites, t0, t1, log, max_tries: int = 6) -> list[dict]:
    url = f"{USGS_OGC_BASE}/collections/daily/items"
    q = {"f": "json", "skipGeometry": "true", "limit": "10000", "properties": ",".join(PROPS),
         "monitoring_location_id": ",".join(f"USGS-{s}" for s in sites), "time": f"{t0}/{t1}"}
    out, nxt, params = [], url, q
    while nxt:
        for k in range(max_tries):
            try:
                r = sess.get(nxt, params=params, headers=headers, timeout=120)
            except requests.RequestException as e:
                log(f"network error {e.__class__.__name__}; retry {k + 1}")
                time.sleep(10 * (k + 1))
                continue
            if r.status_code == 200:
                break
            if r.status_code == 429:
                wait = min(int(r.headers.get("Retry-After", "60") or 60), 3600)
                log(f"HTTP 429, waiting {wait} s")
                time.sleep(wait)
                continue
            if 500 <= r.status_code < 600:
                time.sleep(10 * (k + 1))
                continue
            raise RuntimeError(f"HTTP {r.status_code} for sites {sites[0]}..")
        else:
            raise RuntimeError(f"failed after {max_tries} tries for sites {sites[0]}..")
        doc = r.json()
        out += [f.get("properties") or {} for f in doc.get("features") or []]
        nxt = next((ln.get("href") for ln in doc.get("links") or [] if ln.get("rel") == "next"), None)
        if nxt and not str(nxt).startswith(USGS_OGC_BASE):
            raise RuntimeError("next link points to another host (key never sent elsewhere)")
        params = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=_dt.datetime.now(_dt.timezone.utc).date().isoformat(), help="issue day D")
    ap.add_argument("--mode", choices=("live", "dryrun"), default="live")
    ap.add_argument("--max-sites", type=int, default=0, help="dry-run only: limit the number of sites")
    a = ap.parse_args()
    cfg, cfg_sha = load_config()
    D = pd.Timestamp(a.date)
    root = (LIVE_ROOT if a.mode == "live" else DRYRUN_ROOT) / "inputs" / "usgs" / D.date().isoformat()
    if (root / "manifest.json").exists():
        print(f"{root} already fetched")
        return 0
    roles = None if a.mode == "live" else {"train", "val"}
    sites = sorted({r["site_no"] for v in cfg["sites"].values() for r in v if roles is None or r["role"] in roles})
    if a.max_sites and a.mode == "dryrun":
        sites = sites[: a.max_sites]
    lb = int(cfg["fetch"]["lookback_days"])
    # never request anything before the live block (WY2026 = Test-B is protected)
    t0 = max(D - pd.Timedelta(days=lb), pd.Timestamp(LIVE_BLOCK[0])).date().isoformat()
    t1 = (D - pd.Timedelta(days=1)).date().isoformat()
    key = read_api_key()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"}
    if key:
        headers["X-Api-Key"] = key
    sess = requests.Session()
    root.mkdir(parents=True, exist_ok=True)
    logf = open(root / "fetch.log", "a")

    def log(m):
        logf.write(f"{_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')} {m}\n")
        logf.flush()

    started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    recs, failed, n_req = [], [], 0
    bs = int(cfg["fetch"]["batch_sites"])
    for i in range(0, len(sites), bs):
        batch = sites[i:i + bs]
        try:
            recs += fetch_batch(sess, headers, batch, t0, t1, log)
            n_req += 1
        except Exception as e:  # noqa: BLE001 — a failed batch is recorded; those sites get no issue today
            failed += batch
            log(f"batch {batch[0]}.. failed: {e}")
        time.sleep(1.0)
    df = pd.DataFrame(recs, columns=list(PROPS))
    for c in df.columns:
        if c != "value":
            df[c] = df[c].astype("string")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    out = root / "records.parquet"
    df.to_parquet(out, index=False)
    out.chmod(0o444)
    man = {"issue_date": D.date().isoformat(), "mode": a.mode, "window": [t0, t1], "n_sites_requested": len(sites),
           "n_sites_failed": len(failed), "failed_sites": failed, "n_requests": n_req, "n_records": int(len(df)),
           "n_sites_with_records": int(df["monitoring_location_id"].nunique()) if len(df) else 0,
           "authenticated": bool(key), "started_utc": started,
           "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "records_sha256": sha256_file(out), "config_sha256": cfg_sha, "source": f"{USGS_OGC_BASE} daily"}
    excl_write(root / "manifest.json", man)
    print(json.dumps({k: v for k, v in man.items() if k != "failed_sites"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
