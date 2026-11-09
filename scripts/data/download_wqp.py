#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download Water Quality Portal (WQX 3.0) discrete samples for SENTINEL v2 (resume-safe).

Phases (run in this order; ``all`` runs them in sequence):

  summary   legacy per-site/year/characteristic counts for each state (planning + completeness
            reference; WQX 2.2, lacks USGS data after 2024-03-11) -> raw/wqp/summary/<ST>.parquet
  results   target groups (optical, phosphorus, nitrogen, carbon, ions, bacteria, cyanotoxins)
            for all 52 states, one request per (group x date window) -> raw/wqp/results/
  physchem  DO / pH / water temperature for all 52 states in semi-monthly windows, stored only
            for site-dates (site + local date) that have a target result -> raw/wqp/physchem/.
            (Restricting requests by site was measured and rejected: the server caps POST bodies
            at ~100 site IDs, so ~5,000 extra requests would save only ~25% of the rows.)
  stations  WQX3 Station/search per state, merged with the location fields embedded in results
            -> raw/wqp/stations.parquet
  manifest  raw manifest (sentinel.core.manifest)

Every finished request is appended to ``raw/wqp/ledger.jsonl``; reruns skip finished chunks.
Windows that the server truncates (``INCOMPLETE DATA``) are split in half recursively.
Log: ``$SENTINEL_DATA/raw/wqp/download.log``.

    python -B scripts/data/download_wqp.py all --workers 2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import wqp as W  # noqa: E402

log = logging.getLogger("wqp")
START = "2000-01-01"
# Window length (days) per group for the national (52-state) target requests.
WINDOW_DAYS = {"optical": 31, "phosphorus": 31, "nitrogen": 31, "carbon": 92, "ions": 31,
               "bacteria": 31, "cyanotoxins": 184}
PHYS_WINDOW = "SMS"      # semi-monthly windows (1st-14th, 15th-end) for physchem


# ------------------------------------------------------------------------------------------------
# Ledger (append-only JSONL; one record per finished chunk)
# ------------------------------------------------------------------------------------------------
class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.done: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "done":
                    self.done[rec["id"]] = rec

    def is_done(self, cid: str) -> bool:
        return cid in self.done

    def add(self, rec: dict) -> None:
        rec = {**rec, "t": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        with self.lock:
            with self.path.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            if rec.get("status") == "done":
                self.done[rec["id"]] = rec


def _windows(start: str, end: str, days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Calendar-aligned windows: monthly for 28-31 d, quarterly for ~92, half-years for ~184."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    freq = {31: "MS", 92: "QS", 184: "6MS", 366: "YS"}.get(days)
    out = []
    if freq:
        starts = pd.date_range(s.to_period("M").to_timestamp() if freq == "MS" else
                               pd.Timestamp(year=s.year, month=1, day=1), e, freq=freq)
        for a in starts:
            b = (a + pd.tseries.frequencies.to_offset(freq)) - pd.Timedelta(days=1)
            a2, b2 = max(a, s), min(b, e)
            if a2 <= b2:
                out.append((a2, b2))
    else:
        a = s
        while a <= e:
            b = min(a + pd.Timedelta(days=days - 1), e)
            out.append((a, b))
            a = b + pd.Timedelta(days=1)
    return out


def _cid(prefix: str, a: pd.Timestamp, b: pd.Timestamp) -> str:
    return f"{prefix}_{a:%Y%m%d}_{b:%Y%m%d}"


# ------------------------------------------------------------------------------------------------
# Phase: summary
# ------------------------------------------------------------------------------------------------
def phase_summary(out: Path, ledger: Ledger, states: list[str], workers: int) -> None:
    sdir = paths.ensure(out / "summary")
    names = W.all_requested_names()

    def one(st: str):
        cid = f"summary_{st}"
        if ledger.is_done(cid) and (sdir / f"{st}.parquet").exists():
            return
        t0 = time.time()
        try:
            text = W.fetch_summary(st, names)
        except (W.WQPError, W.IncompleteResponse) as e:
            log.error("%s failed: %s", cid, e)
            ledger.add({"id": cid, "status": "failed", "error": str(e)[:300]})
            return
        df = W.parse_summary(text)
        df["request_state"] = st
        df.to_parquet(sdir / f"{st}.parquet", index=False, compression="zstd")
        ledger.add({"id": cid, "status": "done", "rows": len(df), "secs": round(time.time() - t0, 1),
                    "n_results": int(df["n_results"].sum()) if len(df) else 0})
        log.info("%s: %d site-year-characteristic rows (%.0fs)", cid, len(df), time.time() - t0)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, states))


def load_summary(out: Path) -> pd.DataFrame:
    files = sorted((out / "summary").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["group"] = df["characteristic"].map(W.name_to_group())
    return df


# ------------------------------------------------------------------------------------------------
# Result chunks (shared by target and physchem phases)
# ------------------------------------------------------------------------------------------------
def _save_chunk(df: pd.DataFrame, locs: pd.DataFrame, dest: Path, locdest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    locdest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(dest)
    ltmp = locdest.with_suffix(".tmp")
    locs.to_parquet(ltmp, index=False, compression="zstd")
    ltmp.replace(locdest)


def fetch_window(group: str, names: tuple[str, ...], states: list[str] | None,
                 a: pd.Timestamp, b: pd.Timestamp, *, siteids: list[str] | None = None,
                 depth: int = 0) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict]:
    """Fetch one window; on truncation split the dates (then the names) and recurse."""
    if siteids is not None:
        body = {"siteid": siteids, "characteristicName": list(names), "startDateLo": W._mdy(a),
                "startDateHi": W._mdy(b), "dataProfile": "fullPhysChem", "sampleMedia": ["Water"]}
    else:
        body = W.result_query(states, names, a, b)
    try:
        text, info = W.post_csv(W.RESULT_URL, body)
    except W.IncompleteResponse as e:
        log.warning("incomplete %s %s..%s (%d names, %d lines) -> split", group, a.date(),
                    b.date(), len(names), e.n_lines)
        if b > a:
            mid = a + (b - a) / 2
            mid = pd.Timestamp(mid.date())
            parts = [(a, mid), (mid + pd.Timedelta(days=1), b)]
            name_parts = [names, names]
        elif len(names) > 1:
            h = len(names) // 2
            parts = [(a, b), (a, b)]
            name_parts = [names[:h], names[h:]]
        elif siteids is not None and len(siteids) > 1:
            h = len(siteids) // 2
            r1 = fetch_window(group, names, states, a, b, siteids=siteids[:h], depth=depth + 1)
            r2 = fetch_window(group, names, states, a, b, siteids=siteids[h:], depth=depth + 1)
            return r1[0] + r2[0], r1[1] + r2[1], {"splits": 1 + r1[2]["splits"] + r2[2]["splits"],
                                                  "bytes": r1[2]["bytes"] + r2[2]["bytes"]}
        else:
            raise
        frames, locs, splits, nbytes = [], [], 1, 0
        for (pa, pb), nm in zip(parts, name_parts):
            f, l_, inf = fetch_window(group, nm, states, pa, pb, siteids=siteids, depth=depth + 1)
            frames += f
            locs += l_
            splits += inf["splits"]
            nbytes += inf["bytes"]
        return frames, locs, {"splits": splits, "bytes": nbytes}
    df = W.parse_results(text, group=group)
    loc = W.parse_result_locations(text)
    return [df], [loc], {"splits": 0, "bytes": info["bytes"]}


def run_target_chunk(out: Path, ledger: Ledger, group: str, a: pd.Timestamp, b: pd.Timestamp,
                     states: list[str]) -> dict | None:
    cid = _cid(group, a, b)
    dest = out / "results" / f"group={group}" / f"{cid}.parquet"
    locdest = out / "locations" / f"{cid}.parquet"
    if ledger.is_done(cid) and dest.exists():
        return None
    t0 = time.time()
    try:
        frames, locs, info = fetch_window(group, W.CHARACTERISTIC_GROUPS[group], states, a, b)
    except (W.WQPError, W.IncompleteResponse) as e:
        log.error("%s failed: %s", cid, e)
        ledger.add({"id": cid, "status": "failed", "error": str(e)[:300]})
        return None
    df = pd.concat(frames, ignore_index=True) if frames else W.empty_result_frame()
    df = df[df["state"].isin(states) | (df["state"] == "")]
    lc = pd.concat(locs, ignore_index=True).drop_duplicates("site_id") if locs else pd.DataFrame()
    _save_chunk(df, lc, dest, locdest)
    rec = {"id": cid, "status": "done", "phase": "results", "group": group, "start": str(a.date()),
           "end": str(b.date()), "rows": len(df), "sites": int(df["site_id"].nunique()),
           "bytes": info["bytes"], "splits": info["splits"], "secs": round(time.time() - t0, 1)}
    ledger.add(rec)
    log.info("%s: %d rows, %d sites, %d splits, %.1f MB, %.0fs", cid, len(df), rec["sites"],
             info["splits"], info["bytes"] / 1e6, rec["secs"])
    return rec


def phase_results(out: Path, ledger: Ledger, states: list[str], groups: list[str], start: str,
                  end: str, workers: int) -> None:
    jobs = []
    for g in groups:
        for a, b in _windows(start, end, WINDOW_DAYS[g]):
            if not ledger.is_done(_cid(g, a, b)):
                jobs.append((g, a, b))
    # interleave groups so two workers rarely hit the same table at once
    jobs.sort(key=lambda j: (j[1], groups.index(j[0])))
    log.info("results: %d chunks to fetch (%d already done)", len(jobs),
             sum(1 for k in ledger.done if k.split("_")[0] in groups))
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_target_chunk, out, ledger, g, a, b, states) for g, a, b in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            if rec:
                total += rec["rows"]
            if i % 25 == 0:
                log.info("results progress: %d/%d chunks, %d rows this run", i, len(jobs), total)


# ------------------------------------------------------------------------------------------------
# Phase: physchem (co-sampled only)
# ------------------------------------------------------------------------------------------------
def target_site_dates(out: Path) -> pd.DataFrame:
    """Unique (site_id, activity_start_date) with any target result (all statuses/QC classes)."""
    frames = []
    for f in sorted((out / "results").glob("group=*/*.parquet")):
        frames.append(pd.read_parquet(f, columns=["site_id", "activity_start_date"]).drop_duplicates())
    if not frames:
        return pd.DataFrame(columns=["site_id", "activity_start_date"])
    sd = pd.concat(frames, ignore_index=True).drop_duplicates()
    return sd.sort_values("activity_start_date").reset_index(drop=True)


def semimonthly_windows(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    starts = pd.date_range(s.to_period("M").to_timestamp(), e, freq=PHYS_WINDOW)
    out = []
    for i, a in enumerate(starts):
        b = (starts[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(starts) else (
            a + pd.offsets.SemiMonthBegin() - pd.Timedelta(days=1))
        a2, b2 = max(a, s), min(b, e)
        if a2 <= b2:
            out.append((a2, b2))
    return out


def run_phys_window(out: Path, ledger: Ledger, a: pd.Timestamp, b: pd.Timestamp,
                    states: list[str], sd: pd.DataFrame) -> dict | None:
    cid = _cid("physchem", a, b)
    dest = out / "physchem" / f"{cid}.parquet"
    locdest = out / "locations" / f"{cid}.parquet"
    if ledger.is_done(cid) and dest.exists():
        return None
    t0 = time.time()
    try:
        frames, locs, info = fetch_window("physchem", W.CHARACTERISTIC_GROUPS["physchem"], states, a, b)
    except (W.WQPError, W.IncompleteResponse) as e:
        log.error("%s failed: %s", cid, e)
        ledger.add({"id": cid, "status": "failed", "error": str(e)[:300]})
        return None
    df = pd.concat(frames, ignore_index=True) if frames else W.empty_result_frame()
    n_all = len(df)
    lo, hi = str(a.date()), str(b.date())
    w = sd[(sd["activity_start_date"] >= lo) & (sd["activity_start_date"] <= hi)]
    keys = set(w["site_id"] + "|" + w["activity_start_date"])
    if n_all:
        df = df[(df["site_id"] + "|" + df["activity_start_date"]).isin(keys)]
    used = set(df["site_id"])
    lc = pd.concat(locs, ignore_index=True).drop_duplicates("site_id") if locs else pd.DataFrame(columns=["site_id"])
    lc = lc[lc["site_id"].isin(used)]
    _save_chunk(df, lc, dest, locdest)
    rec = {"id": cid, "status": "done", "phase": "physchem", "start": lo, "end": hi,
           "rows_downloaded": n_all, "rows": len(df), "sites": int(df["site_id"].nunique()),
           "bytes": info["bytes"], "splits": info["splits"], "secs": round(time.time() - t0, 1)}
    ledger.add(rec)
    log.info("%s: %d rows downloaded, %d co-sampled kept (%.0f%%), %d splits, %.1f MB, %.0fs", cid,
             n_all, len(df), 100 * len(df) / max(n_all, 1), info["splits"], info["bytes"] / 1e6,
             rec["secs"])
    return rec


def phase_physchem(out: Path, ledger: Ledger, states: list[str], start: str, end: str,
                   workers: int) -> None:
    sd = target_site_dates(out)
    jobs = [(a, b) for a, b in semimonthly_windows(start, end)
            if not ledger.is_done(_cid("physchem", a, b))]
    log.info("physchem: %d windows to fetch; %d target site-dates", len(jobs), len(sd))
    kept = down = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_phys_window, out, ledger, a, b, states, sd) for a, b in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            if rec:
                kept += rec["rows"]
                down += rec["rows_downloaded"]
            if i % 25 == 0:
                log.info("physchem progress: %d/%d windows, %d downloaded, %d kept this run", i,
                         len(jobs), down, kept)


# ------------------------------------------------------------------------------------------------
# Phase: stations
# ------------------------------------------------------------------------------------------------
def phase_stations(out: Path, ledger: Ledger, states: list[str], start: str, end: str,
                   workers: int) -> None:
    sdir = paths.ensure(out / "stations_by_state")
    names = W.all_requested_names()

    def one(st: str):  # noqa: C901
        cid = f"stations_{st}"
        if ledger.is_done(cid) and (sdir / f"{st}.parquet").exists():
            return
        t0 = time.time()
        frames = []
        try:
            try:  # all sites of the state (no result join: cheap); filtered to result sites below
                text, _ = W.fetch_stations([st])
                frames.append(W.parse_stations(text))
            except W.IncompleteResponse:
                for g, nm in W.CHARACTERISTIC_GROUPS.items():  # fall back: one query per group
                    text, _ = W.fetch_stations([st], nm, start, end)
                    frames.append(W.parse_stations(text))
        except (W.WQPError, W.IncompleteResponse) as e:
            log.error("%s failed: %s", cid, e)
            ledger.add({"id": cid, "status": "failed", "error": str(e)[:300]})
            return
        df = pd.concat(frames, ignore_index=True).drop_duplicates("site_id")
        df.to_parquet(sdir / f"{st}.parquet", index=False, compression="zstd")
        ledger.add({"id": cid, "status": "done", "rows": len(df), "secs": round(time.time() - t0, 1)})
        log.info("%s: %d stations (%.0fs)", cid, len(df), time.time() - t0)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, states))
    # merge Station/search with locations embedded in result rows (covers every result site)
    st = [pd.read_parquet(f) for f in sorted(sdir.glob("*.parquet"))]
    st = pd.concat(st, ignore_index=True) if st else pd.DataFrame(columns=["site_id"])
    st["station_source"] = "Station/search"
    loc = [pd.read_parquet(f) for f in sorted((out / "locations").glob("*.parquet"))]
    loc = pd.concat(loc, ignore_index=True).drop_duplicates("site_id") if loc else pd.DataFrame(columns=["site_id"])
    loc["station_source"] = "Result location fields"
    # keep only sites that carry at least one downloaded result
    st = st[st["site_id"].isin(loc["site_id"])]
    merged = pd.concat([st, loc[~loc["site_id"].isin(st["site_id"])]], ignore_index=True)
    merged = merged.drop_duplicates("site_id").reset_index(drop=True)
    for c in merged.columns:
        if merged[c].dtype == object:
            merged[c] = merged[c].fillna("").astype(str)
    merged.to_parquet(out / "stations.parquet", index=False, compression="zstd")
    log.info("stations: %d from Station/search + %d only in result rows = %d", len(st),
             len(merged) - len(st), len(merged))


# ------------------------------------------------------------------------------------------------
# Phase: manifest
# ------------------------------------------------------------------------------------------------
def phase_manifest(out: Path, ledger: Ledger, start: str, end: str) -> None:
    files = sorted(out.glob("results/group=*/*.parquet")) + sorted(out.glob("physchem/*.parquet"))
    files += sorted(out.glob("summary/*.parquet"))
    for extra in ("stations.parquet", "ledger.jsonl"):
        if (out / extra).exists():
            files.append(out / extra)
    recs = list(ledger.done.values())
    by_group: dict[str, int] = {}
    for r in recs:
        if r.get("phase") == "results":
            by_group[r["group"]] = by_group.get(r["group"], 0) + r["rows"]
    stats = {
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "date_range": [start, end],
        "states": list(W.STATES),
        "profile": "fullPhysChem",
        "sample_media": "Water",
        "characteristic_groups": {g: list(v) for g, v in W.CHARACTERISTIC_GROUPS.items()},
        "rows_by_group": by_group,
        "physchem_rows_kept": sum(r.get("rows", 0) for r in recs if r.get("phase") == "physchem"),
        "physchem_rows_downloaded": sum(r.get("rows_downloaded", 0) for r in recs
                                        if r.get("phase") == "physchem"),
        "n_chunks_done": len(recs),
        "request": "POST /wqx3/Result/search?mimeType=csv&zip=yes&sorted=no (JSON body)",
    }
    m = write_manifest(
        "wqp_raw", files=files, sources=[W.RESULT_URL, W.STATION_URL, W.SUMMARY_URL],
        description="Water Quality Portal WQX 3.0 (fullPhysChem) discrete results for SENTINEL "
                    "target characteristics, 52 jurisdictions (48 contiguous states + DC, AK, HI, PR), "
                    f"{start}..{end}; parsed to Parquet (sentinel codes masked, censoring parsed); "
                    "physchem kept only where co-sampled with a target result. Legacy summary "
                    "counts (WQX 2.2) included for planning/completeness reference only.",
        license="US public domain / open government data (WQP; data owned by submitting organizations)",
        stats=stats, hash_files=True)
    log.info("manifest: %s", m)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["summary", "results", "physchem", "stations", "manifest", "all"])
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--states", default=",".join(W.STATES))
    ap.add_argument("--groups", default=",".join(W.TARGET_GROUPS))
    ap.add_argument("--workers", type=int, default=2, help="concurrent requests (polite: <= 2)")
    a = ap.parse_args()
    if a.workers > 2:
        ap.error("--workers must be <= 2 (politeness limit for this project)")
    out = paths.raw("wqp")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
                        handlers=[logging.FileHandler(out / "download.log"), logging.StreamHandler()])
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    ledger = Ledger(out / "ledger.jsonl")
    states = a.states.split(",")
    groups = a.groups.split(",")
    log.info("phase=%s start=%s end=%s states=%d groups=%s workers=%d", a.phase, a.start, a.end,
             len(states), groups, a.workers)
    phases = ["summary", "results", "physchem", "stations", "manifest"] if a.phase == "all" else [a.phase]
    for ph in phases:
        t0 = time.time()
        if ph == "summary":
            phase_summary(out, ledger, states, a.workers)
        elif ph == "results":
            phase_results(out, ledger, states, groups, a.start, a.end, a.workers)
        elif ph == "physchem":
            missing = [g for g in groups if not any(r.get("group") == g for r in ledger.done.values())]
            if missing:
                log.warning("physchem: no finished target chunks for %s yet", missing)
            phase_physchem(out, ledger, states, a.start, a.end, a.workers)
        elif ph == "stations":
            phase_stations(out, ledger, states, a.start, a.end, a.workers)
        elif ph == "manifest":
            phase_manifest(out, ledger, a.start, a.end)
        log.info("phase %s finished in %.1f min", ph, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
