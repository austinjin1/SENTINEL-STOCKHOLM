#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Label snapshot for a live-trial look (amendment §6; review W-L9), built through ONE frozen path:
OGC ``daily`` (all configured sites, all roles) → ``live_data.ogc_long`` (qualifier rule) → frozen series map with
fallback (``apply_series_map``) → ``sonde_daily.clean_long`` + ``splice_site`` → per-site wide frames
``<out>/sites/<site>.parquet`` + ``manifest.json`` (content sha256 = the pin the scorer freeze records).
Labels are then computed by the scorer with the frozen ``events.hazard_labels`` (Approved only).

``--identity-check`` (any time; no outcome data): the same path on the VAL block (WY2022–23), train/val-role sites
only, compared with the legacy-snapshot daily values and the H1/H2-input/H4 onset days computed by the frozen rules;
writes ``dryrun/label_identity/identity.json``. Integrity of the label path, not a model result.

    python -B scripts/live/build_label_snapshot.py --look final_wy2027          # after the approval cut-off
    python -B scripts/live/build_label_snapshot.py --identity-check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time

import numpy as np
import pandas as pd
import requests

import live_common as LC
import live_data as LD
from fetch_usgs import fetch_batch
from sentinel.eval.hazards import HAZARDS, daily_onsets
from sentinel.platform.config import USER_AGENT
from sentinel.platform.live import read_api_key


def pull(sites, t0, t1, log):
    key = read_api_key()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json",
               **({"X-Api-Key": key} if key else {})}
    sess, recs, failed = requests.Session(), [], []
    for i in range(0, len(sites), 10):
        try:
            recs += fetch_batch(sess, headers, sites[i:i + 10], t0, t1, log)
        except Exception as e:  # noqa: BLE001
            failed += sites[i:i + 10]
            log(f"batch {sites[i]} failed: {e}")
        time.sleep(1.0)
    return recs, failed


def _approved(df, var):
    prov = df[f"{var}_prov"].astype(bool) if f"{var}_prov" in df else pd.Series(False, index=df.index)
    return df[var].where(~prov)


def build(recs, smap, window_end):
    long, unknown = LD.ogc_long(recs)
    long, info = LD.apply_series_map(long, smap, window_end=window_end)
    frames, counts = LD.wide_by_site(long)
    return frames, {"unknown_qualifiers": unknown, "series_map": info, "qc": counts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default=None)
    ap.add_argument("--identity-check", action="store_true")
    a = ap.parse_args()
    cfg, cfg_sha = LC.load_config()
    smap_file = LC.REPO / "configs/live/ew_v2_live_series_map.json"
    if not smap_file.is_file():
        smap_file = LC.DRYRUN_ROOT / "series_map_trainval.json"
    smap = json.loads(smap_file.read_text())["map"]
    if a.identity_check:
        out = LC.DRYRUN_ROOT / "label_identity"
        out.mkdir(parents=True, exist_ok=True)
        sites = sorted({r["site_no"] for v in cfg["sites"].values() for r in v if r["role"] in ("train", "val")})
        cache = out / "records_wy2022_23.parquet"
        if cache.exists():
            recs, failed = pd.read_parquet(cache).to_dict("records"), []
        else:
            recs, failed = pull(sites, "2021-10-01", "2023-09-30", print)
            pd.DataFrame(recs).astype({c: "string" for c in pd.DataFrame(recs[:1]).columns if c != "value"}).to_parquet(cache)
        nts = pd.DataFrame(recs).groupby(["monitoring_location_id", "parameter_code", "statistic_id"])[
            "time_series_id"].nunique()
        variants = {}
        for name, sm in (("with_map", smap), ("no_map", None)):
            variants[name] = build(recs, sm, "2023-09-30")
        frames, info = variants["with_map"]
        snapdir = __import__("pathlib").Path(cfg["data_pins"]["sonde_daily_dir"])
        idx = pd.date_range("2021-10-01", "2023-09-30")
        res = {"n_sites": len(sites), "n_failed": len(failed), "info": {k: (v if k != "series_map" else
               {kk: (vv if kk != "fallback_keys" else len(vv)) for kk, vv in v.items()}) for k, v in info.items()},
               "values": {}, "onsets": {}}
        for hz in ("H1_hypoxia", "H4_nitrate"):
            var = HAZARDS[hz].variable
            tot = diff = 0
            on_a = on_b = on_both = 0
            for s, w in frames.items():
                p = snapdir / "sites" / f"{s}.parquet"
                if var not in w or not p.exists():
                    continue
                sn = pd.read_parquet(p)
                sn.index = pd.to_datetime(sn.index)
                if var not in sn:
                    continue
                xa = _approved(w, var).reindex(idx).to_numpy(float)
                xb = _approved(sn, var).reindex(idx).to_numpy(float)
                m = np.isfinite(xa) & np.isfinite(xb)
                tot += int(m.sum())
                diff += int((np.abs(xa[m] - xb[m]) > 1e-4).sum())
                _, oa = daily_onsets(xa, HAZARDS[hz])
                _, ob = daily_onsets(xb, HAZARDS[hz])
                on_a, on_b, on_both = on_a + int(oa.sum()), on_b + int(ob.sum()), on_both + int((oa & ob).sum())
            res["values"][var] = {"compared_days": tot, "differing": diff, "frac_differing": diff / max(tot, 1)}
            res["onsets"][hz] = {"ogc_path": on_a, "legacy_snapshot": on_b, "both": on_both}
            for vname, (fr_, _) in variants.items():      # per-site attribution: multi-series keys vs others
                per = []
                for s, w in fr_.items():
                    p = snapdir / "sites" / f"{s}.parquet"
                    if var not in w or not p.exists():
                        continue
                    sn = pd.read_parquet(p)
                    sn.index = pd.to_datetime(sn.index)
                    if var not in sn:
                        continue
                    xa, xb = _approved(w, var).reindex(idx).to_numpy(float), _approved(sn, var).reindex(idx).to_numpy(float)
                    m = np.isfinite(xa) & np.isfinite(xb)
                    _, oa = daily_onsets(xa, HAZARDS[hz])
                    _, ob = daily_onsets(xb, HAZARDS[hz])
                    pc, st = {"do_mgl_min": ("00300", "00002"), "no3no2_mgl_mean": ("99133", "00003")}[var]
                    per.append({"site": s, "n": int(m.sum()), "ndiff": int((np.abs(xa[m] - xb[m]) > 1e-4).sum()),
                                "only_a": int(np.isfinite(xa).sum() - m.sum()), "only_b": int(np.isfinite(xb).sum() - m.sum()),
                                "on_a": int(oa.sum()), "on_b": int(ob.sum()), "on_both": int((oa & ob).sum()),
                                "multi_series": bool(nts.get((f"USGS-{s}", pc, st), 1) > 1)})
                per = pd.DataFrame(per)
                per.to_csv(out / f"per_site_{hz}_{vname}.csv", index=False)
                g = per.groupby("multi_series")[["n", "ndiff", "only_a", "only_b", "on_a", "on_b", "on_both"]].sum()
                res.setdefault("attribution", {})[f"{hz}|{vname}"] = g.to_dict(orient="index")
        res["created"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        res["series_map_sha256"] = LC.sha256_file(smap_file)
        (out / "identity.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res, indent=1, default=str))
        return 0
    look = cfg["trial"]["looks"][a.look]
    score_freeze = LC.REPO / "docs" / "preregistration" / "FREEZE_ew_live_score.json"
    if not score_freeze.is_file():     # review W-D3: the scorer freeze exists (and is stamped/pushed) BEFORE any labels
        raise SystemExit("FREEZE_ew_live_score.json missing: label snapshots are built only after the scorer freeze")
    if pd.Timestamp.now() < pd.Timestamp(look.get("approval_cutoff", look["earliest"])):
        raise SystemExit(f"label snapshot for {a.look} not before {look.get('approval_cutoff', look['earliest'])}")
    out = LC.LIVE_ROOT / "label_snapshots" / a.look
    if out.exists():
        raise SystemExit(f"{out} exists")
    sites = sorted({r["site_no"] for v in cfg["sites"].values() for r in v})
    end = (pd.Timestamp(look.get("targets_end", look["issues"][1])) + pd.Timedelta(days=8)).date().isoformat()
    recs, failed = pull(sites, LC.LIVE_BLOCK[0], end, print)
    rr = pd.DataFrame(recs)
    nts = rr.groupby(["monitoring_location_id", "parameter_code", "statistic_id"])["time_series_id"].nunique() \
        if len(rr) else pd.Series(dtype=int)
    multi = sorted(f"{k[0].replace('USGS-', '')}|{k[1]}|{k[2]}" for k, v in nts.items() if v > 1)
    frames, info = build(recs, smap, end)
    (out / "sites").mkdir(parents=True)
    for s, w in frames.items():
        w.to_parquet(out / "sites" / f"{s}.parquet")
    files = sorted((out / "sites").glob("*.parquet"))
    h = __import__('hashlib').sha256()
    for f in files:
        h.update(f"{f.relative_to(out).as_posix()}\t{LC.sha256_file(f)}\n".encode())
    man = {"look": a.look, "window": [LC.LIVE_BLOCK[0], end], "n_sites": len(frames), "failed_sites": failed,
           "scorer_freeze_sha256": LC.sha256_file(score_freeze), "multi_series_keys": multi,
           "content_sha256": h.hexdigest(), "series_map_sha256": LC.sha256_file(smap_file), "config_sha256": cfg_sha,
           "info": {**info, "series_map": {**info["series_map"], "fallback_keys": info["series_map"]["fallback_keys"]}},
           "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    LC.excl_write(out / "manifest.json", man)
    print(json.dumps({k: man[k] for k in ("look", "n_sites", "content_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
