#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M6 freeze CANDIDATE record (protocol §8) — not a freeze. Writes
``docs/preregistration/hydromet_freeze_candidate.json`` with everything the lead needs to write
``FREEZE_hydromet.json`` after the pre-test review:

* split sha256, ``scoring_code_sha256`` (guard), candidate + blend + selected-comparator artifact sha256s;
* data snapshot pins: manifest sha256 and a content sha256 over every file the M6 loaders read (the promoted
  ``sonde_daily`` symlink target, forcings, upstream forcings, StreamCat, site registry);
* a no-test-data audit: every val score and every artifact run used data ending on or before 2023-09-30.

    python -B scripts/eval/hydromet_freeze_candidate.py write --score-dir $SENTINEL_RUNS/hydromet/score_val_v4
    python -B scripts/eval/hydromet_freeze_candidate.py verify      # re-hash the data pins (run before the test look)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

from sentinel.core import paths

OUT = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_freeze_candidate.json"
H = paths.RUNS_ROOT / "hydromet"
VAL_END = "2023-09-30"
CANDIDATE = "ealstm_ws_full"
SEEDED = {   # SENTINEL model -> run-dir pattern (seeds 0-2); all seven models of the Holm-28 family
    "ealstm": "ealstm_v4_s{k}", "ealstm_physdo": "physdo_v4_s{k}", "ealstm_full": "full_v4_s{k}",
    "ealstm_compact": "compact_v4_s{k}", "ealstm_ws": "ealstm_ws_v1_s{k}", "ealstm_ws_full": "ealstm_ws_full_v1_s{k}",
    "forcing_gbm_bag": "gbmbag_v1_s{k}",   # comparator + blend member
}
RUNS = {**{f"{m}_s{k}": H / pat.format(k=k) for m, pat in SEEDED.items() for k in range(3)},
        "forcing_gbm": H / "base_v2", "dosat_ratio[air2stream_regional]": H / "base_v2",
        "air2stream_regional_ws": H / "basews_v1", "dosat_ratio[air2stream_regional]_ws": H / "basews_v1"}
PRIMARY_MODELS = ["ealstm", "ealstm_physdo", "ealstm_gbmblend", "ealstm_full", "ealstm_compact", "ealstm_ws",
                  "ealstm_ws_full"]
TRACK_N_MODELS = ["ealstm_ws_full", "ealstm_gbmblend"]
TRACK_N_ARTIFACTS = [f"{m}_s{k}" for m in ("ealstm_ws_full", "ealstm", "forcing_gbm_bag") for k in range(3)] + [
    "forcing_gbm", "air2stream_regional_ws", "dosat_ratio[air2stream_regional]_ws"]
SNAPSHOTS = {   # role -> (loader dir under processed/, manifest name, file patterns the loader reads)
    "sonde_daily": ("sonde_daily", "sonde_daily", ("index.parquet", "sites/*.parquet")),
    "forcings_daily": ("forcings_daily", "forcings_daily", ("forcings_daily.parquet", "coverage.parquet")),
    "hydromet_upstream_forcing": ("hydromet_upstream_forcing", "hydromet_upstream_forcing", ("upstream_forcing.parquet",)),
    "streamcat": ("catchment", "streamcat", ("streamcat_sites.parquet",)),
    "sites": ("sites", "sites", ("sites.parquet",)),
    "hydromet_wqp_national": ("hydromet_wqp_national", "hydromet_wqp_national",
                              ("stations.parquet", "obs_daily.parquet", "eligibility.parquet", "streamcat_comid.parquet",
                               "station_cells.parquet", "forcings.parquet")),
    "hydromet_upstream_forcing_trackn": ("hydromet_upstream_forcing", "hydromet_upstream_forcing_trackn",
                                         ("upstream_forcing_trackn.parquet",)),
}
PRIMARY_PINS = ["sonde_daily", "forcings_daily", "hydromet_upstream_forcing", "streamcat", "sites"]
TRACK_N_PINS = ["hydromet_wqp_national", "hydromet_upstream_forcing_trackn", "streamcat"]


def _sha(f: Path) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def content_pin(role: str) -> dict:
    d, man, pats = SNAPSHOTS[role]
    root = paths.PROCESSED / d
    files = sorted({f for p in pats for f in root.glob(p) if f.is_file()})
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(root).as_posix()}\t{_sha(f)}\n".encode())
    mf = paths.MANIFESTS / f"{man}.json"
    return {"loader_dir": str(root), "loader_dir_rel": d, "resolved_dir": str(root.resolve()), "patterns": list(pats),
            "n_content_files": len(files), "content_sha256": h.hexdigest(),
            "manifest": man, "manifest_sha256": _sha(mf) if mf.is_file() else None}


def _artifact(name: str, run: Path) -> Path:
    from sentinel.models.hydromet import pipeline

    arts = pipeline.discover_artifacts([run])
    if name not in arts:
        raise KeyError(f"{name} not found in {run}")
    return arts[name]


def no_test_audit(score_dir: Path) -> dict:
    out = {"val_end": VAL_END, "checked": [], "violations": []}
    for d in [score_dir] + sorted(set(RUNS.values())):
        f = d / "data_info.json"
        if not f.exists():
            out["violations"].append(f"{d}: no data_info.json")
            continue
        info = json.loads(f.read_text())
        end, blocks = info.get("date_end"), info.get("blocks")
        out["checked"].append({"dir": str(d), "date_end": end, "blocks": blocks})
        if end is None or end > VAL_END or any(b in ("test", "test_b") for b in (blocks or [])):
            out["violations"].append(f"{d}: date_end {end}, blocks {blocks}")
    return out


def cmd_write(a) -> int:
    from sentinel.models.hydromet import data as D
    from sentinel.models.hydromet import guard

    sp, _ = D.load_split()
    score_dir = Path(a.score_dir)
    comps = json.loads((score_dir / "comparators.json").read_text())
    weights = {n: guard.artifact_sha256(_artifact(n, r)) for n, r in RUNS.items()}
    audit = no_test_audit(score_dir)
    if audit["violations"]:
        raise SystemExit(f"no-test audit failed: {audit['violations']}")
    promo = paths.PROCESSED / "sonde_daily" / "compare_vs_sonde_daily.json"
    rec = {
        "NOT_A_FREEZE": "candidate record for review; FREEZE_hydromet.json is written by the lead only",
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": "docs/preregistration/hydromet_protocol.md (DRAFT v0.11, §8)",
        "protocol_sha256": _sha(paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_protocol.md"),
        "candidate": {"model": CANDIDATE, "seeds": [0, 1, 2], "config": "configs/hydromet/lstm_ws_full.yaml",
                      "rule": "§8: max over standalone EA-LSTMs of mean_endpoints min(Δgbm, Δphys) on M6/val_v4",
                      "reported_separately": "ealstm_gbmblend"},
        "split_sha256": sp.sha256(),
        "scoring_code_sha256": guard.scoring_code_sha256(),
        "weights_sha256": weights,
        "artifacts": {n: str(_artifact(n, r)) for n, r in RUNS.items()},
        "comparators": comps,
        "looks": {
            "M6/test": {"block": "test", "roles": ["test"], "label_model": CANDIDATE,
                        "primary_models": PRIMARY_MODELS, "n_hypotheses": len(PRIMARY_MODELS) * len(comps),
                        "comparators": comps, "artifacts": {n: str(_artifact(n, r)) for n, r in RUNS.items()},
                        "data_snapshots": PRIMARY_PINS},
            "M6N/test": {"block": "test", "roles": ["test"], "label_model": None,
                         "primary_models": TRACK_N_MODELS, "n_hypotheses": len(TRACK_N_MODELS) * 2,
                         "comparators": {v: comps[v] for v in ("temp_c_mean", "do_mgl_mean")},
                         "artifacts": {n: str(_artifact(n, RUNS[n])) for n in TRACK_N_ARTIFACTS},
                         "data_snapshots": TRACK_N_PINS},
        },
        "selection_score": {"experiment": "M6/val_v4", "score_dir": str(score_dir)},
        "data_snapshots": {r: content_pin(r) for r in SNAPSHOTS},
        "sonde_promotion_note": ("processed/sonde_daily -> sonde_daily_refresh_20261108 since 2026-11-08 22:08 EST; "
                                 "identity check compare_vs_sonde_daily.json: all sites identical through 2023-09-30, so "
                                 "artifacts trained before/after the promotion saw identical train/val values"),
        "sonde_identity_check": json.loads(promo.read_text()) if promo.exists() and promo.stat().st_size < 200000 else str(promo),
        "no_test_data_audit": audit,
        "test_look": {"command": "python -B scripts/eval/hydromet_test_look.py --look M6/test [--gpu]; then --look M6N/test",
                      "single_pass": "sentinel/models/hydromet/testlook.py (O_EXCL marker, registry refusal, claim row, "
                                     "sealed predictions); FREEZE = this record minus NOT_A_FREEZE/audit fields",
                      "test_b": "separate later freeze (FREEZE_hydromet_test_b.json) with its own sonde_daily pin",
                      "track_R": "separate later freeze (FREEZE_hydromet_trackR.json): same scoring_code_sha256, fold-unique "
                                 "keys trackr_<fold>/<model>_s<seed>, its own looks block; descriptive"},
    }
    OUT.write_text(json.dumps(rec, indent=1, default=str))
    print(f"wrote {OUT}: {len(weights)} artifacts, snapshots "
          + ", ".join(f"{k} {v['content_sha256'][:12]} ({v['n_content_files']} files)" for k, v in rec["data_snapshots"].items()))
    return 0


def cmd_verify(a) -> int:
    rec = json.loads(OUT.read_text())
    bad = []
    for role, pin in rec["data_snapshots"].items():
        now = content_pin(role)
        for k in ("content_sha256", "manifest_sha256", "resolved_dir"):
            if now[k] != pin[k]:
                bad.append(f"{role}.{k}: {pin[k]} -> {now[k]}")
    from sentinel.models.hydromet import guard

    for n, p in rec["artifacts"].items():
        if guard.artifact_sha256(p) != rec["weights_sha256"][n]:
            bad.append(f"artifact {n} changed")
    if guard.scoring_code_sha256() != rec["scoring_code_sha256"]:
        bad.append("scoring_code_sha256 changed since the record was written")
    if _sha(paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_protocol.md") != rec["protocol_sha256"]:
        bad.append("protocol changed since the record was written")
    from sentinel.models.hydromet import testlook as TL

    for look in rec["looks"]:
        try:
            TL.look_block(rec, look)
        except Exception as e:  # noqa: BLE001
            bad.append(f"look block {look}: {e}")
    print("VERIFY", "PASS" if not bad else "FAIL", *bad, sep="\n")
    return 1 if bad else 0


def cmd_trackr(a) -> int:
    """Track-R candidate record (protocol §10): fold-unique keys trackr_<fold>/<model>_s<seed>; written when all folds
    finished (``--allow-partial`` only for a val rehearsal of the code path, never for a freeze)."""
    from sentinel.models.hydromet import data as D
    from sentinel.models.hydromet import guard
    from sentinel.models.hydromet import tracks as TR

    sp, _ = D.load_split()
    comps = json.loads((paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_track_r_comparators.json").read_text())
    arts, missing = {}, []
    for f in TR.REGION_FOLDS:
        runs = {**{f"ealstm_s{k}": H / f"trackr_{f}_ealstm_s{k}" for k in range(3)},
                **{f"ealstm_full_s{k}": H / f"trackr_{f}_full_s{k}" for k in range(3)},
                **{f"forcing_gbm_bag_s{k}": H / f"trackr_{f}_gbmbag_s{k}" for k in range(3)},
                **{n: H / f"trackr_{f}_base" for n in ("forcing_gbm", "air2stream_regional", "dosat_ratio[air2stream_regional]")}}
        for n, r in runs.items():
            try:
                arts[f"trackr_{f}/{n}"] = str(_artifact(n, r))
            except (KeyError, FileNotFoundError, OSError):
                missing.append(f"trackr_{f}/{n}")
    full_ok = not any("/ealstm_full_" in m for m in missing)
    if not full_ok:
        arts = {k: v for k, v in arts.items() if "/ealstm_full_" not in k}
        missing = [m for m in missing if "/ealstm_full_" not in m]
    if missing and not a.allow_partial:
        raise SystemExit(f"track R incomplete: {len(missing)} artifacts missing, e.g. {missing[:5]}")
    models = ["ealstm", "ealstm_gbmblend"] + (["ealstm_full"] if full_ok else [])
    rec = {"NOT_A_FREEZE": "track-R candidate; FREEZE_hydromet_trackR.json is written by the lead only",
           "partial_rehearsal_only": bool(missing), "missing": missing,
           "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "split_sha256": sp.sha256(), "scoring_code_sha256": guard.scoring_code_sha256(),
           "fold_file_sha256": TR.TRACK_R_SHA256,
           "weights_sha256": {k: guard.artifact_sha256(v) for k, v in arts.items()},
           "data_snapshots": {r: content_pin(r) for r in PRIMARY_PINS},
           "looks": {"M6R/test": {"block": "test", "roles": ["test"], "label_model": None, "primary_models": models,
                                  "n_hypotheses": len(models) * len(comps), "comparators": comps, "artifacts": arts,
                                  "data_snapshots": PRIMARY_PINS}}}
    out = paths.REPO_ROOT / "docs" / "preregistration" / ("hydromet_trackR_rehearsal.json" if missing else "hydromet_trackR_candidate.json")
    out.write_text(json.dumps(rec, indent=1, default=str))
    print(f"wrote {out}: {len(arts)} artifacts, models {models}, missing {len(missing)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--score-dir", required=True)
    sub.add_parser("verify")
    r = sub.add_parser("trackr")
    r.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    return {"write": cmd_write, "verify": cmd_verify, "trackr": cmd_trackr}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
