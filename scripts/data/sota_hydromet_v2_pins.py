#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Write ``configs/hydromet_sota/v2/pins.json`` — once (lead rulings §2.3 PIN) — and the code-pin record
``docs/preregistration/hydromet_arch_v2_code_pins.json`` — once (entry-1 lead rulings, decision 4).

    python -B scripts/data/sota_hydromet_v2_pins.py --dry-run OUT.json     # fingerprints only, nothing pinned
    python -B scripts/data/sota_hydromet_v2_pins.py --expect OUT.json      # the pin record (refused if it exists)
    python -B scripts/data/sota_hydromet_v2_pins.py --code-pins            # the code-pin record (refused if it exists)

``--code-pins`` reads no data: it hashes the arch-v2 code files that pins.json does not cover
(:func:`sentinel.hydromet_sota.gso_mt.runner.code_pin_targets`), needs conditions 1–4 of the release guard, and
writes one registry row under ``M6V2/pins_v1`` (metric ``n_code_pinned_files``; never a stamp row).

The record holds

* ``files``  — sha256 of every file of :func:`sentinel.hydromet_sota.gso_mt.data.pin_targets` (every pool table,
  ``lever_sites.parquet``, the FINAL split, the track-N role table, the sonde-tier site lists, ``sites.py``, both
  reserved modules, ``gso_mt/grab.py``, ``gso_mt/data.py``, the solar and constants modules, the v2 build script and
  every file of ``configs/hydromet_sota/v2/``);
* ``values`` — values pinned by content (the T-E static set ``ea_data.COMMON_STATIC``);
* ``pool`` / ``pool_nograb`` — the fingerprint of the training pool as loaded, in both label forms (location list,
  index sets, labels, forcings, statics, solar channels, VAA statics); every non-smoke load compares with it;
* ``info`` — counts and hashes for the record (never a site identifier, never a label statistic).

The script reads the train block of train-role locations only (the pool loader's own guards), asserts the input
widths 30 dynamic / 191 static on the full pool (lead ruling L5), and calls ``assert_stamped`` — not
``assert_released``: the pin record is written before the lead stamps deviations entry 1, which carries its sha256.
It fits nothing and scores nothing. ``--dry-run`` first: the fingerprint must be the same in two processes before
anything is pinned.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

LEVER_INNER = "M6SOTA/inner_v1"
UPSTREAM_SOTA_SHA = "925a21ecf6e3678eaaf824179e09e453851030673418aefa1285ddeea026984f"   # lever engineer, 14:58 EST


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def lever_inner_rows(registry: Path) -> int:
    """Number of registry rows of the lever round's inner experiment. The rows are counted, never read: only the
    ``experiment`` field of a line is looked at."""
    n = 0
    with registry.open() as fh:
        for line in fh:
            if LEVER_INNER in line:
                try:
                    n += json.loads(line).get("experiment") == LEVER_INNER
                except json.JSONDecodeError:
                    pass
    return int(n)


def pool_record(VD, VR, grab: bool, log) -> dict:
    v2 = VD.load_v2_train(grab=grab, require_pins=False, log=log)
    VR.assert_pool_clean(v2.data.sites, where="M6V2 pins")
    fp = VD.pool_fingerprint(v2)
    info = {k: v for k, v in v2.info.items() if k != "pins_sha256"}
    rec = {"fingerprint": fp, "info": info}
    if grab:
        sc = VD.fit_scales(v2)
        keep = np.r_[v2.final_fit_idx[:2], v2.grab_fit_idx[:2], v2.inner_idx[:1]]
        sub = v2.subset(keep)
        dyn, st, dn, sn = VD.make_inputs(sub.data, sub.solar, sub.vaa, sc)
        if (dyn.shape[-1], st.shape[-1]) != (VR.N_DYNAMIC, VR.N_STATIC):
            raise SystemExit(f"the full pool gives {dyn.shape[-1]} dynamic / {st.shape[-1]} static inputs; the "
                             f"stamped set is {VR.N_DYNAMIC} / {VR.N_STATIC} (lead ruling L5): nothing is pinned")
        elev = np.isnan(v2.data.static["elev_m"].to_numpy(dtype="float64"))
        n_usgs = int(v2.info["n_usgs_sites"])
        rec["inputs"] = {"n_dynamic": int(dyn.shape[-1]), "n_static": int(st.shape[-1]),
                         "dynamic_names_sha256": hashlib.sha256("\n".join(dn).encode()).hexdigest(),
                         "static_names_sha256": hashlib.sha256("\n".join(sn).encode()).hexdigest(),
                         "scales_from": "FINAL train-role fit sites (final_fit_idx)", "n_scale_sites": int(len(v2.final_fit_idx))}
        rec["elev_missing"] = {"fit": int(elev[v2.fit_idx].sum()), "inner": int(elev[v2.inner_idx].sum()),
                               "grab_fit": int(elev[v2.grab_fit_idx].sum()),
                               "grab_inner": int(elev[v2.grab_inner_idx].sum()), "all_locations": int(elev.sum())}
        rec["site_tables"] = VD.site_tables_digest(list(v2.data.sites[:n_usgs]))
    del v2
    return rec


def write_code_pins(VR, VD, guard, paths) -> int:
    """The code-pin record: sha256 of every file of ``code_pin_targets`` in the state of the first non-smoke fit.
    Hashes only; nothing is read from any data table."""
    if VR.CODE_PINS.exists():
        raise SystemExit(f"{VR.CODE_PINS} exists: it is written once (entry-1 lead rulings, decision 4)")
    st = VR.assert_entry_released("M6V2 code-pin record")
    code0 = guard.scoring_code_sha256()
    files = {k: VD.sha256_file(p) for k, p in VR.code_pin_targets().items()}
    pk = paths.REPO_ROOT / "sentinel" / "hydromet_sota"
    tests = sorted((paths.REPO_ROOT / "tests").glob("test_hydromet_sota_v2_*.py"))
    kill = json.loads(VR.D2_KILL_RULE.read_text()) if VR.D2_KILL_RULE.is_file() else {}
    rec = {
        "what": "M6 architecture v2 code-pin record (entry-1 lead rulings, decision 4). Written once; read-only. "
                "After this record an arch-v2 code file changes, and a driver or stack module is added, only through "
                "a stamped deviations entry that names the file, the new sha256 and the runs repeated; proposed "
                "line: 'CODE PIN CHANGE: <label> sha256 <sha256>' (the last line of a label is its state).",
        "written_utc": _now(),
        "amendment_sha256": st["amendment_sha256"], "amendment_registry_id": st["registry_id"],
        "deviations_sha256": st["deviations_sha256"], "deviations_registry_id": st["deviations_registry_id"],
        "pins_sha256": st["pins_sha256"], "changed_under_stamp": st["changed_under_stamp"],
        "entry1_lead_rulings_sha256": VD.sha256_file(paths.REPO_ROOT / "docs" / "preregistration"
                                                     / "hydromet_arch_v2_entry1_lead_rulings.md"),
        "files": files,
        "info": {
            "not_pinned_here": "files of configs/hydromet_sota/v2/pins.json (data.py, grab.py, solar.py, "
                               "constants.py, thermox/physics.py, the build script, the configs, tables, split, "
                               "role tables) stay under that record",
            "synthetic_py": {"sha256": VD.sha256_file(pk / "gso_mt" / "synthetic.py"), "pinned": False,
                             "reason": "imported only under --synthetic, which every driver refuses without --smoke"},
            "tests_sha256_informational": {p.name: VD.sha256_file(p) for p in tests},
            "d2_kill_rule_nets_sha256": kill.get("nets_sha256"),
            "d2_kill_rule_matches_thermox_nets": kill.get("nets_sha256") == files[VR.D2_NETS_REL],
            "frozen_m6_code_sha256": code0,
        },
    }
    if guard.scoring_code_sha256() != code0:
        raise SystemExit("the frozen M6 code hash changed during the run")
    VR.CODE_PINS.write_text(json.dumps(rec, indent=1) + "\n")
    os.chmod(VR.CODE_PINS, 0o444)
    out = VR.verify_code_pins()
    if out["code_changed_under_stamp"] or out["n_code_pinned"] != len(files):
        raise SystemExit("the code-pin record does not verify as written")
    from sentinel.core.registry import log_result

    row = log_result(workstream=VR.WORKSTREAM, experiment="M6V2/pins_v1", model="code_pins",
                     metric="n_code_pinned_files", value=float(len(files)),
                     split="none (code-pin record; no data read)", artifacts=[str(VR.CODE_PINS)],
                     notes="arch-v2 code-pin record written once (entry-1 lead rulings, decision 4); not a stamp "
                           "row — the lead records this sha256 in the lead decisions file",
                     extra={"code_pins_sha256": out["code_pins_sha256"], "pins_sha256": st["pins_sha256"],
                            "deviations_sha256": st["deviations_sha256"],
                            "deviations_registry_id": st["deviations_registry_id"], "files": files})
    print(json.dumps({"code_pins": str(VR.CODE_PINS), "sha256": out["code_pins_sha256"], "n_files": len(files),
                      "registry_id": row.get("id"), "files": files}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-pins", action="store_true", help="write the code-pin record (once); reads no data")
    ap.add_argument("--dry-run", default=None, metavar="OUT.json", help="fingerprint of the grab-form pool only")
    ap.add_argument("--expect", default=None, metavar="OUT.json", help="dry-run record the fingerprint must equal")
    a = ap.parse_args()
    from sentinel.core import paths
    from sentinel.hydromet_sota.gso_mt import data as VD
    from sentinel.hydromet_sota.gso_mt import runner as VR
    from sentinel.models.hydromet import guard
    from sentinel.models.hydromet import runner as HR

    HR.env_threads(6)
    stamp = VR.assert_stamped("M6V2 pin record")
    if a.code_pins:
        return write_code_pins(VR, VD, guard, paths)
    log = print
    if a.dry_run:
        rec = pool_record(VD, VR, True, log)
        Path(a.dry_run).write_text(json.dumps({"written_utc": _now(), **rec}, indent=1))
        print(f"dry run: fingerprint written to {a.dry_run}; nothing pinned")
        return 0
    if VD.PINS_FILE.exists():
        raise SystemExit(f"{VD.PINS_FILE} exists: it is written once (lead rulings §2.3)")
    if not a.expect:
        raise SystemExit("run --dry-run first and pass its record with --expect")
    expect = json.loads(Path(a.expect).read_text())
    targets = VD.pin_targets()
    before = {k: VD.sha256_file(p) for k, p in targets.items()}
    up = before["processed/hydromet_upstream_forcing/upstream_forcing_sota.parquet"]
    if up != UPSTREAM_SOTA_SHA:
        raise SystemExit(f"upstream_forcing_sota.parquet is {up[:12]}…, not the lever engineer's final file")
    lever_pins = json.loads((paths.REPO_ROOT / "docs/preregistration/hydromet_sota_lever_pins.json").read_text())
    cross = {"lever_sites.parquet": before["processed/hydromet_sota/lever_sites.parquet"]
             == lever_pins["outputs"]["lever_sites.parquet"],
             "sonde_split.json": before["docs/preregistration/sonde_split.json"] == lever_pins["inputs"]["sonde_split.json"],
             "sonde_newsites_list.json": before["docs/preregistration/sonde_newsites_list.json"]
             == lever_pins["inputs"]["sonde_newsites_list.json"]}
    if not all(cross.values()):
        raise SystemExit(f"a file differs from the lever pin record: {cross}")
    trackn = json.loads(targets["docs/preregistration/hydromet_track_n_stations.json"].read_text())
    if trackn["sha256"] != lever_pins["inputs"]["hydromet_track_n_stations.json"]:
        raise SystemExit("the track-N role table differs from the lever pin record")
    code0 = guard.scoring_code_sha256()
    g = pool_record(VD, VR, True, log)
    if g["fingerprint"] != expect["fingerprint"] or g["inputs"] != expect["inputs"] or g["site_tables"] != expect["site_tables"]:
        raise SystemExit("the pool fingerprint differs from the dry run: the load is not reproducible; nothing pinned")
    ng = pool_record(VD, VR, False, log)
    after = {k: VD.sha256_file(p) for k, p in targets.items()}
    if after != before:
        raise SystemExit(f"a pinned file changed while the pool was read: {[k for k in before if before[k] != after[k]]}")
    from sentinel.core.manifest import write_manifest

    report = json.loads((VD.V2_DIR / "build_report.json").read_text())
    mf = write_manifest(
        "hydromet_sota_v2", files=sorted(p for p in VD.V2_DIR.iterdir() if p.is_file()),
        sources=["Water Quality Portal (WQP) discrete samples: train-role Tier-1 stations, train block, with sample "
                 "time and provider (re-read of the track-N raw archive)",
                 "USGS NWIS instantaneous values aggregated to hourly means at train-role fit sites, train block",
                 "NHDPlus V2 value-added attributes (PathTimeMA, StreamOrde, LengthKM) by COMID"],
        description="M6 architecture v2 pool additions: hour-resolved grab rows and stations, hourly sonde table of the "
                    "fit sites, VAA statics. Train block of train-role locations only; date filter inside the scan.",
        license="US public domain (USGS NWIS; NHDPlus V2); WQP records are public (provider terms: see docs/licenses.md)",
        stats={k: v for k, v in report.items() if isinstance(v, (int, float, str))})
    reg = Path(paths.REGISTRY_PATH)
    n_lever, t_lever = lever_inner_rows(reg), _now()
    rec = {
        "what": "M6 architecture v2 pin record (lead rulings §2.3 PIN). Written once; read-only. A pinned code or "
                "config file changes only through a stamped deviations line 'PIN CHANGE: <label> sha256 <sha256>'; "
                "tables, split, role tables and the pool fingerprint never change.",
        "written_utc": _now(),
        "amendment_sha256": stamp["amendment_sha256"], "amendment_stamp_row": stamp.get("registry_id"),
        "rulings_sha256": VD.sha256_file(paths.REPO_ROOT / "docs/preregistration/hydromet_arch_v2_lead_rulings.md"),
        "files": before, "values": VD.pin_values(), "pool": g["fingerprint"], "pool_nograb": ng["fingerprint"],
        "info": {
            "pool": g["info"], "pool_nograb": ng["info"], "inputs": g["inputs"], "elev_missing": g["elev_missing"],
            "site_tables": g["site_tables"], "bytes": {k: int(p.stat().st_size) for k, p in targets.items()},
            "track_n_content_sha256": trackn["sha256"], "lever_pins_crosscheck": cross,
            "ea_data_py_sha256_informational": VD.sha256_file(paths.REPO_ROOT / "sentinel/hydromet_sota/ea_data.py"),
            "manifest_hydromet_sota_v2_sha256": VD.sha256_file(mf),
            "lever_inner_rows": {"experiment": LEVER_INNER, "count": n_lever, "counted_utc": t_lever,
                                 "note": "rows counted by experiment name only; no value read"},
            "frozen_m6_code_sha256": code0, "dry_run_utc": expect.get("written_utc"),
        },
    }
    if guard.scoring_code_sha256() != code0:
        raise SystemExit("the frozen M6 code hash changed during the run")
    VD.PINS_FILE.write_text(json.dumps(rec, indent=1) + "\n")
    os.chmod(VD.PINS_FILE, 0o444)
    VD.verify_pins()
    sha = VD.sha256_file(VD.PINS_FILE)
    from sentinel.core.registry import log_result

    row = log_result(workstream=VR.WORKSTREAM, experiment="M6V2/pins_v1", model="pins", metric="n_pinned_files",
                     value=float(len(before)), split="none (pin record; train-role pool x train block)",
                     n=int(g["info"]["n_locations"]), data_manifests=list(VR.DATA_MANIFESTS),
                     artifacts=[str(VD.PINS_FILE)],
                     notes="arch-v2 pin record written once (rulings §2.3); not a stamp row — the lead stamps "
                           "deviations entry 1, which carries this sha256",
                     extra={"pins_sha256": sha, "pool_sites_sha256": g["fingerprint"]["sites"],
                            "pool_nograb_sites_sha256": ng["fingerprint"]["sites"], "n_values": len(rec["values"]),
                            "lever_inner_rows": n_lever})
    print(json.dumps({"pins": str(VD.PINS_FILE), "sha256": sha, "n_files": len(before), "registry_id": row.get("id"),
                      "inputs": g["inputs"], "elev_missing": g["elev_missing"], "lever_inner_rows": n_lever,
                      "counted_utc": t_lever, "pool_locations": g["info"]["n_locations"],
                      "pool_nograb_locations": ng["info"]["n_locations"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
