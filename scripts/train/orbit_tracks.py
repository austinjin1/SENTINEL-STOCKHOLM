#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit harder tracks (orbit_protocol.md §11; RESEARCH_PLAN §15.5 amendment 2026-11-08).

``--freeze``: write docs/preregistration/orbit_track_u.json (track U rule, refuses to overwrite) with the
track E thresholds (a priori 20 and 50 µg/L chl-a, plus the fit-cell 95th percentile of the given dev
version) and print its sha256. ``--counts``: COUNTS ONLY of the protected track cells (U-test by target /
region; projected track-E counts = headline-cell chl-a count x val-cell exceedance fraction, so no test label
value is used) and the size of U-fit -> docs/preregistration/orbit_tracks_counts_<version>.json.

    python -B scripts/train/orbit_tracks.py --freeze --version dev_v4
    python -B scripts/train/orbit_tracks.py --counts --version dev_v4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import features as F  # noqa: E402
from sentinel.models.orbit import split as SP  # noqa: E402
from sentinel.models.orbit import track_u as U  # noqa: E402

E_FIXED = (20.0, 50.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--version", required=True)
    a = ap.parse_args()
    dev, _, _ = D.load_dev(a.version, training=True)
    fit, inner = F.fit_and_inner(dev, "base")
    p95 = float(np.nanquantile(dev.chla.to_numpy(float)[fit], 0.95))
    if a.freeze:
        sha = U.write_track_file({"created": "2026-11-08", "track_E": {
            "thresholds_ugL_fixed": list(E_FIXED), "threshold_fit_p95_ugL": p95, "p95_from": f"fit cell of {a.version}",
            "cell": "headline cell rows (in-lake, test role x 2024-2025) with observed chl-a >= threshold; also the U-test cell",
            "metrics": "MdSA, SSPB (bias) within stratum; bloom detection AUROC/AUPRC/recall@FPR10 at 20 and 50 ug/L"},
            "note": "defined before any model result on these tracks (RESEARCH_PLAN §15.5 items 5-7)"})
        print(f"wrote {U.TRACK_FILE} sha256 {sha}")
    if a.counts:
        ver = U.verify_track_file()
        raw = pd.concat([D.aquacorr_rows(print), D.wqp_rows(print)], ignore_index=True)
        df = raw[D.image_qc(raw).to_numpy()].reset_index(drop=True)
        df = D.dedupe(df, print)
        df = pd.concat([df, SP.assign(df.lon.to_numpy(), df.lat.to_numpy(), df.date.to_numpy(), log=print)], axis=1)
        D.apply_valid_ranges(df)
        D.apply_turbidity_unit_policy(df)
        u = U.assign(df.anchor_lon, df.anchor_lat)
        tm = U.test_mask(df, u)
        out = {"track_file": ver, "version": a.version, "u_test": {}, "u_fit": {}}
        for t in D.TARGETS + ("turbidity_fnu",):
            ok = tm & df[t].notna().to_numpy()
            g = df[ok]
            out["u_test"][t] = {"n": int(ok.sum()), "n_waterbodies": int(g.waterbody.nunique()),
                                "n_blocks": int(g.block.nunique()),
                                "by_region": {k: int(v) for k, v in g.groupby(u.u_region[ok].to_numpy()).size().items()},
                                "by_main_role": {k: int(v) for k, v in g.role.value_counts().items()}}
        del df
        fu, iu = F.fit_and_inner(dev, "track_u")
        out["u_fit"] = {"n_fit_rows_base": int(fit.sum()), "n_fit_rows_track_u": int(fu.sum()),
                        "n_inner_rows_track_u": int(iu.sum()),
                        "targets": {t: int(dev[t].notna().to_numpy()[fu].sum()) for t in D.TARGETS}}
        # track E projected counts: headline chl-a count x val-cell exceedance fraction (dev labels only)
        rp = json.loads((paths.REPO_ROOT / "docs" / "preregistration" / f"orbit_split_report_{a.version}.json").read_text())
        n_head = rp["protected_cell_counts"]["testxtest_lake"]["chla"]["n"]
        vm = ((dev.role == "val") & (dev.tblock == "val") & dev.in_lake & ~dev.buffer_excluded).to_numpy()
        vy = dev.chla.to_numpy(float)[vm]
        vy = vy[np.isfinite(vy)]
        out["track_E_projected"] = {f"ge{thr:g}": {"val_fraction": float(np.mean(vy >= thr)),
                                                   "projected_headline_n": float(n_head * np.mean(vy >= thr))}
                                    for thr in E_FIXED + (p95,)}
        out["track_E_projected"]["headline_chla_n"] = n_head
        p = paths.REPO_ROOT / "docs" / "preregistration" / f"orbit_tracks_counts_{a.version}.json"
        p.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
