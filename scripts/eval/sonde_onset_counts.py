#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EW-v2 protocol §9: calibration-period onset counts per hazard from the TRAIN and VAL blocks
only (train- and val-role stream sites; test-role sites and test blocks are never loaded — the
guarded loader refuses them). Counts onsets (daily tier, Approved values only, exactly the labels
the event heads use), sites and basin groups with onsets, by block × site role, and logs them to
the registry (``--register``).

    python -B scripts/eval/sonde_onset_counts.py --config configs/sonde/events.yaml --register
"""

from __future__ import annotations

from sentinel.data.reserved import assert_no_reserved  # noqa: E402

import numpy as np
import pandas as pd

from sentinel.data.sonde_splits import TIME_BLOCKS
from sentinel.eval.hazards import HAZARDS
from sentinel.models.sonde import events as E
from sentinel.models.sonde import runner


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--hazards", default="H1_hypoxia,H1_hypoxia_t4,H1_hypoxia_t2,H4_nitrate,H5b_chl_bloom")
    a = ap.parse_args()
    cfg = runner.apply_overrides(runner.load_config(a.config, smoke=a.smoke), a.set)
    cfg["use_weather"] = cfg["use_subdaily"] = False
    run_dir = runner.run_directory("onset_counts", cfg, a.run_name)
    log = runner.make_logger(run_dir / "log.txt")
    data = runner.prepare(a, cfg, run_dir, training=True, log=log)
    assert_no_reserved(data.sites, 'sonde_onset_counts')   # reserved new-site block
    basin = {str(k): str(v) for k, v in data.split.basin_group.items()}
    dates = pd.DatetimeIndex(data.cube.dates)
    rows = []
    for hz in a.hazards.split(","):
        if HAZARDS[hz].variable not in data.cube.variables:
            continue
        lab = E.hazard_labels(data, hz, (3,))
        for block in ("train", "val"):
            lo, hi = (pd.Timestamp(x) for x in TIME_BLOCKS[block])
            tm = np.asarray((dates >= lo) & (dates <= hi))
            for role in ("train", "val", "all"):
                sm = np.ones(len(data.sites), bool) if role == "all" else (np.asarray(data.roles) == role)
                on = lab.onset[np.ix_(sm, tm)]
                per_site = on.sum(axis=1)
                sites = np.asarray(data.sites)[sm][per_site > 0]
                x = data.cube.y[np.ix_(sm, tm, [data.cube.var_index(HAZARDS[hz].variable)])][..., 0]
                rows.append({"hazard": hz, "block": block, "role": role, "n_onsets": int(on.sum()),
                             "n_sites_observed": int(np.isfinite(x).any(axis=1).sum()),
                             "n_sites_with_onset": int(sites.size),
                             "n_basin_groups_with_onset": len({basin.get(str(s)) for s in sites}),
                             "n_issue_rows_valid_L3": int(lab.valid[3][np.ix_(sm, tm)].sum())})
    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "onset_counts.csv", index=False)
    log(f"[onsets] train/val blocks only (test never loaded):\n{res.to_string(index=False)}")
    if a.register:
        from sentinel.core.registry import log_result

        for r in res.to_dict("records"):
            log_result(workstream=runner.WORKSTREAM, experiment="ew_v2_onset_counts", model="labels",
                       metric=f"n_onsets_{r['block']}block_{r['role']}sites", value=float(r["n_onsets"]),
                       split=f"{r['block']}/sonde_split_{data.split_info['sha256'][:12]}", n=r["n_sites_observed"],
                       target=r["hazard"], config=runner.public_config(cfg), data_manifests=["sonde_daily"],
                       artifacts=[str(run_dir)], notes="EW-v2 protocol §9 power count; Approved values; daily tier",
                       extra={k: r[k] for k in ("n_sites_with_onset", "n_basin_groups_with_onset",
                                                "n_issue_rows_valid_L3")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
