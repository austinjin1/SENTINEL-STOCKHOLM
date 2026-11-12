#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Event-layer label feasibility [S1], M1 Sonde and the M6 HydroMet sensor tracks (plan §3.1).

Guarded loader ``sentinel.models.sonde.common.prepare_data(blocks=("train","val"))`` on train- and val-role
stream sites of the FINAL ``sonde_split.json`` only (test-role sites and Test-A/Test-B are never requested;
the guard refuses them). Approved values only. Labels come from ``sentinel.eval.events`` (E1a/E2a/E2b/E4a/E5a
delegate to ``hazards.py``). Per event × block × site role: event days, onsets, sites / basin groups observed
and with onsets, observed site-years; E7 also on the coldwater-eligible population.

    python -B scripts/eval/events_feasibility_sonde.py [--register]
"""

from __future__ import annotations

from sentinel.data.reserved import assert_no_reserved  # noqa: E402

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import events_feasibility_common as C

from sentinel.data.sonde_splits import TIME_BLOCKS
from sentinel.eval import events as EV
from sentinel.models.sonde import common
from sentinel.models.sonde.data import TARGETS

STREAM_TYPES = {"ST", "ST-TS", "ST-CA", "ST-DCH"}
EXTRA_VARS = ["ph_min", "ph_max", "temp_c_max"]
CODES = ["E1a", "E1a_t4", "E1a_t2", "E1b", "E2a", "E2b", "E3a", "E3b", "E4a", "E5a", "E6", "E7"]


def main() -> int:
    a = C.parser(__doc__).parse_args()
    out = C.out_dir(a, "sonde")
    split, info = common.load_split(common.FINAL_SPLIT)
    if not info.get("final"):
        raise RuntimeError("FINAL split required")
    reg = common.site_registry()
    have = {p.stem for p in common.SITE_DIR.glob("*.parquet")}
    sites = sorted(s for r in ("train", "val") for s in split.sites(r)
                   if s in have and s in reg.index and reg.loc[s, "site_type"] in STREAM_TYPES)
    assert_no_reserved(sites, "events_feasibility_sonde")   # reserved new-site block (sentinel.data.reserved)
    C.assert_allowed([split.site_role[s] for s in sites], {"train", "val"}, "sonde site roles")
    data = common.prepare_data(split, info, sites, blocks=("train", "val"),
                               variables=list(dict.fromkeys(TARGETS + EXTRA_VARS)))
    common.attach_chloride(data)
    cube = data.cube
    C.assert_allowed(data.roles, {"train", "val"}, "loaded site roles")
    C.assert_dates_within(cube.dates, hi=TIME_BLOCKS["val"][1], what="cube dates")
    dates = pd.DatetimeIndex(cube.dates)
    roles = np.asarray(data.roles)
    basin = {str(k): str(v) for k, v in split.basin_group.items()}
    S, T = len(cube.sites), len(dates)

    def approved(v):
        j = cube.var_index(v)
        return np.where(cube.prov[:, :, j], np.nan, cube.y[:, :, j].astype(float))

    raw = {v: approved(v) for v in ("do_mgl_min", "do_mgl_mean", "cl_mgl_4dmean", "cl_mgl_max", "chl_ugl_mean",
                                    "no3no2_mgl_mean", "turb_fnu_mean", "ph_min", "ph_max", "temp_c_max")}
    ev = {c: np.zeros((S, T), bool) for c in CODES}
    on = {c: np.zeros((S, T), bool) for c in CODES}
    obs = {c: np.zeros((S, T), bool) for c in CODES}
    elig = np.zeros(S, bool)
    for s in range(S):
        frame = pd.DataFrame({k: v[s] for k, v in raw.items()}, index=dates)
        elig[s] = EV.coldwater_eligible(dates, raw["temp_c_max"][s], TIME_BLOCKS["train"])
        for c in CODES:
            x = EV.derive_series(frame, c).to_numpy(float)
            fin = np.isfinite(x)
            if not fin.any():
                continue
            obs[c][s] = fin
            ev[c][s] = EV.event_state(x, c) == 1
            on[c][s] = EV.event_onsets(x, c)[1]
    for k in (ev, on, obs):
        k["E7_coldwater"] = k["E7"] & elig[:, None]
    rows, table = [], []
    for c in CODES + ["E7_coldwater"]:
        for block in ("train", "val"):
            lo, hi = (pd.Timestamp(t) for t in TIME_BLOCKS[block])
            tm = np.asarray((dates >= lo) & (dates <= hi))
            for role in ("train", "val", "trainval"):
                sm = np.isin(roles, ["train", "val"]) if role == "trainval" else roles == role
                e, o, b = ev[c][np.ix_(sm, tm)], on[c][np.ix_(sm, tm)], obs[c][np.ix_(sm, tm)]
                st = np.asarray(cube.sites)[sm]
                with_on = st[o.sum(axis=1) > 0]
                with_ev = st[e.sum(axis=1) > 0]
                st_obs = st[b.any(axis=1)]
                years = dates[tm].year.to_numpy()
                n_sy = int(sum(np.unique(years[b[i]]).size for i in range(b.shape[0])))
                r = {"event": c, "block": block, "role": role, "n_event_days": int(e.sum()), "n_obs_days": int(b.sum()),
                     "n_onsets": int(o.sum()), "n_sites_obs": int(st_obs.size),
                     "n_basins_obs": len({basin.get(x) for x in st_obs}), "n_sites_with_onset": int(with_on.size),
                     "n_basins_with_onset": len({basin.get(x) for x in with_on}),
                     "n_sites_with_event": int(with_ev.size), "n_basins_with_event": len({basin.get(x) for x in with_ev}),
                     "n_site_years_obs": n_sy}
                if block == "val":
                    r["power_flag_onsets"] = EV.power_flag(r["n_onsets"], r["n_basins_with_onset"])
                table.append(r)
                split_lbl = f"{block}block_{role}roles/sonde_split_{info['sha256'][:12]}"
                extra = {k: v for k, v in r.items() if k not in ("event", "block", "role")}
                for metric in ("n_onsets", "n_event_days"):
                    rows.append({"model": "M1_sonde", "target": c, "split": split_lbl, "metric": metric,
                                 "value": r[metric], "n": r["n_obs_days"], "extra": extra})
    res = pd.DataFrame(table)
    f = C.write(res, out, "sonde_event_counts", {"split_sha256": info["sha256"], "n_sites": S,
                                                 "roles": pd.Series(roles).value_counts().to_dict(),
                                                 "n_coldwater_eligible": int(elig.sum()), "approved_only": True,
                                                 "events_version": EV.__version__, "created": C.stamp()})
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))
    print("coldwater-eligible sites:", int(elig.sum()), "| written:", f)
    if a.register:
        n = C.register(rows, script=__file__, artifacts=[f], data_manifests=["sonde_daily", "chloride_derived"])
        print(f"registered {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
