#!/usr/bin/env python
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Validation scores of the architecture-round-v2 arms (orbit_arch_v2_amendment.md §3.2, §3.6, §3.7).

    python scripts/train/orbitv2_units.py --arms a0 a1_c1 a1_c2 a1_c3 --v1-compare compare_<label> --register

Reads the 9 networks of each arm (written by ``orbitv2_arm.py``), builds the seed units and the 3-unit ensemble,
scores the four base targets on v1's evaluation cells of the base data version, and applies the A1 kill criterion
and the selection rule when all of their inputs are present. Every number this script computes is a validation
number of a new arm, so the whole script is refused before the lead's stamp of the amendment.

``--v1-compare`` names the ``orbit_compare.py`` run of the reported v1 model on the base data version; its
``val_row_preds.parquet`` (column SENTINEL) gives the v1 predictions for the kill criterion. ``--v1-reference``
names the file that records that model: the reviewed v1 freeze candidate or ``FREEZE_orbit.json`` (lead ruling
R-18a); its sha256 goes into the record. A kill criterion first computed against a candidate is recomputed once
``FREEZE_orbit.json`` exists, with ``--kill-first <units.json of the first run>``: the stricter outcome stands.

Not in this script: the parity comparators of §3.5 and Track N scores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.models.orbit import evaluate as E  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402
from sentinel.models.orbit_v2 import arm as ARM  # noqa: E402
from sentinel.models.orbit_v2 import devm as DM  # noqa: E402
from sentinel.models.orbit_v2 import stamp as ST  # noqa: E402
from sentinel.models.orbit_v2 import units as U  # noqa: E402

CI_METHOD = "role-block cluster bootstrap (B={b}, percentile)"


def v1_predictions(compare_run: str) -> dict:
    f = paths.RUNS_ROOT / "orbit" / compare_run / "val_row_preds.parquet"
    vp = pd.read_parquet(f, columns=["target", "row_id", "SENTINEL"])
    return {t: g.drop_duplicates("row_id").set_index("row_id")["SENTINEL"].astype(float) for t, g in vp.groupby("target")}


def v1_reference(path: str) -> dict:
    """R-18a: the file that records the reported v1 model, by sha256. Only the lead's ``FREEZE_orbit.json`` is the
    frozen reference; any other file is a candidate and the criterion is recomputed later."""
    import hashlib

    from sentinel.models.orbit import guard as G1

    f = Path(path).resolve()
    if not f.is_file():
        raise SystemExit(f"--v1-reference {path}: no such file")
    kind = "frozen" if f == Path(G1.FREEZE_FILE).resolve() else "candidate"
    return {"file": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "kind": kind}


def model_name(arm: str, model: str) -> str:
    member = model.split(":", 1)[1]
    return arm if member == "arm" else f"{arm}@{member}"


def register(res: dict, cfgs: dict, out_dir: Path, version: str, n_boot: int, stamp_row: str) -> int:
    from sentinel.core.registry import log_result

    exp = f"orbit_v2_val_{version}"
    common = {"workstream": PL.WORKSTREAM, "experiment": exp, "data_manifests": [f"orbit_{version}", "gloria"],
              "artifacts": [str(out_dir)],
              "notes": f"validation (development) cell; NOT a claim; test never scored; amendment stamp row {stamp_row}"}
    n = 0
    for r in res["scores"].itertuples():
        if r.level == "network" and r.cell != U.SELECT_CELL:
            continue
        log_result(model=model_name(r.arm, r.model), metric="mdsa", value=r.mdsa, split=f"{r.cell}@{version}", n=r.n,
                   ci_low=r.mdsa_ci_low, ci_high=r.mdsa_ci_high, ci_method=CI_METHOD.format(b=n_boot), target=r.target,
                   config=cfgs[r.arm], extra={"n_clusters": r.n_clusters, "level": r.level}, **common)
        n += 1
    k = res["kill"]
    if k is not None:
        for t, p in k["paired"].items():
            log_result(model=k["a1_configuration"], metric="mdsa_diff_vs_v1", value=p["diff"], n=p["n"],
                       split=f"{k['cell']}@{version}", ci_low=p["ci_low"], ci_high=p["ci_high"], target=t,
                       ci_method="paired " + CI_METHOD.format(b=n_boot) + "; negative = A1 better; common rows",
                       config=cfgs[k["a1_configuration"]],
                       extra={"p_value": p["p_value"], "mdsa_a1": p["mdsa_a"], "mdsa_v1": p["mdsa_b"],
                              "n_clusters": p["n_clusters"]}, **common)
            n += 1
        log_result(model=k["a1_configuration"], metric="a1_kill_criterion_passed", value=float(k["passed"]),
                   split=f"{k['cell']}@{version}", config=cfgs[k["a1_configuration"]],
                   extra={"gain_points": k["gain_points"], "margin_points": k["margin_points"],
                          "v1_reference": res.get("v1_reference"), "r18a": k.get("r18a")}, **common)
        n += 1
    s = res["selection"]
    if s is not None:
        log_result(model=s["v2"], metric="v2_selected", value=1.0, split=f"{U.SELECT_CELL}@{version}",
                   target=U.SELECT_TARGET, config=cfgs[s["v2"]],
                   extra={k_: s[k_] for k_ in ("candidates", "tied", "lowest", "tie_points", "a1_passed", "scores")},
                   **common)
        n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(U.ORDER), choices=list(U.ORDER))
    ap.add_argument("--v1-compare", default=None, help="orbit_compare run of v1 as frozen for the headline")
    ap.add_argument("--v1-reference", default=None,
                    help="the v1 freeze candidate or FREEZE_orbit.json that records the reported v1 model (R-18a)")
    ap.add_argument("--kill-first", default=None,
                    help="units.json of the kill criterion first computed against a v1 candidate (R-18a)")
    ap.add_argument("--tag", default="val")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--n-boot", type=int, default=E.N_BOOT)
    a = ap.parse_args(argv)
    st = ST.require_stamp("orbitv2_units: validation scores of the architecture-round-v2 arms")
    if bool(a.v1_compare) != bool(a.v1_reference):
        raise SystemExit("--v1-compare and --v1-reference go together (R-18a: the v1 reference is named by sha256)")
    ref = v1_reference(a.v1_reference) if a.v1_reference else None
    first = None
    if a.kill_first:
        if ref is None or ref["kind"] != "frozen":
            raise SystemExit("--kill-first is for the recomputation against FREEZE_orbit.json (R-18a)")
        import json

        first = json.loads(Path(a.kill_first).read_text())
        if (first.get("v1_reference") or {}).get("kind") != "candidate" or first.get("kill") is None:
            raise SystemExit("--kill-first: not the record of a kill criterion computed against a v1 candidate")
    cfgs = {arm: ARM.resolve_config(REPO / "configs" / "orbit_v2" / f"{arm}.yaml") for arm in a.arms}
    schemes = {c["fit_scheme"] for c in cfgs.values()}
    if len(schemes) != 1:
        raise SystemExit(f"the arms name different fit schemes: {sorted(schemes)}")
    root = paths.RUNS_ROOT / "orbit" / "v2"
    asms = {}
    for arm in a.arms:
        asms[arm] = U.assemble(root, arm)
        meta = asms[arm]["meta"]
        if meta["arm_file_sha256"] != cfgs[arm]["arm_file_sha256"]:
            raise SystemExit(f"{arm}: the networks were trained with another version of the arm file")
        if (meta["data"] or {}).get("version") != DM.VERSION:
            raise SystemExit(f"{arm}: the networks were trained on {meta['data']}, not on {DM.VERSION}")
    base_df, _, _, masks = PL.load_dev(DM.BASE_VERSION, schemes.pop())
    v1 = v1_predictions(a.v1_compare) if a.v1_compare else None
    res = U.evaluate_arms(asms, base_df, masks, v1, n_boot=a.n_boot)
    res["v1_reference"] = ref
    if first is not None:
        if res["kill"] is None:
            raise SystemExit("--kill-first: the kill criterion could not be recomputed (see the notes)")
        res["kill"] = U.kill_stricter(first["kill"], res["kill"])
        if "a0" in res["chla"]:
            res["selection"] = U.select_v2(res["chla"], res["kill"]["passed"])
    elif res["kill"] is not None and ref["kind"] == "candidate":
        res["notes"].append("R-18a: computed against a v1 candidate; recomputed against FREEZE_orbit.json if the "
                            "frozen v1 differs, and the stricter outcome stands")
    out_dir = paths.run_dir("orbit", f"v2/units_{a.tag}")
    res["scores"].to_csv(out_dir / "scores.csv", index=False)
    for name, rows in res["rows"].items():
        rows.to_parquet(out_dir / f"rows_{name}.parquet", index=False)
    PL.write_json(out_dir / "units.json", {"arms": a.arms, "v1_compare": a.v1_compare, "v1_reference": ref,
                                           "kill_first": a.kill_first, "chla": res["chla"],
                                           "kill": res["kill"], "selection": res["selection"], "notes": res["notes"],
                                           "stamp_row": st["row_id"], "n_boot": a.n_boot,
                                           "arm_file_sha256": {k: c["arm_file_sha256"] for k, c in cfgs.items()}})
    for arm in a.arms:
        PL.log_config_tried(arm, cfgs[arm], out_dir, "val_scored", {"v1_compare": a.v1_compare})
    pd.set_option("display.width", 250)
    sc = res["scores"]
    for cell in sc["cell"].unique():
        print(f"\n=== {cell} (validation, development; CI = block bootstrap) ===")
        print(sc[(sc.cell == cell) & (sc.level != "network")][["arm", "target", "model", "n", "n_clusters", "mdsa",
                                                                "mdsa_ci_low", "mdsa_ci_high"]].round(3).to_string(index=False))
    for k in ("kill", "selection"):
        print(f"{k}: {res[k]}")
    for note in res["notes"]:
        print("NOTE:", note)
    if a.register:
        n = register(res, cfgs, out_dir, DM.VERSION, a.n_boot, st["row_id"])
        print(f"registered {n} rows under experiment orbit_v2_val_{DM.VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
