#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Inner-row decisions of the M6 architecture-v2 round (amendment §7; lead rulings Q2): grid-point selection, gate
G-D1a and the inner table. **Train-role inner groups × train block only** — this script cannot read a validation-,
test-block or England row and writes no validation metric.

    python -B scripts/eval/sota_hydromet_v2_inner.py select --arm d1     # lowest seed-0 J among the grid points
    python -B scripts/eval/sota_hydromet_v2_inner.py gate_d1a            # 3-seed mean J_sonde, selected D1 vs (a)
    python -B scripts/eval/sota_hydromet_v2_inner.py table [--rescore]   # every model's inner numbers, one table

* ``select``: ties within 1e-4 go to the earlier grid point; written once to ``sota_v2/<arm>/selection.json`` and to
  the registry (``M6V2/decisions_v1``) before any seed 1–2, ablation or refit of the arm.
* ``gate_d1a``: ONE gate for all tracks. Grab rows stay only if the selected D1 lowers the 3-seed mean of J_sonde by
  ≥ 3 % against ablation (a). J_grab is reported and decides nothing. The outcome is written once to
  ``sota_v2/gates/g_d1a.json`` and to the registry **before any other model's variant is fitted**.
* ``table``: reads the run records; ``--rescore`` recomputes J_sonde and J_grab of every model from its
  ``inner_pred.npz`` with the shared criterion code on the pinned pool's inner rows (the same code and rows score
  every model) and fails if a recomputed number differs from the registered one.
* Every step passes :func:`sentinel.hydromet_sota.gso_mt.runner.assert_released` first (lead rulings L10) and
  refuses to combine runs made under another pin state or another code-pin sha256 (entry-1 lead rulings,
  decision 4); the release record goes into every file and registry row written here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)


def read_info(rdir: Path) -> dict | None:
    """The inner numbers of one finished run (neural, GBM, P or stack record), or ``None``."""
    if not (rdir / ".done").exists():
        return None
    for f, keys in (("train_info.json", ("best_J", "best_j_sonde", "best_j_grab")),
                    ("gbm_info.json", ("J", "j_sonde", "j_grab")), ("physics_info.json", ("J", "j_sonde", "j_grab")),
                    ("stack_info.json", ("J", "j_sonde", "j_grab"))):
        if (rdir / f).is_file():
            d = json.loads((rdir / f).read_text())
            return {"J": d.get(keys[0]), "j_sonde": d.get(keys[1]), "j_grab": d.get(keys[2]),
                    "best_epoch": d.get("best_epoch")}
    return None


def select(root: Path, arm: str, grid: list, release: dict) -> dict:
    js = {}
    for g in grid:
        info = read_info(root / arm / f"{arm}_{g}_s0")
        if info is None or info["J"] is None:
            raise SystemExit(f"{arm}.{g} seed 0 is not finished: the selection needs every grid point")
        js[g] = float(info["J"])
    from sentinel.hydromet_sota.gso_mt import gates as GT
    from sentinel.hydromet_sota.gso_mt import runner as VR

    comb = VR.assert_combinable([root / arm / f"{arm}_{g}_s0" for g in grid], release, f"select {arm}")
    return {"arm": arm, "selected": GT.select_point(js), "seed0_J": js, "tie_tolerance": GT.TIE_TOL,
            "rule": "lowest J among the grid points at seed 0; ties within 1e-4 to the earlier grid point (§7)",
            "release": release, "combined_runs": comb}


def gate_d1a(root: Path, selected: str, release: dict) -> dict:
    from sentinel.hydromet_sota.gso_mt import gates as GT
    from sentinel.hydromet_sota.gso_mt import runner as VR

    def js(point):
        out = {}
        for s in SEEDS:
            info = read_info(root / "d1" / f"d1_{point}_s{s}")
            if info is None or info["j_sonde"] is None:
                raise SystemExit(f"d1.{point} seed {s} is not finished: gate G-D1a needs 3 seeds of both")
            out[s] = info
        return out
    a, b = js(selected), js("abl_nograb")
    comb = VR.assert_combinable([root / "d1" / f"d1_{p}_s{s}" for p in (selected, "abl_nograb") for s in SEEDS],
                                release, "gate G-D1a")
    res = GT.g_d1a([a[s]["j_sonde"] for s in SEEDS], [b[s]["j_sonde"] for s in SEEDS])
    res.update({"selected_point": selected, "j_sonde_selected": [a[s]["j_sonde"] for s in SEEDS],
                "j_sonde_nograb": [b[s]["j_sonde"] for s in SEEDS],
                "j_grab_selected_reported_only": [a[s]["j_grab"] for s in SEEDS],
                "release": release, "combined_runs": comb})
    return res


def model_dirs(root: Path) -> dict:
    """model label → run directories of its seeds (every finished non-smoke arch-v2 run with inner numbers)."""
    out = {}
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ("smoke", "gates")):
        for r in sorted(p for p in arm_dir.iterdir() if p.is_dir() and not p.name.endswith("_refit")):
            if read_info(r) is None:
                continue
            label = r.name.rsplit("_s", 1)[0] if r.name.rsplit("_s", 1)[-1].isdigit() else r.name
            out.setdefault(label, []).append(r)
    return out


def write_once(path: Path, rec: dict, keys) -> bool:
    """Write ``rec`` unless the file exists; an existing file with a different decision is an error."""
    from sentinel.hydromet_sota.gso_mt import runner as VR

    if path.is_file():
        old = json.loads(path.read_text())
        if any(old.get(k) != rec.get(k) for k in keys):
            raise SystemExit(f"{path} exists with a different outcome; it is written once")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    VR.write_json(path, rec)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("select", "gate_d1a", "table"))
    ap.add_argument("--arm", default="d1", choices=("d1", "d2"))
    ap.add_argument("--rescore", action="store_true")
    a = ap.parse_args()
    import yaml

    from sentinel.hydromet_sota.gso_mt import runner as VR

    stamp = VR.assert_released(f"M6V2 inner {a.step}")
    rel = VR.release_fields(stamp, reader=VR.POOL_READER if a.step == "table" and a.rescore else VR.NO_POOL_READ)
    root = VR.ROOT
    if a.step == "select":
        cfg = yaml.safe_load((VR.CFG_DIR / f"{a.arm}.yaml").read_text())
        VR.assert_arm_released({"arm": a.arm})
        rec = select(root, a.arm, list(cfg["grid"]), rel)
        if write_once(root / a.arm / "selection.json", rec, ("selected",)):
            VR.log_decision("selection", f"{a.arm}", rec, rdir=root / a.arm, release=rel,
                            value=float(list(cfg["grid"]).index(rec["selected"]) + 1))
        print(json.dumps(rec, indent=1))
        return 0
    if a.step == "gate_d1a":
        sel = root / "d1" / "selection.json"
        if not sel.is_file():
            raise SystemExit("run `select --arm d1` first")
        rec = gate_d1a(root, json.loads(sel.read_text())["selected"], rel)
        if write_once(root / "gates" / "g_d1a.json", rec, ("passed", "label_form", "selected_point")):
            VR.log_decision("gate_G-D1a", "d1", rec, rdir=root / "gates", release=rel,
                            value=float(rec["relative_reduction"]),
                            notes=f"passed {rec['passed']}; label form of every model's claim configuration: "
                                  f"{rec['label_form']}")
        print(json.dumps(rec, indent=1))
        return 0
    rows = {}
    comb = VR.assert_combinable([d for dirs in model_dirs(root).values() for d in dirs], rel, "inner table")
    for label, dirs in model_dirs(root).items():
        infos = [read_info(d) for d in dirs]
        rows[label] = {"n_seeds": len(dirs), **{k: (float(np.mean([i[k] for i in infos]))
                                                    if all(i[k] is not None for i in infos) else None)
                                                for k in ("J", "j_sonde", "j_grab")},
                       "by_seed": {d.name: i for d, i in zip(dirs, infos)}}
    if a.rescore:
        from sentinel.hydromet_sota.gso_mt import criterion as C
        from sentinel.hydromet_sota.gso_mt import data as VD

        v2 = VD.load_v2_train(grab=True, require_pins=True)
        VR.assert_pool_read({"_release": rel}, v2)
        sc = VD.fit_scales(v2)
        Yd, inst, _ = VD.standardise_labels(v2, sc)
        ii, gi = v2.inner_idx, v2.grab_inner_idx
        j2 = [VD.DAILY.index("T_mean"), VD.DAILY.index("DO_mean")]
        for label, dirs in model_dirs(root).items():
            for d in dirs:
                z = np.load(d / "inner_pred.npz")
                if int(z["n_inner"]) != len(ii) or str(z["date_start"]) != str(v2.data.dates[0].date()):
                    raise SystemExit(f"{d}: inner predictions are not on the pinned pool's inner rows")
                js = C.j_sonde((z["daily"][:, :, :7] - sc.label_mean[:7]) / sc.label_std[:7], Yd[ii][:, :, :7],
                               sc.crit_factor())["j_sonde"]
                jg = None
                if "inst" in z.files and int(z["n_grab_inner"]) == len(gi):
                    jg = C.j_grab((z["inst"] - sc.label_mean[j2]) / sc.label_std[j2], inst[gi])["j_grab"]
                reg = rows[label]["by_seed"][d.name]
                rows[label]["by_seed"][d.name] = {**reg, "rescored_j_sonde": js, "rescored_j_grab": jg}
                for got, want in ((js, reg["j_sonde"]), (jg, reg["j_grab"])):
                    if got is not None and want is not None and abs(got - want) > 1e-4 * max(1.0, abs(want)):
                        raise SystemExit(f"{d.name}: rescored {got} differs from the registered {want}")
    VR.write_json(root / "inner_table.json", {"rows": rows, "rescored": bool(a.rescore), "release": rel,
                                              "combined_runs": comb,
                                              "split": "inner (train-role inner groups x train block)"})
    for k, r in rows.items():
        print(f"{k:28s} seeds {r['n_seeds']}  J {r['J']}  J_sonde {r['j_sonde']}  J_grab {r['j_grab']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
