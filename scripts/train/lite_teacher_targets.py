# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Generate distillation targets for SENTINEL-Lite from the frozen M2 SENTINEL-Orbit teacher
(docs/model_cards/orbit.md, ``sentinel.models.orbit.teacher.OrbitTeacher``).

Only **train-role** Lite chips with an S2 teacher chip (±1 d) are processed. Per S2 water pixel:
``mu`` = teacher per-pixel log10 median (imagery only, context off: no location prior); ``sigma`` = the
chip-level teacher spread (imgonly var) of that chip — the teacher has no per-pixel variance — and ``trust`` = 1
(per-target trust can be lowered with --trust). Output: parquet in distill.load_kd_tensors format.

Leakage check (protocol §5, review W8; ``distill.flag_overlap``): every row the teacher fitted on (its
``seen_points``) is compared with every Lite site (same site id, <= 2 km, same HydroLAKES lake). The script
refuses a teacher that trained on any Lite ``test``/``test_time`` water body, and records the val overlap.
The stock M2 teacher (``--manifest orbit``) fails this check (2026-11-08: 131/420 test water bodies), so the
default is the Lite-safe retrain (``scripts/train/lite_teacher_litesafe.py``).

    python -m sentinel.core.gpu --name lite_teacher --prefer 2 -- python -B scripts/train/lite_teacher_targets.py
"""

from __future__ import annotations

import argparse
import json

from sentinel.models.lite import runner as R

R.cpu_threads(2)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402

# Lite target -> Orbit target name
ORBIT_NAME = {"secchi": "secchi", "turbidity": "turbidity", "tss": "tss", "chla": "chla"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trust", default="", help="per-target trust, e.g. 'turbidity=0.5,tss=0.5'")
    ap.add_argument("--max-chips", type=int, default=0, help="debug only (smoke)")
    ap.add_argument("--track", default="original", choices=["original", "national"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--manifest", default="litesafe",
                    help="'litesafe' ($SENTINEL_RUNS/lite/teacher_litesafe/manifest.json), 'orbit' (stock M2) or a path")
    a = ap.parse_args(argv)
    from sentinel.core import paths
    from sentinel.models.lite import distill as KD
    from sentinel.models.orbit.teacher import TEACHER_MANIFEST, OrbitTeacher

    man_path = {"orbit": TEACHER_MANIFEST,
                "litesafe": paths.RUNS_ROOT / "lite" / "teacher_litesafe" / "manifest.json",
                "litesafe_national": paths.RUNS_ROOT / "lite" / "teacher_litesafe_national" / "manifest.json"}.get(a.manifest, a.manifest)
    man_path = __import__("pathlib").Path(man_path)
    if not man_path.exists():
        raise SystemExit(f"teacher manifest {man_path} does not exist")
    man = json.loads(man_path.read_text())
    # ---- W8 overlap check (before any teacher output is produced)
    from sentinel.models.lite import national as N

    split0 = G.load_split() if a.track == "original" else N.load_split()
    labf = None if a.track == "original" else N.labels_path()
    prot = ("test", "test_time") if a.track == "original" else N.NATIONAL_PROTECTED
    if man.get("seen_points"):
        if G.artifact_sha256(man["seen_points"]) != man["seen_points_sha256"]:
            raise RuntimeError("teacher seen_points sha256 mismatch")
        seen = pd.read_parquet(man["seen_points"])
    else:
        seen = KD.orbit_seen_points("base")
    flagged = KD.flag_overlap(KD.lite_sites(split0, labf), seen)
    overlap = KD.overlap_report(flagged)
    print(json.dumps({r: {k: v[k] for k in ("water_bodies", "water_bodies_overlap")} for r, v in overlap.items()}))
    KD.assert_teacher_disjoint(flagged, prot)
    if a.device == "cuda":
        R.setup_gpu("lite_teacher")
    trust = {t: 1.0 for t in D.TARGETS}
    for kv in filter(None, a.trust.split(",")):
        k, v = kv.split("=")
        trust[k] = float(v)
    split = split0
    tr = D.role_frame(split, ("train",), training=True) if a.track == "original" else N.role_frame(split, ("train",))
    G.assert_no_protected(tr["role"])
    tr = tr[tr["s2_chip_id"].notna()].reset_index(drop=True)
    if a.max_chips:
        tr = tr.head(a.max_chips)
    for c, sha in zip(man["checkpoints"], man["sha256"]):
        if G.artifact_sha256(c) != sha:
            raise RuntimeError(f"teacher checkpoint {c} sha256 mismatch")
    teacher = OrbitTeacher(man["checkpoints"], device=a.device)
    store = D.teacher_store()
    chip_df = teacher.predict_store(store.root, chip_ids=tr["s2_chip_id"].unique().tolist(), workers=2)
    chip_df = chip_df.set_index("chip_id")
    rows = []
    for sid in tr["s2_chip_id"].unique():
        meta = store.index.set_index("chip_id").loc[sid]
        maps = teacher.predict_pixels(store.load(sid), meta, ctx_off=True)
        for t in D.TARGETS:
            m = maps.get(ORBIT_NAME[t])
            if m is None:
                continue
            var = chip_df.loc[sid].get(f"imgonly_{ORBIT_NAME[t]}_log10_var", np.nan) if sid in chip_df.index else np.nan
            rr, cc = np.nonzero(np.isfinite(m))
            for r_, c_ in zip(rr, cc):
                rows.append((sid, int(r_), int(c_), t, float(m[r_, c_]), float(np.sqrt(var)) if np.isfinite(var)
                             else np.nan, trust[t]))
    out = pd.DataFrame(rows, columns=["s2_chip_id", "row", "col", "target", "mu", "sigma", "trust"])
    out = out[np.isfinite(out["sigma"])]
    run_dir = R.run_directory("teacher_targets" + ("" if a.track == "original" else "_national"), bool(a.max_chips))
    out.to_parquet(run_dir / "teacher_targets.parquet", index=False)
    rep = {"teacher_manifest": str(man_path), "teacher_label": man.get("label"), "teacher_ckpts": teacher.ckpts,
           "teacher_sha256": man["sha256"], "w8_overlap": overlap,
           "w8_val_overlap_water_bodies": KD.overlap_water_bodies(flagged, ("val", "val_n")), "track": a.track, "n_rows": int(len(out)),
           "n_s2_chips": int(out["s2_chip_id"].nunique()), "n_lite_train_chips": int(len(tr)), "trust": trust,
           "split_sha256": split["_sha256"],
           "note": "turbidity units: Orbit may mix NTU/FNU (orbit.md); Lite labels are NTU only"}
    R.write_json(run_dir / "report.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
