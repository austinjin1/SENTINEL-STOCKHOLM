# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build ``docs/preregistration/FREEZE_lite_candidate.json`` for the lead's review (protocol §8, §8a). This is never
the freeze. Its status is "CANDIDATE" and ``scoring_code_sha256`` is "PENDING" (review C2). The lead writes
``FREEZE_lite.json`` from it with status FROZEN, ``authorised_by`` / ``authorised_at`` and the scoring-code hash at
freeze time.

The candidate records:
- every model the single look scores, with the sha256 of each checkpoint seed, ``pool.json`` (if any) and the
  ONNX deployable;
- the export report (budgets);
- the Lite-safe teacher manifest;
- the data pins (``testlook.compute_data_pins``);
- B, device and package versions;
- the val-selected band-ratio forms;
- the configuration count with its breakdown.

    python -B scripts/eval/lite_freeze_candidate.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sentinel.models.lite import runner as R

R.cpu_threads(2)

from sentinel.core import paths  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402
from sentinel.models.lite import testlook as T  # noqa: E402

L = paths.RUNS_ROOT / "lite"
OUT = paths.REPO_ROOT / "docs" / "preregistration" / "FREEZE_lite_candidate.json"
PRIMARY = L / "student_mnv4s_raw_crop128_kd03_20261108-152950"
NOKD_OLD = L / "student_mnv4s_raw_crop128_20261024-224754"
TORCH = {"primary": (PRIMARY, (0, 1, 2)),
         "robust_variant": (L / "student_mnv4s_raw_crop128_kd03_glint_20261108-201455", (0,)),
         "gloria_ablation": (L / "student_mnv4s_raw_gloria_20261024-230636", (0, 1, 2)),
         "gloria_base": (L / "student_mnv4s_raw_20261024-221134", (0, 1, 2))}


def latest(pattern: str):
    c = sorted(p for p in L.glob(pattern) if (p / "summary.json").exists())
    return c[-1] if c else None


def main() -> int:
    nokd = latest("student_mnv4s_raw_crop128_rerun_*")
    rgb = latest("student_mnv4s_raw_crop128_kd03_rgb_*")
    torch_models = dict(TORCH)
    torch_models["nokd"] = (nokd or NOKD_OLD, (0, 1, 2))
    models, weights, not_run = {}, {}, {}
    if rgb:
        torch_models["rgb_ablation"] = (rgb, (0, 1, 2))
    else:
        not_run["rgb_ablation"] = "RGB-only ablation run not finished at candidate time"
    for name, (run, seeds) in torch_models.items():
        models[name] = {"kind": "torch", "run": str(run), "seeds": list(seeds)}
        for s in seeds:
            weights[f"{name}_seed{s}"] = G.artifact_sha256(run / f"model_seed{s}.pt")
        if (run / "pool.json").exists():
            weights[f"{name}_pool_json"] = G.artifact_sha256(run / "pool.json")
    ex = PRIMARY / "export_seed0"
    for name, f in (("deployable_int8_seed0", "lite_int8.onnx"), ("deployable_fp32_seed0", "lite_fp32.onnx")):
        models[name] = {"kind": "onnx", "path": str(ex / f), "crop": 128}
        weights[name] = G.artifact_sha256(ex / f)
    weights["export_report_seed0"] = G.artifact_sha256(ex / "export_report.json")
    # event track (protocol §13): released events.py v1 + the hashed val-stage artifacts
    ev_dir = L / "events_e4a"
    events = None
    if (ev_dir / "calibrators.json").exists():
        from sentinel.eval import events as EVM

        events = {"events_module": str(Path(EVM.__file__)), "events_module_sha256": T.sha_file(Path(EVM.__file__)),
                  "heads": str(ev_dir / "heads.json"), "comparators": str(ev_dir / "comparators.json"),
                  "calibrators": str(ev_dir / "calibrators.json"),
                  "added": "2026-11-09, after val model selection; the selection score S used Secchi/turbidity/TSS "
                           "only (scripts/eval/lite_select.py TARGETS) — chl-a was a training target of every "
                           "configuration but never entered selection"}
        for k in ("heads", "comparators", "calibrators"):
            weights[f"events_{k}"] = G.artifact_sha256(ev_dir / f"{k}.json")
    else:
        not_run["events"] = "event val stage (scripts/eval/events_lite.py) not run: sentinel/eval/events.py v1 not released"
    info = json.loads((L / "baselines_20261024-221014" / "summary.json").read_text())["info"]
    tm = L / "teacher_litesafe" / "manifest.json"
    split = G.load_split()
    cand = {
        "status": "CANDIDATE — not a freeze; refused by guard/testlook. The lead writes FREEZE_lite.json "
                  "(status FROZEN, authorised_by/at, scoring_code_sha256 at freeze time).",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split_sha256": split["_sha256"],
        "scoring_code_sha256": "PENDING",
        "scoring_code_sha256_at_candidate": G.scoring_code_sha256(),
        "scoring_code_files": len(G.frozen_files()),
        "weights_sha256": weights,
        "models": models,
        "not_run": not_run,
        "deployable": "deployable_int8_seed0",
        "h_g6_carrier": "deployable_int8_seed0 (protocol §8a)",
        "export_report": str(ex / "export_report.json"),
        "teacher_manifest": str(tm),
        "teacher_manifest_sha256": G.artifact_sha256(tm),
        "band_ratio_forms": {t: info[f"band_ratio/{t}"]["selected_form"] for t in ("secchi", "turbidity", "tss", "chla")},
        "baselines_run": str(L / "baselines_20261024-221014"),
        "n_boot": 2000,
        "device": "cpu",
        "packages": G.package_versions(),
        "data_snapshots": T.compute_data_pins(split),
        "h2_comparator": ("retrained under current code (review W1): " + str(nokd)) if nokd else
                         "DEVIATION: 2026-10-24 run (re-run not available at candidate time)",
        "configurations_tried": {"original_track_selection_candidates": 16,
                                 "breakdown": "2026-10-24: raw, film, scene, crop128, effl0, gloria, spectral, scenefilm; "
                                              "2026-11-08/09: kd03, kd1, ep20, lrbb2e4, crop128_secchi, kd03_medpool, "
                                              "kd03_glint, kd03_secchi",
                                 "not_counted": "no-KD re-run (W1, comparator), RGB-only ablation (pre-registered "
                                                "ablation), 4 crashed runs (TARGETS bug, re-run), deleted partial "
                                                "run 221015 (duplicate)"},
        "events": events,
        "single_test_run_command": "python -B scripts/eval/lite_score_test.py --allow-test",
    }
    OUT.write_text(json.dumps(cand, indent=1) + "\n")
    print(json.dumps({k: cand[k] for k in ("scoring_code_sha256_at_candidate", "h2_comparator", "not_run")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
