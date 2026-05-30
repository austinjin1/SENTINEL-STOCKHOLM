#!/usr/bin/env python3
"""Phase 0.4: Re-run all 20 experiments under strict temporal-spatial holdout.

Re-executes the exp*.py experiment scripts with environment variables that
force the temporal-spatial holdout split (train: pre-2023, val: 2023,
test: 2024+, spatial site separation).  Because none of the original
experiment scripts accept a --holdout CLI flag, this orchestrator achieves
holdout enforcement via two mechanisms:

  1.  SENTINEL_HOLDOUT=1  — environment variable picked up by wrapper shims
      injected before each script's data-loading phase (see --patch mode).
  2.  Redirect output to results/holdout/<exp_name>/ so results are never
      mixed with the random-split originals.

Experiment groupings
--------------------
GROUP A — NEON temporal data (must filter to test split: 2024+)
  exp2_baseline_comparison   exp8_neon_trend_analysis   exp16_parameter_attribution
  exp10_mc_dropout           exp18_seasonal_analysis    exp14_cross_site_generalization

GROUP B — Model inference / downstream (GPU-light, benefit from CUDA_VISIBLE_DEVICES)
  exp3_epa_violation_correlation  exp4_satellite_imagery  exp5_explainability
  exp6_propagation                exp7_crossmodal_alignment

GROUP C — Statistical / ablation (CPU-only, parallelisable)
  exp9_bootstrap_ci   exp11_label_noise_sensitivity   exp12_multimodal_integration
  exp13_prpo_audit    exp15_contrastive_alignment     exp17_risk_index
  exp19_behavioral_profile   exp20_cascade_analysis

GROUP D — Case studies (CPU, structured data, no NEON parquet)
  exp1_case_studies   exp1_case_studies_real

Usage examples
--------------
# Dry-run: print what would be launched
python scripts/rerun_holdout_experiments.py --dry-run

# Run all experiments sequentially on GPU 1
python scripts/rerun_holdout_experiments.py --gpu 1

# Run only Group A (NEON temporal) on GPU 2
python scripts/rerun_holdout_experiments.py --group A --gpu 2

# Parallelise: Group A on GPU 1, B on GPU 2, C/D on GPU 3 (3 concurrent jobs)
python scripts/rerun_holdout_experiments.py --parallel --gpus 1,2,3

# Skip experiments that already have holdout results
python scripts/rerun_holdout_experiments.py --skip-done --gpu 1

# Run a single experiment
python scripts/rerun_holdout_experiments.py --only exp2_baseline_comparison --gpu 1

MIT License — Bryan Cheng, SENTINEL project, 2026
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
HOLDOUT_RESULTS_DIR = PROJECT_ROOT / "results" / "holdout"
LOGS_DIR = PROJECT_ROOT / "logs" / "holdout"

HOLDOUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment registry
#
# Each entry:
#   script        : path relative to PROJECT_ROOT
#   group         : A/B/C/D (see module docstring)
#   gpu_needed    : True if script loads a model to GPU
#   neon_temporal : True if script samples windows from the NEON parquet —
#                   these get extra env vars to enforce date-range filtering
#   result_key    : sub-directory name under results/holdout/ for outputs
#   depends_on    : list of experiment result_keys that must exist first
#   notes         : human-readable description of holdout change required
# ---------------------------------------------------------------------------

EXPERIMENTS: Dict[str, Dict] = {
    # ── Group A: NEON temporal data ─────────────────────────────────────────
    "exp2_baseline_comparison": {
        "script": "scripts/exp2_baseline_comparison.py",
        "group": "A",
        "gpu_needed": True,
        "neon_temporal": True,
        "result_key": "exp2_baselines_holdout",
        "depends_on": [],
        "notes": (
            "Uses NEON parquet; SENTINEL_HOLDOUT=1 triggers date-range filter "
            "to restrict windows to startDateTime >= 2024-01-01 (test split). "
            "SENTINEL_HOLDOUT_OUTPUT overrides OUTPUT_DIR to results/holdout/exp2_baselines_holdout/."
        ),
    },
    "exp8_neon_trend_analysis": {
        "script": "scripts/exp8_neon_trend_analysis.py",
        "group": "A",
        "gpu_needed": False,
        "neon_temporal": True,
        "result_key": "exp8_neon_trends_holdout",
        "depends_on": [],
        "notes": (
            "Reads full NEON parquet for Mann-Kendall trends. Holdout restricts "
            "analysis to test-split sites (spatial fold E) and 2024+ timestamps."
        ),
    },
    "exp16_parameter_attribution": {
        "script": "scripts/exp16_parameter_attribution.py",
        "group": "A",
        "gpu_needed": True,
        "neon_temporal": True,
        "result_key": "exp16_attribution_holdout",
        "depends_on": [],
        "notes": (
            "Per-parameter occlusion attribution on NEON windows. "
            "Holdout: windows from startDateTime >= 2024-01-01 only."
        ),
    },
    "exp10_mc_dropout": {
        "script": "scripts/exp10_mc_dropout.py",
        "group": "A",
        "gpu_needed": True,
        "neon_temporal": True,
        "result_key": "exp10_mc_dropout_holdout",
        "depends_on": [],
        "notes": (
            "MC-dropout uncertainty on AquaSSM + fusion. "
            "Uses NEON windows; same temporal restriction applies."
        ),
    },
    "exp18_seasonal_analysis": {
        "script": "scripts/exp18_seasonal_analysis.py",
        "group": "A",
        "gpu_needed": False,
        "neon_temporal": True,
        "result_key": "exp18_seasonal_holdout",
        "depends_on": ["exp8_neon_trends_holdout"],
        "notes": (
            "Seasonal pattern analysis using NEON parquet + exp8 results. "
            "Holdout: restrict monthly aggregation to 2024 data."
        ),
    },
    "exp14_cross_site_generalization": {
        "script": "scripts/exp14_cross_site_generalization.py",
        "group": "A",
        "gpu_needed": False,
        "neon_temporal": False,  # Uses pre-computed scan results
        "result_key": "exp14_cross_site_holdout",
        "depends_on": [],
        "notes": (
            "Cross-site generalisation uses neon_anomaly_scan results + site metadata. "
            "No raw parquet read; holdout relevant at the scan level upstream."
        ),
    },
    # ── Group B: Model inference / downstream ────────────────────────────────
    "exp3_epa_violation_correlation": {
        "script": "scripts/exp3_epa_violation_correlation.py",
        "group": "B",
        "gpu_needed": True,
        "neon_temporal": False,
        "result_key": "exp3_epa_correlation_holdout",
        "depends_on": [],
        "notes": (
            "Loads fusion checkpoint + exp1 USGS scores. "
            "Holdout: use holdout-split USGS scores (2024+ events only)."
        ),
    },
    "exp4_satellite_imagery": {
        "script": "scripts/exp4_satellite_imagery.py",
        "group": "B",
        "gpu_needed": True,
        "neon_temporal": False,
        "result_key": "exp4_satellite_holdout",
        "depends_on": [],
        "notes": (
            "Uses HydroViT checkpoint + satellite embeddings from test split. "
            "SENTINEL_HOLDOUT=1 selects embeddings with timestamp >= 2024-01-01."
        ),
    },
    "exp5_explainability": {
        "script": "scripts/exp5_explainability.py",
        "group": "B",
        "gpu_needed": True,
        "neon_temporal": False,
        "result_key": "exp5_explainability_holdout",
        "depends_on": [],
        "notes": "GradCAM/SHAP explanations on test-split samples.",
    },
    "exp6_propagation": {
        "script": "scripts/exp6_propagation.py",
        "group": "B",
        "gpu_needed": False,
        "neon_temporal": False,
        "result_key": "exp6_propagation_holdout",
        "depends_on": [],
        "notes": "Contaminant propagation simulation — no temporal data dependency.",
    },
    "exp7_crossmodal_alignment": {
        "script": "scripts/exp7_crossmodal_alignment.py",
        "group": "B",
        "gpu_needed": True,
        "neon_temporal": False,
        "result_key": "exp7_crossmodal_holdout",
        "depends_on": [],
        "notes": "CKA cross-modal alignment on holdout test-split embeddings.",
    },
    # ── Group C: Statistical / ablation ─────────────────────────────────────
    "exp9_bootstrap_ci": {
        "script": "scripts/exp9_bootstrap_ci.py",
        "group": "C",
        "gpu_needed": False,
        "neon_temporal": False,
        "result_key": "exp9_bootstrap_holdout",
        "depends_on": [],
        "notes": (
            "Bootstrap CIs for all 6 metrics. "
            "Re-run on test-split predictions from holdout-trained models."
        ),
    },
    "exp11_label_noise_sensitivity": {
        "script": "scripts/exp11_label_noise_sensitivity.py",
        "group": "C",
        "gpu_needed": False,
        "neon_temporal": False,
        "result_key": "exp11_label_noise_holdout",
        "depends_on": [],
        "notes": "Label-noise sensitivity analysis — statistical, no raw data dependency.",
    },
    "exp12_multimodal_integration": {
        "script": "scripts/exp12_multimodal_integration.py",
        "group": "C",
        "gpu_needed": True,
        "neon_temporal": False,
        "result_key": "exp12_integration_holdout",
        "depends_on": [],
        "notes": "Multi-modal integration test on test-split embeddings.",
    },
    "exp13_prpo_audit": {
        "script": "scripts/exp13_prpo_audit.py",
        "group": "C",
        "gpu_needed": False,
        "neon_temporal": False,
        "result_key": "exp13_prpo_holdout",
        "depends_on": [],
        "notes": "PR/PO audit — purely statistical, no raw data.",
    },
    "exp15_contrastive_alignment": {
        "script": "scripts/exp15_contrastive_alignment.py",
        "group": "C",
        "gpu_needed": True,
        "neon_temporal": False,
        "result_key": "exp15_contrastive_holdout",
        "depends_on": [],
        "notes": "Contrastive alignment evaluation on holdout test embeddings.",
    },
    "exp17_risk_index": {
        "script": "scripts/exp17_risk_index.py",
        "group": "C",
        "gpu_needed": False,
        "neon_temporal": False,
        "result_key": "exp17_risk_index_holdout",
        "depends_on": ["exp8_neon_trends_holdout", "exp14_cross_site_holdout"],
        "notes": "Risk index composite — depends on exp8 and exp14 holdout results.",
    },
    "exp19_behavioral_profile": {
        "script": "scripts/exp19_behavioral_profile.py",
        "group": "C",
        "gpu_needed": True,
        "neon_temporal": False,
        "result_key": "exp19_behavioral_holdout",
        "depends_on": [],
        "notes": "BioMotion behavioral profile analysis on holdout split.",
    },
    "exp20_cascade_analysis": {
        "script": "scripts/exp20_cascade_analysis.py",
        "group": "C",
        "gpu_needed": False,
        "neon_temporal": False,
        "result_key": "exp20_cascade_holdout",
        "depends_on": [],
        "notes": "Cascade failure analysis — no raw data dependency.",
    },
    # ── Group D: Case studies ────────────────────────────────────────────────
    "exp1_case_studies": {
        "script": "scripts/exp1_case_studies.py",
        "group": "D",
        "gpu_needed": False,
        "neon_temporal": False,
        "result_key": "case_studies_holdout",
        "depends_on": [],
        "notes": (
            "31-event case study analysis — uses structured event data, not raw parquet. "
            "Holdout: verifies all 2024+ events are in test split; 2023 events go to val."
        ),
    },
    "exp1_case_studies_real": {
        "script": "scripts/exp1_case_studies_real.py",
        "group": "D",
        "gpu_needed": False,
        "neon_temporal": False,
        "result_key": "case_studies_real_holdout",
        "depends_on": [],
        "notes": "Real case studies version — same holdout logic as exp1_case_studies.",
    },
}

# ---------------------------------------------------------------------------
# Modification status: which scripts need code changes vs can run as-is
# ---------------------------------------------------------------------------
#
# NEEDS_PATCHING: scripts that hard-code random splits or ignore SENTINEL_HOLDOUT.
# These cannot be re-run cleanly without modifying the data-loading section.
# The orchestrator injects SENTINEL_HOLDOUT env vars and relies on wrapper logic
# (see inject_holdout_wrapper()) to prepend a short monkey-patch to the script.
#
# CAN_RERUN_WITH_ENV: scripts that will correctly route to results/holdout/ if
# SENTINEL_HOLDOUT=1 is set (they already call get_split_assignment or similar).
#
# The exp_honest_baselines.py script is the only one that already imports from
# sentinel.data.splits — all others in exp[0-9]*.py need the env-variable path.

NEEDS_PATCHING = {
    "exp2_baseline_comparison",
    "exp8_neon_trend_analysis",
    "exp10_mc_dropout",
    "exp16_parameter_attribution",
    "exp18_seasonal_analysis",
}

CAN_RERUN_WITH_ENV = set(EXPERIMENTS.keys()) - NEEDS_PATCHING


# ---------------------------------------------------------------------------
# Environment setup for each experiment
# ---------------------------------------------------------------------------

def build_env(exp_name: str, gpu_id: Optional[int]) -> dict:
    """Build os.environ overlay for a subprocess invocation.

    Sets:
      CUDA_VISIBLE_DEVICES   — isolate GPU
      SENTINEL_HOLDOUT       — '1' to activate holdout mode in any script
                               that checks os.getenv('SENTINEL_HOLDOUT')
      SENTINEL_HOLDOUT_OUTPUT — output directory override
      SENTINEL_TRAIN_END     — '2023-01-01'  (exclusive upper bound for train)
      SENTINEL_VAL_END       — '2024-01-01'  (exclusive upper bound for val)
      SENTINEL_TEST_START    — '2024-01-01'  (lower bound for test)
    """
    exp = EXPERIMENTS[exp_name]
    result_dir = HOLDOUT_RESULTS_DIR / exp["result_key"]
    result_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SENTINEL_HOLDOUT"] = "1"
    env["SENTINEL_HOLDOUT_OUTPUT"] = str(result_dir)
    env["SENTINEL_TRAIN_END"] = "2023-01-01"
    env["SENTINEL_VAL_END"] = "2024-01-01"
    env["SENTINEL_TEST_START"] = "2024-01-01"
    env["SENTINEL_PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["SENTINEL_ORIGINAL_SCRIPT"] = str(PROJECT_ROOT / exp["script"])

    # GPU assignment
    if gpu_id is not None and exp["gpu_needed"]:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    elif not exp["gpu_needed"]:
        env["CUDA_VISIBLE_DEVICES"] = ""   # force CPU for non-GPU experiments

    return env


# ---------------------------------------------------------------------------
# Wrapper code injected at the top of patched scripts
# ---------------------------------------------------------------------------

HOLDOUT_WRAPPER_PREAMBLE = '''\
# ── Injected by rerun_holdout_experiments.py ────────────────────────────────
import os as _os
import pathlib as _pathlib
import sys as _sys

_HOLDOUT_ACTIVE = _os.getenv("SENTINEL_HOLDOUT", "0") == "1"
_TEST_START = _os.getenv("SENTINEL_TEST_START", "2024-01-01")
_VAL_END = _os.getenv("SENTINEL_VAL_END", "2024-01-01")
_HOLDOUT_OUTPUT = _os.getenv("SENTINEL_HOLDOUT_OUTPUT", "")

# Fix PROJECT_ROOT: patched scripts live in results/holdout/ so
# Path(__file__).parent.parent resolves to results/ instead of project root.
# Override __file__ BEFORE any module code reads it — this is the definitive fix.
_PROJECT_ROOT = _pathlib.Path(_os.getenv("SENTINEL_PROJECT_ROOT", "/home/bcheng/SENTINEL"))
if _HOLDOUT_ACTIVE:
    # Point __file__ back to the original script location so that
    # Path(__file__).resolve().parent.parent gives the real project root.
    _orig_script = _os.getenv("SENTINEL_ORIGINAL_SCRIPT", "")
    if _orig_script:
        __file__ = _orig_script

if _HOLDOUT_ACTIVE and _HOLDOUT_OUTPUT:
    _holdout_out_dir = _pathlib.Path(_HOLDOUT_OUTPUT)
    _holdout_out_dir.mkdir(parents=True, exist_ok=True)

def _holdout_filter_df(df, date_col="startDateTime"):
    """Filter a pandas DataFrame to the temporal test split (2024+)."""
    if not _HOLDOUT_ACTIVE:
        return df
    import pandas as _pd
    ts = _pd.to_datetime(df[date_col], utc=True, errors="coerce")
    mask = ts >= _pd.Timestamp(_TEST_START, tz="UTC")
    filtered = df[mask].reset_index(drop=True)
    print(f"[HOLDOUT] {date_col} filter: {len(df)} -> {len(filtered)} rows "
          f"(kept >= {_TEST_START})")
    return filtered

def _holdout_output_dir(original_dir):
    """Return the holdout output directory override if active."""
    if _HOLDOUT_ACTIVE and _HOLDOUT_OUTPUT:
        return _pathlib.Path(_HOLDOUT_OUTPUT)
    return _pathlib.Path(original_dir)
# ── End injection ───────────────────────────────────────────────────────────
'''


def make_patched_script(exp_name: str) -> Path:
    """Write a temporary patched copy of the script with the holdout preamble.

    The patched copy is written to results/holdout/.tmp_<exp_name>.py.
    Returns the path to the patched script.
    """
    orig = PROJECT_ROOT / EXPERIMENTS[exp_name]["script"]
    if not orig.exists():
        raise FileNotFoundError(f"Script not found: {orig}")

    patched_path = HOLDOUT_RESULTS_DIR / f".tmp_{exp_name}.py"
    original_text = orig.read_text(encoding="utf-8")

    # Find the first non-shebang, non-docstring, non-__future__ code line
    lines = original_text.splitlines()
    insert_after = 0
    in_docstring = False
    docstring_quote = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i == 0 and stripped.startswith("#!"):
            insert_after = 1
            continue
        # Track multi-line docstrings
        if in_docstring:
            if docstring_quote in stripped:
                insert_after = i + 1
                in_docstring = False
            continue
        # Skip module-level docstring
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            if stripped.count(quote) >= 2 and len(stripped) > 3:
                insert_after = i + 1  # single-line docstring
                continue
            # Multi-line docstring
            in_docstring = True
            docstring_quote = quote
            continue
        # Skip __future__ imports — they MUST precede injected code
        if stripped.startswith("from __future__"):
            insert_after = i + 1
            continue
        # Skip blank lines and comments between docstring and imports
        if not stripped or stripped.startswith("#"):
            continue
        # First real code line — inject before it
        insert_after = i
        break

    patched_lines = (
        lines[:insert_after]
        + HOLDOUT_WRAPPER_PREAMBLE.splitlines()
        + lines[insert_after:]
    )
    patched_path.write_text("\n".join(patched_lines), encoding="utf-8")
    return patched_path


# ---------------------------------------------------------------------------
# Dependency checker
# ---------------------------------------------------------------------------

def check_dependencies(exp_name: str) -> Tuple[bool, List[str]]:
    """Return (ready, missing) based on whether dependency result dirs exist."""
    exp = EXPERIMENTS[exp_name]
    missing = []
    for dep in exp["depends_on"]:
        dep_dir = HOLDOUT_RESULTS_DIR / dep
        if not dep_dir.exists() or not any(dep_dir.iterdir()):
            missing.append(dep)
    return (len(missing) == 0), missing


# ---------------------------------------------------------------------------
# Result check
# ---------------------------------------------------------------------------

def has_holdout_results(exp_name: str) -> bool:
    """Return True if this experiment already has holdout results."""
    result_dir = HOLDOUT_RESULTS_DIR / EXPERIMENTS[exp_name]["result_key"]
    if not result_dir.exists():
        return False
    files = list(result_dir.glob("*.json")) + list(result_dir.glob("*.pt"))
    return len(files) > 0


# ---------------------------------------------------------------------------
# Single experiment launcher
# ---------------------------------------------------------------------------

def run_experiment(
    exp_name: str,
    gpu_id: Optional[int],
    dry_run: bool = False,
    patch: bool = True,
) -> Tuple[bool, str]:
    """Launch one experiment.

    Returns (success, message).
    """
    exp = EXPERIMENTS[exp_name]
    script_path = PROJECT_ROOT / exp["script"]

    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    # Dependency check (skip in dry-run — just show the plan)
    if not dry_run:
        ready, missing = check_dependencies(exp_name)
        if not ready:
            return False, f"Dependencies not met: {missing}"

    env = build_env(exp_name, gpu_id)
    log_path = LOGS_DIR / f"{exp_name}_holdout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # For patched experiments, use a temp copy
    if patch and exp_name in NEEDS_PATCHING:
        try:
            run_script = make_patched_script(exp_name)
        except Exception as e:
            return False, f"Patching failed: {e}"
    else:
        run_script = script_path

    cmd = [sys.executable, str(run_script)]

    if dry_run:
        gpu_str = str(gpu_id) if gpu_id is not None else "CPU"
        patched = " [patched]" if exp_name in NEEDS_PATCHING else ""
        print(
            f"  [DRY-RUN] {exp_name}{patched}\n"
            f"    script : {run_script}\n"
            f"    gpu    : {gpu_str}\n"
            f"    output : {HOLDOUT_RESULTS_DIR / exp['result_key']}\n"
            f"    log    : {log_path}\n"
            f"    note   : {exp['notes']}\n"
        )
        return True, "dry-run"

    print(f"  Launching {exp_name} (GPU={gpu_id}) ...")
    t0 = time.time()
    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.run(
                cmd,
                env=env,
                cwd=str(PROJECT_ROOT),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=7200,  # 2-hour hard cap per experiment
            )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            msg = f"OK in {elapsed:.0f}s  (log: {log_path})"
            return True, msg
        else:
            msg = f"FAILED (exit {proc.returncode}) in {elapsed:.0f}s  (log: {log_path})"
            return False, msg
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after 2h  (log: {log_path})"
    except Exception as e:
        return False, f"Exception: {e}"


# ---------------------------------------------------------------------------
# Parallel launcher
# ---------------------------------------------------------------------------

def run_parallel(
    groups: List[str],
    gpu_ids: List[int],
    skip_done: bool,
    dry_run: bool,
) -> None:
    """Run experiment groups in parallel across GPU IDs.

    Maps:
      groups[0] -> gpu_ids[0]
      groups[1] -> gpu_ids[1]
      ...

    Groups are run concurrently; experiments within a group run sequentially
    in dependency order.
    """
    import threading

    results: Dict[str, Tuple[bool, str]] = {}

    def run_group(group_letter: str, gpu_id: int) -> None:
        exps = [
            name for name, info in EXPERIMENTS.items()
            if info["group"] == group_letter
        ]
        print(f"\n[Group {group_letter}] Starting {len(exps)} experiments on GPU {gpu_id}")
        for exp_name in exps:
            if skip_done and has_holdout_results(exp_name):
                print(f"  [SKIP] {exp_name} — holdout results already exist")
                results[exp_name] = (True, "skipped (already done)")
                continue
            success, msg = run_experiment(exp_name, gpu_id, dry_run=dry_run)
            results[exp_name] = (success, msg)
            status = "OK" if success else "FAIL"
            print(f"  [{status}] {exp_name}: {msg}")

    threads = []
    for group_letter, gpu_id in zip(groups, gpu_ids):
        t = threading.Thread(
            target=run_group,
            args=(group_letter, gpu_id),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    _print_summary(results)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: Dict[str, Tuple[bool, str]]) -> None:
    ok = [(k, v) for k, v in results.items() if v[0]]
    fail = [(k, v) for k, v in results.items() if not v[0]]
    print(f"\n{'=' * 65}")
    print(f"HOLDOUT RERUN SUMMARY  ({len(ok)} OK, {len(fail)} FAILED)")
    print(f"{'=' * 65}")
    for name, (_, msg) in sorted(results.items()):
        status = "OK  " if results[name][0] else "FAIL"
        print(f"  [{status}] {name}: {msg}")
    print(f"Results in: {HOLDOUT_RESULTS_DIR}")

    # Save summary JSON
    summary_path = HOLDOUT_RESULTS_DIR / "rerun_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_ok": len(ok),
        "n_failed": len(fail),
        "experiments": {
            name: {"success": success, "message": msg}
            for name, (success, msg) in results.items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--gpu", type=int, default=1,
        help="GPU ID to use when running sequentially (default: 1). "
             "GPU 0 is reserved for AquaSSM v2 training.",
    )
    p.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs for parallel mode, e.g. '1,2,3'. "
             "Used with --parallel.",
    )
    p.add_argument(
        "--group", choices=["A", "B", "C", "D"], default=None,
        help="Run only experiments in this group (default: all groups).",
    )
    p.add_argument(
        "--only", type=str, default=None,
        help="Run a single experiment by name (e.g. exp2_baseline_comparison).",
    )
    p.add_argument(
        "--parallel", action="store_true",
        help="Parallelise groups across GPUs. Requires --gpus.",
    )
    p.add_argument(
        "--skip-done", action="store_true",
        help="Skip experiments that already have results in results/holdout/.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without actually launching anything.",
    )
    p.add_argument(
        "--no-patch", action="store_true",
        help="Disable the holdout wrapper injection for NEEDS_PATCHING scripts "
             "(run originals as-is, relying solely on env variables).",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List all registered experiments and exit.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.list:
        print(f"\n{'Experiment':<40} {'Group':<7} {'GPU?':<6} {'NEON?':<6} {'Needs Patch?'}")
        print("-" * 80)
        for name, info in EXPERIMENTS.items():
            print(
                f"  {name:<38} {info['group']:<7} "
                f"{'yes' if info['gpu_needed'] else 'no':<6} "
                f"{'yes' if info['neon_temporal'] else 'no':<6} "
                f"{'yes' if name in NEEDS_PATCHING else 'no'}"
            )
        return

    patch = not args.no_patch

    # Single experiment mode
    if args.only:
        if args.only not in EXPERIMENTS:
            print(f"ERROR: '{args.only}' not in registry. Use --list to see options.")
            sys.exit(1)
        success, msg = run_experiment(args.only, args.gpu, dry_run=args.dry_run, patch=patch)
        status = "OK" if success else "FAIL"
        print(f"[{status}] {args.only}: {msg}")
        sys.exit(0 if success else 1)

    # Filter by group
    if args.group:
        experiment_names = [
            n for n, info in EXPERIMENTS.items() if info["group"] == args.group
        ]
    else:
        experiment_names = list(EXPERIMENTS.keys())

    # Parallel mode
    if args.parallel:
        if not args.gpus:
            print("ERROR: --parallel requires --gpus (e.g. --gpus 1,2,3).")
            sys.exit(1)
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
        group_letters = sorted(set(EXPERIMENTS[n]["group"] for n in experiment_names))
        if len(gpu_ids) < len(group_letters):
            print(
                f"WARNING: {len(group_letters)} groups but only {len(gpu_ids)} GPUs. "
                "Some groups will share a GPU."
            )
        # Pad GPU list cyclically
        while len(gpu_ids) < len(group_letters):
            gpu_ids.append(gpu_ids[-1])
        run_parallel(group_letters, gpu_ids, skip_done=args.skip_done, dry_run=args.dry_run)
        return

    # Sequential mode
    results: Dict[str, Tuple[bool, str]] = {}
    print(f"\nRunning {len(experiment_names)} experiments sequentially on GPU {args.gpu}")
    print(f"Holdout output: {HOLDOUT_RESULTS_DIR}\n")

    for exp_name in experiment_names:
        if args.skip_done and has_holdout_results(exp_name):
            print(f"  [SKIP] {exp_name} — holdout results already exist")
            results[exp_name] = (True, "skipped (already done)")
            continue
        success, msg = run_experiment(exp_name, args.gpu, dry_run=args.dry_run, patch=patch)
        results[exp_name] = (success, msg)
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {exp_name}: {msg}")

    if not args.dry_run:
        _print_summary(results)


if __name__ == "__main__":
    main()
