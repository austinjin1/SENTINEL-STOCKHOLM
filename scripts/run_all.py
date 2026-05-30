#!/usr/bin/env python3
"""SENTINEL v2 training orchestrator — temporal-spatial holdout protocol.

Orchestrates all v2 training, evaluation, and benchmarking scripts with
dependency-aware phased execution, GPU allocation, and result aggregation.

Phases:
  A — Train all 5 encoders in parallel (each on assigned GPU)
  B — Train fusion model (requires encoder checkpoints from A)
  C — Run honest baselines (can run parallel with B)
  D — Fusion ablation / prospective validation (after B)

GPU allocation:
  GPU 0: AquaSSM  ->  Fusion (Phase A -> B)
  GPU 1: HydroViT                (Phase A)
  GPU 2: MicroBiomeNet -> ToxiGene  (Phase A, sequential)
  GPU 3: BioMotion -> Honest Baselines  (Phase A -> C)

Usage:
  python scripts/run_all_v2.py                    # Run all phases
  python scripts/run_all_v2.py --dry-run           # Check data readiness only
  python scripts/run_all_v2.py --phase A           # Run Phase A only
  python scripts/run_all_v2.py --skip-if-done      # Skip scripts with results
  python scripts/run_all_v2.py --conda-env myenv   # Use specific conda env

MIT License -- Bryan Cheng, SENTINEL project, 2026
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Script registry: name -> (script_path, gpu, data_dirs, result_file)
#
# data_dirs: list of directories (any one must exist with files)
# result_file: expected result JSON filename in results/benchmarks/
# ---------------------------------------------------------------------------

SCRIPT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "aquassm_v2": {
        "script": "scripts/train_aquassm_v2.py",
        "gpu": "0",
        "data_dirs": [
            "data/processed/sensor/full",
        ],
        "result_file": "aquassm_v2_real_benchmark.json",
        "phase": "A",
        "description": "AquaSSM sensor encoder (temporal-spatial holdout)",
    },
    "hydrovit_v2": {
        "script": "scripts/train_hydrovit_v2.py",
        "gpu": "1",
        "data_dirs": [
            "data/processed/satellite/real",
            "data/processed/satellite",  # fallback: paired_wq_v5.npz
        ],
        "result_file": "hydrovit_v2_holdout.json",
        "phase": "A",
        "description": "HydroViT satellite encoder (temporal-spatial holdout)",
    },
    "microbiomenet_v3": {
        "script": "scripts/train_microbiomenet_v3.py",
        "gpu": "2",
        "data_dirs": [
            "data/processed/microbial/emp_16s",
        ],
        "result_file": "microbiomenet_v3_holdout.json",
        "phase": "A",
        "description": "MicroBiomeNet microbial encoder (spatial holdout)",
    },
    "toxigene_v3": {
        "script": "scripts/train_toxigene_v3.py",
        "gpu": "2",
        "data_dirs": [
            "data/processed/molecular",
        ],
        "result_file": "toxigene_v3_holdout.json",
        "phase": "A",
        "description": "ToxiGene molecular encoder (spatial holdout)",
    },
    "biomotion_v2": {
        "script": "scripts/train_biomotion_v2.py",
        "gpu": "3",
        "data_dirs": [
            "data/processed/behavioral_fullreal",
            "data/processed/behavioral_real",
        ],
        "result_file": "biomotion_v2_holdout.json",
        "phase": "A",
        "description": "BioMotion behavioral encoder (spatial holdout)",
    },
    "fusion_v2": {
        "script": "scripts/train_fusion_v2.py",
        "gpu": "0",
        "data_dirs": [
            "data/raw/sensor/full",
            "data/processed/sensor/full",
        ],
        "result_file": "fusion_v2_holdout.json",
        "phase": "B",
        "description": "Perceiver IO fusion (temporal-spatial holdout)",
    },
    "honest_baselines": {
        "script": "scripts/exp_honest_baselines.py",
        "gpu": "3",
        "data_dirs": [
            "data/processed/sensor/full",
        ],
        "result_file": "honest_baselines.json",
        "phase": "C",
        "description": "Honest baselines and negative controls",
    },
    "prospective_validation": {
        "script": "scripts/prospective_validation.py",
        "gpu": "0",
        "data_dirs": [],  # No data dir needed — uses checkpoints + live fetch
        "result_file": None,  # Saves to results/prospective/
        "phase": "D",
        "description": "Forward prediction pre-registration",
    },
}

# v1 result files for comparison (random split baselines)
V1_RESULT_FILES = {
    "aquassm": "aquassm_benchmark.json",
    "hydrovit": "hydrovit_v9_results.json",
    "microbiomenet": "microbiomenet_benchmark.json",
    "toxigene": "toxigene_fullreal_benchmark.json",
    "biomotion": "biomotion_benchmark.json",
}

# GPU assignment plan: gpu_id -> list of script names (in execution order)
GPU_PLAN = {
    "0": ["aquassm_v2"],      # Phase A, then fusion_v2 in Phase B
    "1": ["hydrovit_v2"],     # Phase A only
    "2": ["microbiomenet_v3", "toxigene_v3"],  # Phase A, sequential
    "3": ["biomotion_v2"],    # Phase A, then honest_baselines in Phase C
}


# ---------------------------------------------------------------------------
# Data readiness check
# ---------------------------------------------------------------------------

def check_data_ready(data_dirs: List[str]) -> Tuple[bool, str]:
    """Check if at least one data directory exists and has files.

    Searches both the directory itself and one level of subdirectories
    (e.g., data/processed/molecular/real/*.npz).

    Returns (is_ready, message).
    """
    if not data_dirs:
        return True, "No data directory required"

    for d in data_dirs:
        dir_path = PROJECT_ROOT / d
        if not dir_path.exists():
            continue
        # Check for data files in dir and immediate subdirectories
        file_count = 0
        for ext in ("*.npz", "*.npy", "*.parquet", "*.csv", "*.pt"):
            file_count += len(list(dir_path.glob(ext)))
            file_count += len(list(dir_path.glob(f"*/{ext}")))
        if file_count > 0:
            return True, f"{dir_path} ({file_count} files)"
    # None found
    tried = ", ".join(str(PROJECT_ROOT / d) for d in data_dirs)
    return False, f"No data found in: {tried}"


def data_readiness_summary() -> Dict[str, Tuple[bool, str]]:
    """Check all scripts' data readiness."""
    summary = {}
    for name, info in SCRIPT_REGISTRY.items():
        ready, msg = check_data_ready(info["data_dirs"])
        summary[name] = (ready, msg)
    return summary


def print_readiness_table(summary: Dict[str, Tuple[bool, str]]) -> None:
    """Print a formatted data readiness table."""
    print("\n" + "=" * 78)
    print("DATA READINESS SUMMARY")
    print("=" * 78)

    max_name = max(len(n) for n in summary)
    for name, (ready, msg) in summary.items():
        status = "READY" if ready else "MISSING"
        icon = "[+]" if ready else "[-]"
        desc = SCRIPT_REGISTRY[name]["description"]
        print(f"  {icon} {name:<{max_name}}  {status:<8}  {msg}")

    n_ready = sum(1 for r, _ in summary.values() if r)
    n_total = len(summary)
    print(f"\n  {n_ready}/{n_total} scripts have data ready")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Result checking
# ---------------------------------------------------------------------------

def result_exists(name: str) -> bool:
    """Check if a result file already exists for the given script."""
    info = SCRIPT_REGISTRY[name]
    rf = info.get("result_file")
    if rf is None:
        return False
    return (RESULTS_DIR / rf).exists()


# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------

class ScriptRunner:
    """Runs a training script as a subprocess with logging."""

    def __init__(
        self,
        name: str,
        conda_env: Optional[str] = None,
    ):
        self.name = name
        self.info = SCRIPT_REGISTRY[name]
        self.conda_env = conda_env
        self.process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.exit_code: Optional[int] = None
        self.log_path = LOGS_DIR / f"v2_{name}.log"

    def build_command(self) -> List[str]:
        """Build the command to run the script."""
        script_path = str(PROJECT_ROOT / self.info["script"])
        if self.conda_env:
            # Use conda run to execute in the right environment
            return [
                "conda", "run", "-n", self.conda_env, "--no-capture-output",
                "python", "-u", script_path,
            ]
        else:
            return [sys.executable, "-u", script_path]

    def start(self) -> None:
        """Launch the script as a subprocess."""
        cmd = self.build_command()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.info["gpu"]

        # Open log file
        self.log_file = open(self.log_path, "w")
        self.log_file.write(f"# SENTINEL v2 Orchestrator\n")
        self.log_file.write(f"# Script: {self.name}\n")
        self.log_file.write(f"# Command: {' '.join(cmd)}\n")
        self.log_file.write(f"# GPU: {self.info['gpu']}\n")
        self.log_file.write(f"# Started: {datetime.now().isoformat()}\n")
        self.log_file.write(f"# {'=' * 60}\n\n")
        self.log_file.flush()

        self.start_time = time.time()
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        print(f"  [STARTED] {self.name} (PID {self.process.pid}, GPU {self.info['gpu']}) "
              f"-> {self.log_path}")

    def poll(self) -> Optional[int]:
        """Check if the process has finished. Returns exit code or None."""
        if self.process is None:
            return self.exit_code
        rc = self.process.poll()
        if rc is not None:
            self.end_time = time.time()
            self.exit_code = rc
            self.log_file.write(f"\n# {'=' * 60}\n")
            self.log_file.write(f"# Finished: {datetime.now().isoformat()}\n")
            self.log_file.write(f"# Exit code: {rc}\n")
            elapsed = self.end_time - self.start_time
            self.log_file.write(f"# Elapsed: {format_duration(elapsed)}\n")
            self.log_file.close()
            return rc
        return None

    def wait(self) -> int:
        """Wait for the process to finish."""
        if self.process is None:
            return self.exit_code if self.exit_code is not None else -1
        self.process.wait()
        return self.poll()

    def kill(self) -> None:
        """Kill the subprocess."""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            except Exception:
                pass
            self.end_time = time.time()
            self.exit_code = -9
            try:
                self.log_file.write(f"\n# KILLED by orchestrator\n")
                self.log_file.close()
            except Exception:
                pass

    @property
    def elapsed(self) -> Optional[float]:
        if self.start_time is None:
            return None
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def is_running(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds + td.days * 86400, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------

def run_parallel(
    names: List[str],
    conda_env: Optional[str],
    skip_if_done: bool,
    readiness: Dict[str, Tuple[bool, str]],
) -> Dict[str, ScriptRunner]:
    """Run a list of scripts in parallel, respecting data readiness.

    Returns dict of name -> ScriptRunner (completed).
    """
    runners: Dict[str, ScriptRunner] = {}
    skipped: List[str] = []

    for name in names:
        # Check data readiness
        ready, msg = readiness.get(name, (True, ""))
        if not ready:
            print(f"  [SKIP] {name}: data not ready ({msg})")
            skipped.append(name)
            continue

        # Check if already done
        if skip_if_done and result_exists(name):
            rf = SCRIPT_REGISTRY[name].get("result_file", "")
            print(f"  [SKIP] {name}: results already exist ({rf})")
            skipped.append(name)
            continue

        runner = ScriptRunner(name, conda_env)
        runner.start()
        runners[name] = runner

    if not runners:
        print("  No scripts to run in this batch.")
        return runners

    # Wait for all to complete, printing status every 30 seconds
    try:
        while any(r.is_running for r in runners.values()):
            time.sleep(10)
            # Print running status every ~30s (every 3rd check)
            still_running = [n for n, r in runners.items() if r.is_running]
            finished_now = [
                n for n, r in runners.items()
                if not r.is_running and r.exit_code is not None
            ]
            for name in finished_now:
                r = runners[name]
                elapsed = format_duration(r.elapsed or 0)
                if r.exit_code == 0:
                    print(f"  [DONE]  {name} (exit 0, {elapsed})")
                else:
                    print(f"  [FAIL]  {name} (exit {r.exit_code}, {elapsed})")
            if still_running:
                elapsed_strs = []
                for n in still_running:
                    e = runners[n].elapsed
                    elapsed_strs.append(f"{n}={format_duration(e or 0)}")
                # Only print if meaningful time has passed
    except KeyboardInterrupt:
        raise

    # Final poll for any that finished on the last check
    for name, runner in runners.items():
        runner.poll()

    return runners


def run_sequential_on_gpu(
    names: List[str],
    conda_env: Optional[str],
    skip_if_done: bool,
    readiness: Dict[str, Tuple[bool, str]],
) -> Dict[str, ScriptRunner]:
    """Run scripts sequentially on the same GPU.

    Returns dict of name -> ScriptRunner (completed).
    """
    runners = {}
    for name in names:
        ready, msg = readiness.get(name, (True, ""))
        if not ready:
            print(f"  [SKIP] {name}: data not ready ({msg})")
            continue
        if skip_if_done and result_exists(name):
            rf = SCRIPT_REGISTRY[name].get("result_file", "")
            print(f"  [SKIP] {name}: results already exist ({rf})")
            continue

        runner = ScriptRunner(name, conda_env)
        runner.start()
        runners[name] = runner

        # Wait for it to finish before starting the next
        rc = runner.wait()
        elapsed = format_duration(runner.elapsed or 0)
        if rc == 0:
            print(f"  [DONE]  {name} (exit 0, {elapsed})")
        else:
            print(f"  [FAIL]  {name} (exit {rc}, {elapsed})")

    return runners


def run_phase_a(
    conda_env: Optional[str],
    skip_if_done: bool,
    readiness: Dict[str, Tuple[bool, str]],
) -> Dict[str, ScriptRunner]:
    """Phase A: Train all 5 encoders.

    GPU 0: AquaSSM
    GPU 1: HydroViT
    GPU 2: MicroBiomeNet -> ToxiGene (sequential, same GPU)
    GPU 3: BioMotion

    All GPU pipelines run in parallel, but GPU 2 runs its two scripts
    sequentially since they share the same card.
    """
    print("\n" + "=" * 78)
    print("PHASE A: Train all encoders (temporal-spatial holdout)")
    print("=" * 78)

    all_runners: Dict[str, ScriptRunner] = {}

    # GPU 0, 1, 3: single scripts that run in parallel
    parallel_names = ["aquassm_v2", "hydrovit_v2", "biomotion_v2"]

    # GPU 2: sequential (microbiomenet_v3 then toxigene_v3)
    gpu2_names = ["microbiomenet_v3", "toxigene_v3"]

    # Filter out ready/skip for parallel scripts
    parallel_to_run = []
    for name in parallel_names:
        ready, msg = readiness.get(name, (True, ""))
        if not ready:
            print(f"  [SKIP] {name}: data not ready ({msg})")
            continue
        if skip_if_done and result_exists(name):
            rf = SCRIPT_REGISTRY[name].get("result_file", "")
            print(f"  [SKIP] {name}: results already exist ({rf})")
            continue
        parallel_to_run.append(name)

    # Start parallel scripts
    parallel_runners: Dict[str, ScriptRunner] = {}
    for name in parallel_to_run:
        runner = ScriptRunner(name, conda_env)
        runner.start()
        parallel_runners[name] = runner

    # Start GPU 2 sequential chain in its own thread-like loop
    # We'll interleave polling of parallel scripts with GPU 2 sequential work
    gpu2_runners: Dict[str, ScriptRunner] = {}
    gpu2_queue = list(gpu2_names)  # copy

    def start_next_gpu2() -> Optional[ScriptRunner]:
        """Start the next GPU 2 script if available."""
        while gpu2_queue:
            name = gpu2_queue.pop(0)
            ready, msg = readiness.get(name, (True, ""))
            if not ready:
                print(f"  [SKIP] {name}: data not ready ({msg})")
                continue
            if skip_if_done and result_exists(name):
                rf = SCRIPT_REGISTRY[name].get("result_file", "")
                print(f"  [SKIP] {name}: results already exist ({rf})")
                continue
            runner = ScriptRunner(name, conda_env)
            runner.start()
            gpu2_runners[name] = runner
            return runner
        return None

    current_gpu2 = start_next_gpu2()

    # Poll all scripts until everything is done
    try:
        while True:
            any_running = False

            # Check parallel scripts
            for name, runner in parallel_runners.items():
                if runner.is_running:
                    any_running = True
                    rc = runner.poll()
                    if rc is not None:
                        elapsed = format_duration(runner.elapsed or 0)
                        if rc == 0:
                            print(f"  [DONE]  {name} (exit 0, {elapsed})")
                        else:
                            print(f"  [FAIL]  {name} (exit {rc}, {elapsed})")

            # Check GPU 2 sequential chain
            if current_gpu2 is not None and current_gpu2.is_running:
                any_running = True
                rc = current_gpu2.poll()
                if rc is not None:
                    name = [n for n, r in gpu2_runners.items() if r is current_gpu2][0]
                    elapsed = format_duration(current_gpu2.elapsed or 0)
                    if rc == 0:
                        print(f"  [DONE]  {name} (exit 0, {elapsed})")
                    else:
                        print(f"  [FAIL]  {name} (exit {rc}, {elapsed})")
                    # Start next GPU 2 script
                    current_gpu2 = start_next_gpu2()
                    if current_gpu2 is not None:
                        any_running = True
            elif current_gpu2 is None and gpu2_queue:
                # Try to start next (in case previous was skipped)
                current_gpu2 = start_next_gpu2()
                if current_gpu2 is not None:
                    any_running = True

            if not any_running:
                break

            time.sleep(10)
    except KeyboardInterrupt:
        raise

    # Final poll
    for runner in list(parallel_runners.values()) + list(gpu2_runners.values()):
        runner.poll()

    all_runners.update(parallel_runners)
    all_runners.update(gpu2_runners)
    return all_runners


def run_phase_b(
    conda_env: Optional[str],
    skip_if_done: bool,
    readiness: Dict[str, Tuple[bool, str]],
) -> Dict[str, ScriptRunner]:
    """Phase B: Train fusion model (requires encoder checkpoints from A)."""
    print("\n" + "=" * 78)
    print("PHASE B: Train Perceiver IO fusion model")
    print("=" * 78)
    return run_parallel(
        ["fusion_v2"], conda_env, skip_if_done, readiness,
    )


def run_phase_c(
    conda_env: Optional[str],
    skip_if_done: bool,
    readiness: Dict[str, Tuple[bool, str]],
) -> Dict[str, ScriptRunner]:
    """Phase C: Run honest baselines (can run parallel with B)."""
    print("\n" + "=" * 78)
    print("PHASE C: Honest baselines comparison")
    print("=" * 78)
    return run_parallel(
        ["honest_baselines"], conda_env, skip_if_done, readiness,
    )


def run_phase_d(
    conda_env: Optional[str],
    skip_if_done: bool,
    readiness: Dict[str, Tuple[bool, str]],
) -> Dict[str, ScriptRunner]:
    """Phase D: Prospective validation (after fusion is trained)."""
    print("\n" + "=" * 78)
    print("PHASE D: Prospective validation registration")
    print("=" * 78)
    return run_parallel(
        ["prospective_validation"], conda_env, skip_if_done, readiness,
    )


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def load_json_safe(path: Path) -> Optional[Dict]:
    """Load a JSON file, returning None on failure."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None


def extract_metrics(data: Dict) -> Dict[str, Any]:
    """Extract key metrics from a benchmark JSON file.

    Handles various result formats across our scripts.
    """
    metrics: Dict[str, Any] = {}

    # Direct top-level metrics
    for key in ("auroc", "f1", "accuracy", "auprc", "lead_time_hours",
                "precision", "recall", "r2", "mae", "rmse"):
        if key in data:
            metrics[key] = data[key]

    # Test metrics nested under "test_metrics" or "test"
    for test_key in ("test_metrics", "test", "evaluation"):
        if test_key in data and isinstance(data[test_key], dict):
            for key in ("auroc", "f1", "accuracy", "auprc", "lead_time_hours",
                        "precision", "recall", "r2", "mae", "rmse"):
                if key in data[test_key] and key not in metrics:
                    metrics[key] = data[test_key][key]

    # Models dict (benchmark format: {"models": {"ModelName": {"auroc": ...}}})
    if "models" in data and isinstance(data["models"], dict):
        # Take the primary model (first one, or the one matching the script name)
        for model_name, model_data in data["models"].items():
            if isinstance(model_data, dict):
                for key in ("auroc", "f1", "accuracy", "auprc"):
                    if key in model_data and key not in metrics:
                        metrics[key] = model_data[key]
                # Only take the first model's metrics
                break

    # Phase 2 results (train_*_v2 format)
    if "phase2" in data and isinstance(data["phase2"], dict):
        p2 = data["phase2"]
        for key in ("auroc", "f1", "accuracy", "auprc"):
            if key in p2 and key not in metrics:
                metrics[key] = p2[key]
        if "test" in p2 and isinstance(p2["test"], dict):
            for key in ("auroc", "f1", "accuracy", "auprc"):
                if key in p2["test"] and key not in metrics:
                    metrics[key] = p2["test"][key]

    return metrics


def aggregate_results() -> Dict[str, Any]:
    """Load all benchmark JSONs and build a consolidated comparison."""
    consolidated = {
        "timestamp": datetime.now().isoformat(),
        "protocol": "temporal-spatial holdout (v2)",
        "v2_results": {},
        "v1_results": {},
        "comparison": {},
    }

    # Load v2 results
    for name, info in SCRIPT_REGISTRY.items():
        rf = info.get("result_file")
        if rf is None:
            continue
        data = load_json_safe(RESULTS_DIR / rf)
        if data is not None:
            metrics = extract_metrics(data)
            consolidated["v2_results"][name] = {
                "file": rf,
                "metrics": metrics,
                "description": info["description"],
            }

    # Load v1 results for comparison
    for short_name, rf in V1_RESULT_FILES.items():
        data = load_json_safe(RESULTS_DIR / rf)
        if data is not None:
            metrics = extract_metrics(data)
            consolidated["v1_results"][short_name] = {
                "file": rf,
                "metrics": metrics,
                "split": "random 70/15/15",
            }

    # Build comparison table: v1 vs v2 where both exist
    v2_to_v1 = {
        "aquassm_v2": "aquassm",
        "hydrovit_v2": "hydrovit",
        "microbiomenet_v3": "microbiomenet",
        "toxigene_v3": "toxigene",
        "biomotion_v2": "biomotion",
    }

    for v2_name, v1_name in v2_to_v1.items():
        v2_data = consolidated["v2_results"].get(v2_name, {}).get("metrics", {})
        v1_data = consolidated["v1_results"].get(v1_name, {}).get("metrics", {})

        if v2_data or v1_data:
            comparison_entry = {}
            all_keys = set(list(v2_data.keys()) + list(v1_data.keys()))
            for key in sorted(all_keys):
                v1_val = v1_data.get(key)
                v2_val = v2_data.get(key)
                entry = {"v1_random": v1_val, "v2_holdout": v2_val}
                if (
                    isinstance(v1_val, (int, float))
                    and isinstance(v2_val, (int, float))
                    and v1_val != 0
                ):
                    entry["delta"] = round(v2_val - v1_val, 4)
                    entry["pct_change"] = round(
                        (v2_val - v1_val) / abs(v1_val) * 100, 2
                    )
                comparison_entry[key] = entry
            consolidated["comparison"][v2_name] = comparison_entry

    return consolidated


def print_comparison_table(consolidated: Dict[str, Any]) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 90)
    print("RESULTS COMPARISON: v1 (random split) vs v2 (temporal-spatial holdout)")
    print("=" * 90)

    v2_results = consolidated.get("v2_results", {})
    v1_results = consolidated.get("v1_results", {})
    comparison = consolidated.get("comparison", {})

    if not v2_results and not v1_results:
        print("  No results found.")
        return

    # Print v2 results table
    print("\n--- v2 Holdout Results ---")
    print(f"  {'Model':<25} {'AUROC':>8} {'F1':>8} {'Accuracy':>10} {'AUPRC':>8}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 8}")

    for name, data in sorted(v2_results.items()):
        m = data.get("metrics", {})
        auroc = m.get("auroc", "")
        f1 = m.get("f1", "")
        acc = m.get("accuracy", "")
        auprc = m.get("auprc", "")

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else str(v) or "-"

        print(f"  {name:<25} {fmt(auroc):>8} {fmt(f1):>8} {fmt(acc):>10} {fmt(auprc):>8}")

    # Print comparison
    if comparison:
        print(f"\n--- v1 vs v2 Comparison ---")
        print(f"  {'Model':<25} {'Metric':<10} {'v1 Random':>10} {'v2 Holdout':>10} "
              f"{'Delta':>8} {'% Change':>9}")
        print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 9}")

        for model, metrics in sorted(comparison.items()):
            for metric, vals in sorted(metrics.items()):
                v1 = vals.get("v1_random")
                v2 = vals.get("v2_holdout")
                delta = vals.get("delta")
                pct = vals.get("pct_change")

                def fmt(v):
                    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"

                pct_str = f"{pct:+.1f}%" if isinstance(pct, (int, float)) else "-"
                delta_str = f"{delta:+.4f}" if isinstance(delta, (int, float)) else "-"

                print(f"  {model:<25} {metric:<10} {fmt(v1):>10} {fmt(v2):>10} "
                      f"{delta_str:>8} {pct_str:>9}")
                model = ""  # Only print model name once

    print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_execution_summary(all_runners: Dict[str, ScriptRunner]) -> None:
    """Print a summary of all script executions."""
    print("\n" + "=" * 78)
    print("EXECUTION SUMMARY")
    print("=" * 78)

    if not all_runners:
        print("  No scripts were executed.")
        return

    print(f"\n  {'Script':<25} {'Status':<8} {'Exit':>5} {'Duration':>10} {'GPU':>4}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 5} {'-' * 10} {'-' * 4}")

    n_ok = 0
    n_fail = 0

    for name, runner in sorted(all_runners.items()):
        if runner.exit_code == 0:
            status = "OK"
            n_ok += 1
        elif runner.exit_code == -9:
            status = "KILLED"
            n_fail += 1
        elif runner.exit_code is not None:
            status = "FAIL"
            n_fail += 1
        else:
            status = "???"
            n_fail += 1

        elapsed = format_duration(runner.elapsed or 0) if runner.elapsed else "-"
        ec = str(runner.exit_code) if runner.exit_code is not None else "-"
        gpu = runner.info["gpu"]
        print(f"  {name:<25} {status:<8} {ec:>5} {elapsed:>10} {gpu:>4}")

    total_time = 0
    for r in all_runners.values():
        if r.elapsed:
            total_time = max(total_time, r.elapsed)

    print(f"\n  Total: {n_ok} succeeded, {n_fail} failed")
    print(f"  Wall clock: {format_duration(total_time)}")
    print(f"  Logs: {LOGS_DIR}/v2_*.log")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SENTINEL v2 training orchestrator (temporal-spatial holdout)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  A  Train all 5 encoders in parallel
  B  Train Perceiver IO fusion (needs encoder checkpoints)
  C  Honest baselines (parallel with B)
  D  Prospective validation (after B)

Examples:
  %(prog)s                        Run all phases
  %(prog)s --dry-run              Check data readiness only
  %(prog)s --phase A              Run Phase A only
  %(prog)s --phase B --phase C    Run Phases B and C
  %(prog)s --skip-if-done         Skip scripts with existing results
""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check data readiness and print plan without running anything",
    )
    parser.add_argument(
        "--phase",
        action="append",
        choices=["A", "B", "C", "D", "all"],
        default=None,
        help="Run specific phase(s) only (can be repeated). Default: all",
    )
    parser.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Skip scripts whose result JSON already exists",
    )
    parser.add_argument(
        "--conda-env",
        default=None,
        help="Conda environment name (default: use current Python)",
    )
    args = parser.parse_args()

    # Determine which phases to run
    phases = set()
    if args.phase is None or "all" in args.phase:
        phases = {"A", "B", "C", "D"}
    else:
        phases = set(args.phase)

    # Banner
    print("=" * 78)
    print("  SENTINEL v2 Training Orchestrator")
    print("  Temporal-Spatial Holdout Protocol")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Phases: {', '.join(sorted(phases))}")
    if args.dry_run:
        print("  Mode: DRY RUN (no scripts will be launched)")
    if args.skip_if_done:
        print("  Skip if done: YES")
    if args.conda_env:
        print(f"  Conda env: {args.conda_env}")
    print("=" * 78)

    # Data readiness check
    readiness = data_readiness_summary()
    print_readiness_table(readiness)

    if args.dry_run:
        # Print the execution plan
        print("EXECUTION PLAN (dry run):")
        for phase_label in sorted(phases):
            phase_scripts = [
                n for n, info in SCRIPT_REGISTRY.items()
                if info["phase"] == phase_label
            ]
            print(f"\n  Phase {phase_label}:")
            for name in phase_scripts:
                info = SCRIPT_REGISTRY[name]
                ready, msg = readiness.get(name, (True, ""))
                done = result_exists(name) if args.skip_if_done else False
                status = "READY" if ready else "NO DATA"
                if done:
                    status = "SKIP (done)"
                print(f"    {name:<25} GPU {info['gpu']}  {status}")
        print()
        return

    # Track all runners for summary
    all_runners: Dict[str, ScriptRunner] = {}
    orchestrator_start = time.time()

    # Install signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n\nInterrupted! Killing all running processes...")
        for name, runner in all_runners.items():
            if runner.is_running:
                print(f"  Killing {name} (PID {runner.process.pid})...")
                runner.kill()
        print_execution_summary(all_runners)
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Phase A: Encoders
        if "A" in phases:
            runners_a = run_phase_a(
                args.conda_env, args.skip_if_done, readiness,
            )
            all_runners.update(runners_a)

            # Check for failures
            failures = [n for n, r in runners_a.items() if r.exit_code != 0]
            if failures:
                print(f"\n  WARNING: Phase A failures: {', '.join(failures)}")
                print("  Continuing with remaining phases...")

        # Phase B and C can run in parallel (different GPUs)
        if "B" in phases and "C" in phases:
            print("\n" + "=" * 78)
            print("PHASES B + C: Fusion training and honest baselines (parallel)")
            print("=" * 78)

            # Start both sets of scripts
            bc_runners: Dict[str, ScriptRunner] = {}

            # Phase B scripts
            for name in ["fusion_v2"]:
                ready, msg = readiness.get(name, (True, ""))
                if not ready:
                    print(f"  [SKIP] {name}: data not ready ({msg})")
                    continue
                if args.skip_if_done and result_exists(name):
                    rf = SCRIPT_REGISTRY[name].get("result_file", "")
                    print(f"  [SKIP] {name}: results already exist ({rf})")
                    continue
                runner = ScriptRunner(name, args.conda_env)
                runner.start()
                bc_runners[name] = runner

            # Phase C scripts
            for name in ["honest_baselines"]:
                ready, msg = readiness.get(name, (True, ""))
                if not ready:
                    print(f"  [SKIP] {name}: data not ready ({msg})")
                    continue
                if args.skip_if_done and result_exists(name):
                    rf = SCRIPT_REGISTRY[name].get("result_file", "")
                    print(f"  [SKIP] {name}: results already exist ({rf})")
                    continue
                runner = ScriptRunner(name, args.conda_env)
                runner.start()
                bc_runners[name] = runner

            # Wait for all
            while any(r.is_running for r in bc_runners.values()):
                time.sleep(10)
                for name, runner in bc_runners.items():
                    if not runner.is_running and runner.exit_code is not None:
                        # Already finished — printed in a previous iteration?
                        pass
                for name, runner in list(bc_runners.items()):
                    rc = runner.poll()
                    if rc is not None and name not in all_runners:
                        elapsed = format_duration(runner.elapsed or 0)
                        if rc == 0:
                            print(f"  [DONE]  {name} (exit 0, {elapsed})")
                        else:
                            print(f"  [FAIL]  {name} (exit {rc}, {elapsed})")

            for runner in bc_runners.values():
                runner.poll()
            all_runners.update(bc_runners)

        elif "B" in phases:
            runners_b = run_phase_b(
                args.conda_env, args.skip_if_done, readiness,
            )
            all_runners.update(runners_b)
        elif "C" in phases:
            runners_c = run_phase_c(
                args.conda_env, args.skip_if_done, readiness,
            )
            all_runners.update(runners_c)

        # Phase D: after fusion
        if "D" in phases:
            runners_d = run_phase_d(
                args.conda_env, args.skip_if_done, readiness,
            )
            all_runners.update(runners_d)

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

    # Execution summary
    print_execution_summary(all_runners)

    # Result aggregation
    print("\n" + "=" * 78)
    print("RESULT AGGREGATION")
    print("=" * 78)

    consolidated = aggregate_results()

    # Print comparison table
    print_comparison_table(consolidated)

    # Save consolidated results
    out_path = RESULTS_DIR / "v2_consolidated.json"
    with open(out_path, "w") as f:
        json.dump(consolidated, f, indent=2, default=str)
    print(f"Consolidated results saved to: {out_path}")

    # Total wall clock
    total_elapsed = time.time() - orchestrator_start
    print(f"\nTotal orchestrator wall clock: {format_duration(total_elapsed)}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
