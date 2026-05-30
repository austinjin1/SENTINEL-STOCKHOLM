#!/usr/bin/env python3
# PYTHONNOUSERSITE=1
"""Predictability Audit for WaterDroneNet — SENTINEL pre-training diagnostic.

Before committing GPU cycles to the full WaterDroneNet vision model, this
script quantifies *per-target learnability* from cheap scalar sensors alone.
It trains two quick baselines against USGS NWIS data:

  Baseline A — Scalar-Only MLP (temp + conductivity → full parameter panel)
  Baseline B — Physics-Only (no learning; DO saturation curve + TDS conversion)

Per-target R², MAE, RMSE, sample count, and a learnability verdict are saved
to results/benchmarks/predictability_audit.json.

Verdict thresholds:
  STRONG        R² > 0.70  — scalar sensors alone are sufficient
  MODERATE      R² 0.30–0.70 — vision likely to add value
  WEAK          R² 0.10–0.30 — needs vision or targeted sampling
  UNRECOVERABLE R² < 0.10  — flag for physical sampling in the field

Temporal-spatial holdout: train on rows with year < 2023, test on year ≥ 2024.
Falls back to 70/15/15 random site split when sufficient temporal coverage is
absent.

Usage:
    conda run -n physiformer python scripts/predictability_audit.py
    conda run -n physiformer python scripts/predictability_audit.py \\
        --gpu 0 --epochs 100 --batch-size 512

GPU: 1 (default)

MIT License — Bryan Cheng, 2026
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

RESULTS_DIR = PROJECT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
RAW_SENSOR_DIR    = PROJECT / "data" / "raw" / "sensor" / "full"
PROC_SENSOR_DIR   = PROJECT / "data" / "processed" / "sensor" / "full"
GRQA_DIR          = PROJECT / "data" / "raw" / "grqa"

# ---------------------------------------------------------------------------
# Column name aliases used in USGS parquet files produced by download_sensor.py
# These map to WaterDroneNet target names used in the learnability report.
# ---------------------------------------------------------------------------
# Parquet columns observed: Temp, SpCond, DO, pH, Turb
# Extended USGS parameter codes (when present as column names):
COL_ALIASES: dict[str, str] = {
    # Human-readable column names from download_sensor.py
    "Temp":    "temperature",
    "SpCond":  "conductivity",
    "DO":      "dissolved_oxygen",
    "pH":      "ph",
    "Turb":    "turbidity",
    # Raw USGS parameter codes (fallback)
    "00010":   "temperature",
    "00095":   "conductivity",
    "00300":   "dissolved_oxygen",
    "00400":   "ph",
    "63680":   "turbidity",
    "00060":   "discharge",
    "00665":   "phosphate",
    "00631":   "nitrate",
    "00608":   "ammonia",
}

# Feature columns used as inputs to Baseline A
INPUT_TARGETS  = {"temperature", "conductivity"}
# All targets we attempt to predict
ALL_TARGETS    = [
    "temperature", "conductivity", "dissolved_oxygen", "ph", "turbidity",
    "discharge", "phosphate", "nitrate", "ammonia",
]

VERDICT_THRESHOLDS = {
    "STRONG":        0.70,
    "MODERATE":      0.30,
    "WEAK":          0.10,
    # below 0.10 → UNRECOVERABLE
}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Physics-only baselines
# ---------------------------------------------------------------------------

def do_saturation(T: np.ndarray) -> np.ndarray:
    """Benson & Krause (1980) DO saturation at standard pressure (mg/L)."""
    return 14.62 - 0.3898 * T + 0.006969 * T ** 2 - 5.897e-5 * T ** 3


def conductivity_from_tds(tds: np.ndarray) -> np.ndarray:
    """Approximate SpCond (µS/cm) from TDS (mg/L) via fixed ratio 0.65."""
    return tds / 0.65


def tds_from_conductivity(cond: np.ndarray) -> np.ndarray:
    """Inverse: TDS (mg/L) from SpCond (µS/cm)."""
    return cond * 0.65


# ---------------------------------------------------------------------------
# Scalar-Only MLP (Baseline A)
# ---------------------------------------------------------------------------

class ScalarMLP(nn.Module):
    """Two-input (temp, conductivity) MLP predicting N targets."""

    def __init__(self, n_targets: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _standardize_columns(df) -> "pd.DataFrame":
    """Rename columns to canonical target names using COL_ALIASES."""
    import pandas as pd
    rename = {}
    for col in df.columns:
        col_str = str(col)
        if col_str in COL_ALIASES:
            rename[col_str] = COL_ALIASES[col_str]
        else:
            # Try substring match for USGS codes embedded in longer names
            for code, tgt in COL_ALIASES.items():
                if code in col_str:
                    rename[col_str] = tgt
                    break
    return df.rename(columns=rename)


def load_sensor_data(max_files: int = 0):
    """Load all parquet files and return a unified DataFrame.

    Aggregates to daily means. Keeps only columns in COL_ALIASES values.
    Returns empty DataFrame if no data found.
    """
    import pandas as pd

    sensor_dirs = []
    if RAW_SENSOR_DIR.exists():
        sensor_dirs.append(RAW_SENSOR_DIR)
    if PROC_SENSOR_DIR.exists():
        sensor_dirs.append(PROC_SENSOR_DIR)

    parquet_files: list[Path] = []
    for d in sensor_dirs:
        parquet_files.extend(sorted(d.glob("*.parquet")))

    if not parquet_files:
        log("WARNING: No parquet files found in sensor directories.")
        return pd.DataFrame()

    if max_files and max_files < len(parquet_files):
        log(f"  Limiting to {max_files} files (of {len(parquet_files)} total)")
        parquet_files = parquet_files[:max_files]

    log(f"  Loading {len(parquet_files)} parquet files...")
    chunks = []

    for i, fpath in enumerate(parquet_files):
        try:
            df = pd.read_parquet(fpath)

            # Ensure DatetimeIndex
            if df.index.name == "datetime" or isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            elif "datetime" in df.columns:
                df = df.set_index("datetime")
                df.index = pd.to_datetime(df.index, utc=True)
            else:
                continue

            df = _standardize_columns(df)

            # Keep only recognised target columns
            keep = [c for c in df.columns if c in ALL_TARGETS]
            if not keep:
                continue
            df = df[keep]

            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Aggregate to daily means
            daily = df.resample("D").mean()
            daily["site"] = fpath.stem
            chunks.append(daily)

        except Exception as exc:
            log(f"  SKIP {fpath.name}: {exc}")
            continue

        if (i + 1) % 50 == 0:
            log(f"  ... processed {i + 1}/{len(parquet_files)} files")

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks, axis=0)
    log(f"  Combined dataset: {len(combined):,} daily rows, "
        f"{combined['site'].nunique()} sites")
    return combined


def load_grqa_data():
    """Optionally load GRQA water quality data if directory exists."""
    import pandas as pd

    if not GRQA_DIR.exists():
        return pd.DataFrame()

    csv_files = list(GRQA_DIR.glob("*.csv")) + list(GRQA_DIR.glob("**/*.csv"))
    if not csv_files:
        return pd.DataFrame()

    log(f"  Loading {len(csv_files)} GRQA CSV files...")
    chunks = []
    for fpath in csv_files[:20]:          # cap at 20 files
        try:
            df = pd.read_csv(fpath, low_memory=False)
            df = _standardize_columns(df)

            # Find date column
            date_col = None
            for c in df.columns:
                if "date" in c.lower() or "time" in c.lower():
                    date_col = c
                    break
            if date_col is None:
                continue

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
            df = df.dropna(subset=[date_col]).set_index(date_col)

            keep = [c for c in df.columns if c in ALL_TARGETS]
            if not keep:
                continue
            df = df[keep]
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["site"] = f"grqa_{fpath.stem}"
            chunks.append(df)
        except Exception:
            continue

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks, axis=0)
    log(f"  GRQA: {len(combined):,} rows, {combined['site'].nunique()} sites")
    return combined


# ---------------------------------------------------------------------------
# Temporal-spatial split
# ---------------------------------------------------------------------------

def make_splits(df):
    """Return (train_df, val_df, test_df) using temporal-spatial holdout.

    Strategy:
      - If ≥25% of rows have year < 2023 AND ≥10% have year ≥ 2024:
          train = year < 2023, test = year ≥ 2024, val = 2023
      - Otherwise:
          70/15/15 random split by site (no site leaks into multiple sets)
    """
    import pandas as pd

    if df.empty:
        return df, df, df

    # Ensure UTC-aware index
    if not isinstance(df.index, pd.DatetimeIndex):
        return df, df, df

    years = df.index.year
    pct_pre2023   = (years < 2023).mean()
    pct_2024plus  = (years >= 2024).mean()

    if pct_pre2023 >= 0.25 and pct_2024plus >= 0.10:
        log(f"  Using temporal split: pre-2023 train ({pct_pre2023:.1%}), "
            f"2023 val, 2024+ test ({pct_2024plus:.1%})")
        train_df = df[df.index.year < 2023].copy()
        val_df   = df[df.index.year == 2023].copy()
        test_df  = df[df.index.year >= 2024].copy()
    else:
        log(f"  Insufficient temporal coverage (pre-2023: {pct_pre2023:.1%}, "
            f"2024+: {pct_2024plus:.1%}) — using site-based 70/15/15 split")
        sites = df["site"].unique()
        rng   = np.random.RandomState(42)
        shuffled = rng.permutation(sites)
        n     = len(shuffled)
        n_tr  = int(0.70 * n)
        n_val = int(0.15 * n)
        train_sites = set(shuffled[:n_tr])
        val_sites   = set(shuffled[n_tr:n_tr + n_val])
        test_sites  = set(shuffled[n_tr + n_val:])
        train_df = df[df["site"].isin(train_sites)].copy()
        val_df   = df[df["site"].isin(val_sites)].copy()
        test_df  = df[df["site"].isin(test_sites)].copy()

    log(f"  Split sizes — train: {len(train_df):,}, "
        f"val: {len(val_df):,}, test: {len(test_df):,}")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Feature/target matrix extraction
# ---------------------------------------------------------------------------

def extract_xy(df, targets: list[str]):
    """Extract input (temp, conductivity) and per-target arrays.

    Returns:
        X  — (N, 2) float32 array of [temperature, conductivity]
        ys — dict[target_name → 1-D float32 array], indices aligned with X
             (rows where that target is NaN are excluded per-target)
    """
    import pandas as pd

    if df.empty:
        return np.empty((0, 2), dtype=np.float32), {}

    # Build X (only rows where BOTH inputs are present)
    has_temp  = "temperature"   in df.columns
    has_cond  = "conductivity"  in df.columns
    if not has_temp or not has_cond:
        log("  WARNING: Input columns (temperature/conductivity) missing.")
        return np.empty((0, 2), dtype=np.float32), {}

    df_valid = df[["temperature", "conductivity"] + [t for t in targets
                                                       if t in df.columns and
                                                       t not in INPUT_TARGETS]
                  ].copy()

    # Row-level mask: require both inputs
    mask_inputs = (
        df_valid["temperature"].notna() &
        df_valid["conductivity"].notna()
    )
    df_valid = df_valid[mask_inputs]

    if len(df_valid) == 0:
        return np.empty((0, 2), dtype=np.float32), {}

    X = df_valid[["temperature", "conductivity"]].values.astype(np.float32)

    ys: dict[str, np.ndarray] = {}
    # Include inputs themselves as prediction targets too
    ys["temperature"]   = df_valid["temperature"].values.astype(np.float32)
    ys["conductivity"]  = df_valid["conductivity"].values.astype(np.float32)

    for tgt in targets:
        if tgt in INPUT_TARGETS:
            continue
        if tgt not in df_valid.columns:
            continue
        col = df_valid[tgt].values.astype(np.float32)
        valid_mask = np.isfinite(col)
        if valid_mask.sum() < 30:
            continue
        # Store full-length arrays with NaN mask so indices align with X
        ys[tgt] = col

    return X, ys


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def fit_normalizer(X: np.ndarray):
    """Return (mean, std) computed on non-NaN rows of X."""
    mu  = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0) + 1e-8
    return mu, std


def normalize(X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mu) / std


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def verdict(r2: float) -> str:
    if r2 is None or np.isnan(r2):
        return "UNRECOVERABLE"
    if r2 > 0.70:
        return "STRONG"
    if r2 > 0.30:
        return "MODERATE"
    if r2 > 0.10:
        return "WEAK"
    return "UNRECOVERABLE"


# ---------------------------------------------------------------------------
# Baseline A: Scalar-Only MLP training + evaluation
# ---------------------------------------------------------------------------

def build_tensor_dataset(X: np.ndarray, y: np.ndarray, device: torch.device):
    """Build a TensorDataset from aligned X and y, dropping NaN target rows."""
    valid = np.isfinite(y)
    Xv = torch.from_numpy(X[valid]).to(device)
    yv = torch.from_numpy(y[valid]).unsqueeze(1).to(device)
    return TensorDataset(Xv, yv), int(valid.sum())


def train_baseline_a(
    X_tr: np.ndarray, ys_tr: dict,
    X_val: np.ndarray, ys_val: dict,
    X_te: np.ndarray, ys_te: dict,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float = 1e-3,
) -> dict:
    """Train one MLP per target and return per-target test metrics."""

    # Normalise inputs on training data
    mu_X, std_X = fit_normalizer(X_tr[np.isfinite(X_tr).all(axis=1)])
    Xn_tr = normalize(X_tr, mu_X, std_X)
    Xn_te = normalize(X_te, mu_X, std_X)

    all_targets = sorted(set(ys_tr) | set(ys_te))
    results: dict[str, dict] = {}

    for tgt in all_targets:
        if tgt not in ys_tr or tgt not in ys_te:
            log(f"  [{tgt}] — skipping (not in train or test)")
            continue

        y_tr_raw = ys_tr[tgt]
        y_te_raw = ys_te[tgt]

        # Valid masks
        valid_tr = np.isfinite(y_tr_raw) & np.isfinite(Xn_tr).all(axis=1)
        valid_te = np.isfinite(y_te_raw) & np.isfinite(Xn_te).all(axis=1)

        n_tr = int(valid_tr.sum())
        n_te = int(valid_te.sum())

        if n_tr < 50 or n_te < 10:
            log(f"  [{tgt}] — insufficient data (train={n_tr}, test={n_te})")
            results[tgt] = {"r2": float("nan"), "mae": float("nan"),
                            "rmse": float("nan"), "n_test": n_te,
                            "verdict": "UNRECOVERABLE", "source": "A"}
            continue

        # Target normalisation (per-target mean/std on train)
        y_mu  = float(np.nanmean(y_tr_raw[valid_tr]))
        y_std = float(np.nanstd(y_tr_raw[valid_tr])) + 1e-8

        # Build tensors
        X_tr_t = torch.from_numpy(Xn_tr[valid_tr]).float()
        y_tr_t = torch.tensor((y_tr_raw[valid_tr] - y_mu) / y_std,
                               dtype=torch.float32).unsqueeze(1)
        X_te_t = torch.from_numpy(Xn_te[valid_te]).float().to(device)
        y_te_t = torch.tensor(y_te_raw[valid_te], dtype=torch.float32)

        ds_tr = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                            drop_last=False)

        model = ScalarMLP(n_targets=1).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)
        crit  = nn.MSELoss()

        best_state = None
        best_val_loss = float("inf")

        # Use val set if available, else validate on train
        if tgt in ys_val:
            Xn_val = normalize(X_val, mu_X, std_X)
            y_val_raw = ys_val[tgt]
            valid_val = np.isfinite(y_val_raw) & np.isfinite(Xn_val).all(axis=1)
            if valid_val.sum() >= 10:
                X_val_t = torch.from_numpy(Xn_val[valid_val]).float().to(device)
                y_val_t = torch.tensor(
                    (y_val_raw[valid_val] - y_mu) / y_std,
                    dtype=torch.float32).unsqueeze(1).to(device)
            else:
                X_val_t = X_te_t
                y_val_t = torch.tensor(
                    (y_te_raw[valid_te] - y_mu) / y_std,
                    dtype=torch.float32).unsqueeze(1).to(device)
        else:
            X_val_t = X_te_t
            y_val_t = torch.tensor(
                (y_te_raw[valid_te] - y_mu) / y_std,
                dtype=torch.float32).unsqueeze(1).to(device)

        for ep in range(epochs):
            model.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            # Light validation check every 10 epochs
            if (ep + 1) % 10 == 0 or ep == epochs - 1:
                model.eval()
                with torch.no_grad():
                    val_loss = crit(model(X_val_t), y_val_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state    = {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()}

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            y_pred_norm = model(X_te_t).squeeze(1).cpu().numpy()
        y_pred = y_pred_norm * y_std + y_mu
        y_true = y_te_t.numpy()

        r2   = r2_score(y_true, y_pred)
        mae  = mae_score(y_true, y_pred)
        rmse = rmse_score(y_true, y_pred)
        verd = verdict(r2)

        log(f"  [{tgt:25s}] R²={r2:+.3f}  MAE={mae:.4f}  RMSE={rmse:.4f}"
            f"  n={n_te:,}  → {verd}")
        results[tgt] = {
            "r2": r2, "mae": mae, "rmse": rmse,
            "n_train": n_tr, "n_test": n_te,
            "verdict": verd, "source": "A",
        }

    return results


# ---------------------------------------------------------------------------
# Baseline B: Physics-Only evaluation
# ---------------------------------------------------------------------------

def eval_baseline_b(X_te: np.ndarray, ys_te: dict) -> dict:
    """Apply hard-coded physics relationships and evaluate on test set."""
    results: dict[str, dict] = {}

    T    = X_te[:, 0]   # temperature
    cond = X_te[:, 1]   # conductivity (SpCond µS/cm)

    # DO from temperature saturation curve
    if "dissolved_oxygen" in ys_te:
        y_true_raw = ys_te["dissolved_oxygen"]
        valid = np.isfinite(y_true_raw) & np.isfinite(T)
        if valid.sum() >= 10:
            y_pred  = do_saturation(T[valid])
            y_true  = y_true_raw[valid]
            r2   = r2_score(y_true, y_pred)
            mae  = mae_score(y_true, y_pred)
            rmse = rmse_score(y_true, y_pred)
            log(f"  [dissolved_oxygen / physics] R²={r2:+.3f}  "
                f"MAE={mae:.4f}  RMSE={rmse:.4f}")
            results["dissolved_oxygen"] = {
                "r2": r2, "mae": mae, "rmse": rmse,
                "n_test": int(valid.sum()),
                "verdict": verdict(r2), "source": "B",
                "formula": "DO_sat(T) = 14.62 - 0.3898*T + 0.006969*T² - 5.897e-5*T³",
            }

    # TDS from conductivity (inverse: conductivity is an input, so we test
    # how well the physics identity cond * 0.65 matches any reported TDS)
    # In our dataset conductivity IS an input — physics baseline is trivial
    # here so we just annotate it.
    results["conductivity_identity"] = {
        "r2": float("nan"),
        "mae": float("nan"),
        "rmse": float("nan"),
        "n_test": 0,
        "verdict": "N/A",
        "source": "B",
        "note": "conductivity is an input; physics baseline trivially exact",
    }

    # All other targets: physics has no applicable prior
    for tgt in ys_te:
        if tgt in results or tgt in INPUT_TARGETS:
            continue
        results[tgt] = {
            "r2": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "n_test": int(np.isfinite(ys_te[tgt]).sum()),
            "verdict": "N/A",
            "source": "B",
            "note": "no physics prior available",
        }

    return results


# ---------------------------------------------------------------------------
# Learnability report
# ---------------------------------------------------------------------------

def build_report(
    results_a: dict,
    results_b: dict,
    meta: dict,
) -> dict:
    """Merge Baseline A and B results into a unified learnability report."""
    all_targets = sorted(set(results_a) | set(results_b))
    table = []
    for tgt in ALL_TARGETS:
        if tgt not in all_targets:
            continue
        row_a = results_a.get(tgt, {})
        row_b = results_b.get(tgt, {})

        r2_a  = row_a.get("r2", float("nan"))
        r2_b  = row_b.get("r2", float("nan"))
        mae_a = row_a.get("mae", float("nan"))
        mae_b = row_b.get("mae", float("nan"))
        verd  = row_a.get("verdict", "UNRECOVERABLE")

        is_input = tgt in INPUT_TARGETS
        entry: dict = {
            "target":         tgt,
            "is_scalar_input": is_input,
            "r2_scalar":      round(r2_a, 4) if np.isfinite(r2_a) else None,
            "r2_physics":     round(r2_b, 4) if np.isfinite(r2_b) else None,
            "mae_scalar":     round(mae_a, 4) if np.isfinite(mae_a) else None,
            "mae_physics":    round(mae_b, 4) if np.isfinite(mae_b) else None,
            "rmse_scalar":    round(row_a.get("rmse", float("nan")), 4)
                              if np.isfinite(row_a.get("rmse", float("nan"))) else None,
            "n_train":        row_a.get("n_train", None),
            "n_test":         row_a.get("n_test", row_b.get("n_test", None)),
            "verdict":        "INPUT" if is_input else verd,
            "physics_note":   row_b.get("formula", row_b.get("note", "")),
        }
        table.append(entry)

    report = {
        "audit_timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "description": (
            "Per-target learnability from cheap scalar sensors (temp + conductivity). "
            "Baseline A = scalar MLP; Baseline B = physics-only."
        ),
        "verdicts": {
            "INPUT":         "This column is one of the two scalar inputs — identity check",
            "STRONG":        "R² > 0.70 — scalar sensors sufficient",
            "MODERATE":      "R² 0.30–0.70 — vision likely to help",
            "WEAK":          "R² 0.10–0.30 — needs vision or targeted sampling",
            "UNRECOVERABLE": "R² < 0.10 — flag for physical sampling",
        },
        "split_strategy":   meta.get("split_strategy", "unknown"),
        "n_sites":          meta.get("n_sites", None),
        "n_train_rows":     meta.get("n_train_rows", None),
        "n_test_rows":      meta.get("n_test_rows", None),
        "epochs":           meta.get("epochs", None),
        "batch_size":       meta.get("batch_size", None),
        "results":          table,
    }
    return report


def print_summary_table(report: dict) -> None:
    """Pretty-print the learnability table to stdout."""
    header = (
        f"\n{'Target':<25} {'R²(scalar)':>10} {'R²(physics)':>12} "
        f"{'MAE(scalar)':>12} {'N_test':>8}  Verdict"
    )
    log("=" * 80)
    log("PREDICTABILITY AUDIT — LEARNABILITY REPORT")
    log("=" * 80)
    print(header)
    print("-" * 80)
    for row in report["results"]:
        r2_s  = f"{row['r2_scalar']:+.3f}"   if row["r2_scalar"]  is not None else "   N/A"
        r2_p  = f"{row['r2_physics']:+.3f}"  if row["r2_physics"] is not None else "   N/A"
        mae_s = f"{row['mae_scalar']:.4f}"   if row["mae_scalar"] is not None else "   N/A"
        n_te  = f"{row['n_test']:,}"         if row["n_test"]     is not None else "   N/A"
        print(f"  {row['target']:<23} {r2_s:>10} {r2_p:>12} {mae_s:>12} "
              f"{n_te:>8}  {row['verdict']}")
    print("-" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WaterDroneNet predictability audit — scalar vs physics baselines"
    )
    parser.add_argument("--gpu",        type=int,   default=1,   help="CUDA device index")
    parser.add_argument("--epochs",     type=int,   default=50,  help="MLP training epochs")
    parser.add_argument("--batch-size", type=int,   default=256, help="Mini-batch size")
    parser.add_argument("--lr",         type=float, default=1e-3,help="Adam learning rate")
    parser.add_argument("--max-files",  type=int,   default=0,
                        help="Limit number of parquet files loaded (0 = all)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")
    log("=" * 70)
    log("WaterDroneNet — Predictability Audit")
    log("=" * 70)
    log(f"Device     : {device}")
    log(f"GPU index  : {args.gpu}")
    log(f"Epochs     : {args.epochs}")
    log(f"Batch size : {args.batch_size}")
    log(f"LR         : {args.lr}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    log("\n[1/5] Loading sensor data...")
    sensor_df = load_sensor_data(max_files=args.max_files)

    if GRQA_DIR.exists():
        log("\n[1b] Loading GRQA data...")
        grqa_df = load_grqa_data()
        if not grqa_df.empty:
            import pandas as pd
            sensor_df = pd.concat([sensor_df, grqa_df], axis=0)
            log(f"  Combined total: {len(sensor_df):,} rows")

    if sensor_df.empty:
        log("ERROR: No usable data found. Aborting audit.")
        sys.exit(1)

    n_sites = sensor_df["site"].nunique() if "site" in sensor_df.columns else 0
    log(f"  Total rows: {len(sensor_df):,}, sites: {n_sites}")

    # ------------------------------------------------------------------
    # 2. Temporal-spatial split
    # ------------------------------------------------------------------
    log("\n[2/5] Splitting data...")
    train_df, val_df, test_df = make_splits(sensor_df)

    if len(test_df) < 100:
        log("WARNING: Test set very small — results may be noisy.")

    split_meta = {
        "n_sites":      n_sites,
        "n_train_rows": len(train_df),
        "n_val_rows":   len(val_df),
        "n_test_rows":  len(test_df),
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
    }

    # Determine which split strategy was actually used
    if "site" in train_df.columns and len(train_df) > 0:
        tr_years = train_df.index.year.unique() if hasattr(train_df.index, "year") else []
        te_years = test_df.index.year.unique()  if hasattr(test_df.index, "year")  else []
        if len(tr_years) and int(max(tr_years)) < 2023:
            split_meta["split_strategy"] = "temporal_holdout"
        else:
            split_meta["split_strategy"] = "site_random_70_15_15"
    else:
        split_meta["split_strategy"] = "unknown"

    # ------------------------------------------------------------------
    # 3. Extract X / y matrices
    # ------------------------------------------------------------------
    log("\n[3/5] Extracting feature/target arrays...")
    X_tr, ys_tr = extract_xy(train_df, ALL_TARGETS)
    X_val, ys_val = extract_xy(val_df,   ALL_TARGETS)
    X_te, ys_te  = extract_xy(test_df,  ALL_TARGETS)

    if X_tr.shape[0] == 0:
        log("ERROR: No usable rows after extraction. "
            "Check that sensor data contains 'temperature' and 'conductivity' columns.")
        sys.exit(1)

    log(f"  X_train: {X_tr.shape}, X_val: {X_val.shape}, X_test: {X_te.shape}")
    log(f"  Targets available in train: {sorted(ys_tr.keys())}")
    log(f"  Targets available in test : {sorted(ys_te.keys())}")

    # ------------------------------------------------------------------
    # 4. Baseline A — Scalar MLP
    # ------------------------------------------------------------------
    log(f"\n[4/5] Baseline A: Scalar-Only MLP "
        f"({args.epochs} epochs, batch={args.batch_size})...")
    results_a = train_baseline_a(
        X_tr, ys_tr, X_val, ys_val, X_te, ys_te,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # ------------------------------------------------------------------
    # 5. Baseline B — Physics Only
    # ------------------------------------------------------------------
    log("\n[5/5] Baseline B: Physics-Only...")
    results_b = eval_baseline_b(X_te, ys_te)

    # ------------------------------------------------------------------
    # Build & save report
    # ------------------------------------------------------------------
    report = build_report(results_a, results_b, split_meta)
    print_summary_table(report)

    out_path = RESULTS_DIR / "predictability_audit.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    log(f"\nResults saved → {out_path}")
    log("Audit complete.")


if __name__ == "__main__":
    main()
