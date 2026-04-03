"""
ECOTOX data preprocessing for SENTINEL.

Pipeline:
  1. Standardize endpoints to common units (mg/L, hours)
  2. Handle missing data (drop records without chemical identity/concentration)
  3. Impute missing species attributes from taxonomic databases
  4. Train/val/test split by chemical (80/10/10) for generalization testing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sentinel.utils.config import load_config
from sentinel.utils.logging import get_logger, make_progress

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Unit conversion tables
# ---------------------------------------------------------------------------

# Concentration unit -> multiplier to convert to mg/L
CONC_UNIT_TO_MG_L: dict[str, float] = {
    "mg/L": 1.0,
    "mg/l": 1.0,
    "ug/L": 1e-3,
    "ug/l": 1e-3,
    "µg/L": 1e-3,
    "ng/L": 1e-6,
    "ng/l": 1e-6,
    "ppm": 1.0,         # ppm ~ mg/L in dilute aqueous
    "ppb": 1e-3,         # ppb ~ ug/L
    "ppt": 1e-6,         # parts per trillion ~ ng/L
    "g/L": 1e3,
    "g/l": 1e3,
    "mol/L": None,       # needs molecular weight; handled separately
    "M": None,
    "mM": None,
    "uM": None,
    "nM": None,
    "mg/kg": 1.0,        # approximate for dilute aqueous
    "AI mg/L": 1.0,      # active ingredient mg/L
    "%": 1e4,            # 1% = 10000 mg/L
}

# Duration unit -> multiplier to convert to hours
DURATION_UNIT_TO_HOURS: dict[str, float] = {
    "h": 1.0,
    "hr": 1.0,
    "hrs": 1.0,
    "hour": 1.0,
    "hours": 1.0,
    "d": 24.0,
    "day": 24.0,
    "days": 24.0,
    "wk": 168.0,
    "week": 168.0,
    "weeks": 168.0,
    "mo": 720.0,         # ~30 days
    "month": 720.0,
    "months": 720.0,
    "min": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
    "yr": 8760.0,
    "year": 8760.0,
    "years": 8760.0,
}


# ---------------------------------------------------------------------------
# Unit standardization
# ---------------------------------------------------------------------------


def standardize_concentration(
    df: pd.DataFrame,
    *,
    conc_col: str = "concentration",
    unit_col: str = "concentration_unit",
    output_col: str = "conc_mg_L",
) -> pd.DataFrame:
    """Convert concentration values to mg/L.

    Records with unconvertible units (e.g., molar without MW) are set to NaN.
    """
    df = df.copy()
    df[output_col] = np.nan

    for unit, factor in CONC_UNIT_TO_MG_L.items():
        if factor is None:
            continue
        mask = df[unit_col].astype(str).str.strip().str.lower() == unit.lower()
        if mask.any():
            df.loc[mask, output_col] = (
                pd.to_numeric(df.loc[mask, conc_col], errors="coerce") * factor
            )

    converted = df[output_col].notna().sum()
    total = len(df)
    logger.info(f"Concentration standardization: {converted}/{total} converted to mg/L")
    return df


def standardize_duration(
    df: pd.DataFrame,
    *,
    dur_col: str = "duration_value",
    unit_col: str = "duration_unit",
    output_col: str = "duration_hours",
) -> pd.DataFrame:
    """Convert exposure duration to hours."""
    df = df.copy()
    df[output_col] = np.nan

    for unit, factor in DURATION_UNIT_TO_HOURS.items():
        mask = df[unit_col].astype(str).str.strip().str.lower() == unit.lower()
        if mask.any():
            df.loc[mask, output_col] = (
                pd.to_numeric(df.loc[mask, dur_col], errors="coerce") * factor
            )

    converted = df[output_col].notna().sum()
    logger.info(f"Duration standardization: {converted}/{len(df)} converted to hours")
    return df


# ---------------------------------------------------------------------------
# Missing data handling
# ---------------------------------------------------------------------------


def handle_missing_data(
    df: pd.DataFrame,
    *,
    require_cas: bool = True,
    require_concentration: bool = True,
    require_endpoint: bool = True,
) -> pd.DataFrame:
    """Drop records missing critical fields.

    Parameters
    ----------
    df:
        ECOTOX DataFrame with standardized columns.
    require_cas:
        Drop records without CAS number.
    require_concentration:
        Drop records without valid concentration.
    require_endpoint:
        Drop records without endpoint type.

    Returns
    -------
    Cleaned DataFrame.
    """
    original = len(df)

    if require_cas and "cas_number" in df.columns:
        df = df[df["cas_number"].notna() & (df["cas_number"].astype(str).str.strip() != "")]
        logger.debug(f"After CAS filter: {len(df)}/{original}")

    if require_concentration and "conc_mg_L" in df.columns:
        df = df[df["conc_mg_L"].notna() & (df["conc_mg_L"] > 0)]
        logger.debug(f"After concentration filter: {len(df)}/{original}")

    if require_endpoint and "endpoint" in df.columns:
        df = df[df["endpoint"].notna() & (df["endpoint"].astype(str).str.strip() != "")]
        logger.debug(f"After endpoint filter: {len(df)}/{original}")

    logger.info(f"Missing data handling: {len(df)}/{original} records retained")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Species attribute imputation
# ---------------------------------------------------------------------------

# Taxonomic group defaults for common aquatic test organisms
SPECIES_DEFAULTS: dict[str, dict[str, Any]] = {
    "Daphnia magna": {
        "phylum": "Arthropoda", "class": "Branchiopoda",
        "order": "Cladocera", "trophic_level": "primary_consumer",
        "habitat": "freshwater", "body_mass_mg": 0.1,
    },
    "Danio rerio": {
        "phylum": "Chordata", "class": "Actinopterygii",
        "order": "Cypriniformes", "trophic_level": "omnivore",
        "habitat": "freshwater", "body_mass_mg": 500.0,
    },
    "Oncorhynchus mykiss": {
        "phylum": "Chordata", "class": "Actinopterygii",
        "order": "Salmoniformes", "trophic_level": "predator",
        "habitat": "freshwater", "body_mass_mg": 50000.0,
    },
    "Pimephales promelas": {
        "phylum": "Chordata", "class": "Actinopterygii",
        "order": "Cypriniformes", "trophic_level": "omnivore",
        "habitat": "freshwater", "body_mass_mg": 3000.0,
    },
}


def impute_species_attributes(
    df: pd.DataFrame,
    species_table: pd.DataFrame | None = None,
    *,
    species_col: str = "species_id",
    species_name_col: str = "species_name",
) -> pd.DataFrame:
    """Impute missing species attributes from a taxonomic lookup table
    or built-in defaults.

    Parameters
    ----------
    df:
        ECOTOX DataFrame with species identifiers.
    species_table:
        Optional species metadata table from ECOTOX.
    species_col:
        Column containing species identifier.
    species_name_col:
        Column containing species name (used for default lookup).

    Returns
    -------
    DataFrame with added/imputed species attribute columns.
    """
    df = df.copy()
    attrs_to_add = ["phylum", "class", "order", "trophic_level", "habitat", "body_mass_mg"]

    for attr in attrs_to_add:
        if attr not in df.columns:
            df[attr] = None

    # If species table provided, merge
    if species_table is not None and species_col in df.columns:
        sp_cols = [c for c in attrs_to_add if c in species_table.columns]
        if sp_cols and species_col in species_table.columns:
            merged = df.merge(
                species_table[[species_col] + sp_cols],
                on=species_col,
                how="left",
                suffixes=("", "_imp"),
            )
            for col in sp_cols:
                imp_col = f"{col}_imp"
                if imp_col in merged.columns:
                    merged[col] = merged[col].fillna(merged[imp_col])
                    merged.drop(columns=[imp_col], inplace=True)
            df = merged

    # Fill from built-in defaults where available
    if species_name_col in df.columns:
        for sp_name, defaults in SPECIES_DEFAULTS.items():
            mask = df[species_name_col].astype(str).str.contains(sp_name, na=False)
            for attr, value in defaults.items():
                if attr in df.columns:
                    df.loc[mask & df[attr].isna(), attr] = value

    n_imputed = sum(df[attr].notna().sum() for attr in attrs_to_add)
    logger.info(f"Species attribute imputation: {n_imputed} total attribute values")
    return df


# ---------------------------------------------------------------------------
# Train/val/test split by chemical
# ---------------------------------------------------------------------------


def split_by_chemical(
    df: pd.DataFrame,
    *,
    cas_col: str = "cas_number",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split data by chemical identity for generalization testing.

    All records for a given CAS number go into the same split, preventing
    data leakage.

    Parameters
    ----------
    df:
        ECOTOX DataFrame.
    cas_col:
        Column with CAS numbers.
    train_frac, val_frac, test_frac:
        Split proportions (must sum to 1.0).
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    Dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to DataFrames.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"

    rng = np.random.RandomState(random_seed)
    chemicals = df[cas_col].unique()
    rng.shuffle(chemicals)

    n_train = int(len(chemicals) * train_frac)
    n_val = int(len(chemicals) * val_frac)

    train_chems = set(chemicals[:n_train])
    val_chems = set(chemicals[n_train : n_train + n_val])
    test_chems = set(chemicals[n_train + n_val :])

    splits = {
        "train": df[df[cas_col].isin(train_chems)].reset_index(drop=True),
        "val": df[df[cas_col].isin(val_chems)].reset_index(drop=True),
        "test": df[df[cas_col].isin(test_chems)].reset_index(drop=True),
    }

    for name, split_df in splits.items():
        n_chems = split_df[cas_col].nunique()
        logger.info(
            f"Split '{name}': {len(split_df)} records, {n_chems} chemicals"
        )

    return splits


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_ecotox(
    input_path: str | Path,
    output_dir: str | Path = "data/ecotox/processed",
    *,
    species_table_path: str | Path | None = None,
) -> dict[str, Any]:
    """Full ECOTOX preprocessing pipeline.

    Steps:
      1. Load filtered ECOTOX data
      2. Standardize units
      3. Handle missing data
      4. Impute species attributes
      5. Split by chemical

    Parameters
    ----------
    input_path:
        Path to filtered ECOTOX Parquet file.
    output_dir:
        Directory for output files.
    species_table_path:
        Optional path to species metadata table.

    Returns
    -------
    Summary dict with split sizes and output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading ECOTOX data from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} records")

    # Standardize units
    df = standardize_concentration(df)
    df = standardize_duration(df)

    # Handle missing data
    df = handle_missing_data(df)

    # Impute species attributes
    species_table = None
    if species_table_path:
        species_table = pd.read_csv(species_table_path, sep="|", encoding="latin-1")
    df = impute_species_attributes(df, species_table)

    # Split by chemical
    splits = split_by_chemical(df)

    # Save
    for split_name, split_df in splits.items():
        out_path = output_dir / f"ecotox_{split_name}.parquet"
        split_df.to_parquet(out_path)

    # Also save the full standardized dataset
    df.to_parquet(output_dir / "ecotox_standardized.parquet")

    summary = {
        "total_records": len(df),
        "splits": {k: len(v) for k, v in splits.items()},
        "chemicals": int(df["cas_number"].nunique()) if "cas_number" in df.columns else 0,
        "endpoints": df["endpoint"].value_counts().to_dict() if "endpoint" in df.columns else {},
        "output_dir": str(output_dir),
    }

    with open(output_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"ECOTOX preprocessing complete: {len(df)} records -> {output_dir}")
    return summary
