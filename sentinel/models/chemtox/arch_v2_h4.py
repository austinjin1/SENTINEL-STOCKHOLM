"""H4 (BCF / BCFD / BAF) join and QC rules for the M4 arch-v2 ablation A5. Pure: no I/O.

New module, written under addendum 01 (ruling N2-A). It holds the rules that used to live in
``scripts/eval/chemtox_arch_v2_h4_report.py``, unchanged, as functions over frames the caller has
already loaded. **It opens no file.** The reads belong to
``arch_v2_guard.load_h4_rows()``, which is the only sanctioned reader (addendum 01 §3), so the QC
can be unit-tested on synthetic frames without going near ``ecotox_aquatic.parquet``.

``arch_v2.py`` is byte-frozen by ruling B1 and is not touched; nothing here changes a rule, a count
or a threshold of the stamped amendment §9.

The finding that matters, carried over verbatim: **the BCF value lives in ``bcf1_mean`` /
``bcf1_unit``, not ``conc1_mean``.** On these rows ``conc1_*`` is the exposure water concentration,
so reading it would silently train a bioconcentration head on the exposure concentration with no
error raised.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = ["BCF_ENDPOINT_RE", "DIMENSIONLESS", "PER_KG", "TRIM_LOG10", "DURATION_FACTORS",
           "norm_cas", "select_bcf_family", "parse_values", "admit_units", "attach_basis",
           "join_structures", "attach_duration_h", "trim_implausible", "qc_report"]

#: endpoint codes of the BCF family
BCF_ENDPOINT_RE = r"^(BCF|BCFD|BAF|BAFD)"

#: Only units that are exactly dimensionless-equivalent are admitted. ``L/g`` and ``ml/mg`` are
#: convertible with a factor but each needs its own decision, so they are excluded and counted
#: rather than guessed. ``RA`` is a relative-activity code, not a BCF.
DIMENSIONLESS = frozenset({"", "none", "nan", "ratio", "x", "AI", "--", "<NA>"})
PER_KG = frozenset({"L/kg", "l/kg", "ml/g", "mL/g", "L/kg bdwt", "L/kg wet wt", "L/kg dry wt"})

#: log10 BCF outside this range is a transcription artefact
TRIM_LOG10 = (-2.0, 7.0)

#: observation-duration unit -> hours
DURATION_FACTORS = {"h": 1.0, "hr": 1.0, "hours": 1.0, "d": 24.0, "dy": 24.0, "day": 24.0,
                    "days": 24.0, "wk": 168.0, "week": 168.0, "mo": 720.0, "month": 720.0}

_CENSOR_OPS = ("<", ">", "<=", ">=", "~")


def norm_cas(x) -> str:
    """Digit-stripping CAS normalisation: non-digits removed, leading zeros dropped."""
    return re.sub(r"\D", "", str(x)).lstrip("0")


def select_bcf_family(results: pd.DataFrame) -> pd.DataFrame:
    """Rows of the BCF family, with ``code`` attached."""
    sel = results[results["endpoint"].astype(str).str.match(BCF_ENDPOINT_RE, na=False)].copy()
    sel["code"] = sel["endpoint"].astype(str).str.extract(r"^(BCFD|BCF|BAFD|BAF)")[0]
    return sel


def parse_values(bcf: pd.DataFrame) -> pd.DataFrame:
    """Numeric BCF value from ``bcf1_mean`` (NOT ``conc1_mean``) and the censoring operator."""
    out = bcf.copy()
    out["op"] = out["bcf1_mean_op"].astype(str).str.strip()
    out["val"] = pd.to_numeric(
        out["bcf1_mean"].astype(str).str.replace(r"[^0-9eE.+-]", "", regex=True), errors="coerce")
    out["censored"] = out["op"].isin(_CENSOR_OPS)
    return out


def admit_units(bcf: pd.DataFrame) -> pd.DataFrame:
    """Flag rows whose ``bcf1_unit`` is exactly dimensionless-equivalent."""
    out = bcf.copy()
    u = out["bcf1_unit"].astype(str).str.strip()
    out["unit_norm"] = u
    out["unit_ok"] = u.isin(DIMENSIONLESS) | u.isin(PER_KG)
    return out


def attach_basis(bcf: pd.DataFrame) -> pd.DataFrame:
    """Dry vs wet weight. BCFD/BAFD is dry; the two differ by roughly 4-5x and are never pooled."""
    out = bcf.copy()
    u = out["unit_norm"] if "unit_norm" in out.columns else out["bcf1_unit"].astype(str)
    out["basis"] = np.where(out["code"].isin(["BCFD", "BAFD"]), "dry",
                            np.where(u.str.contains("dry", case=False, na=False), "dry", "wet"))
    return out


def join_structures(bcf: pd.DataFrame, tests: pd.DataFrame,
                    smiles_by_cas: dict[str, str],
                    species_by_number: dict[str, str]) -> pd.DataFrame:
    """Attach ``test_cas``, normalised CAS, SMILES and ``latin_name``.

    ``smiles_by_cas`` is keyed by NORMALISED CAS and ``species_by_number`` by ``species_number``;
    the caller builds both from the frames it owns.
    """
    out = bcf.merge(tests[["test_id", "test_cas", "species_number"]], on="test_id", how="left")
    out["cas_norm"] = out["test_cas"].map(norm_cas)
    out["smiles"] = out["cas_norm"].map(smiles_by_cas)
    out["latin_name"] = out["species_number"].map(species_by_number)
    return out


def attach_duration_h(bcf: pd.DataFrame) -> pd.DataFrame:
    """Observation duration in hours, where the unit is one this module recognises."""
    out = bcf.copy()
    val = pd.to_numeric(out["obs_duration_mean"], errors="coerce")
    unit = out["obs_duration_unit"].astype(str).str.strip().str.lower()
    out["duration_h"] = val * unit.map(DURATION_FACTORS)
    return out


def trim_implausible(bcf: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep rows with a positive value, an admissible unit, a structure and log10 BCF in range."""
    ok = bcf[bcf["val"].notna() & (bcf["val"] > 0) & bcf["unit_ok"]
             & bcf["smiles"].notna()].copy()
    ok["log10_bcf"] = np.log10(ok["val"])
    kept = ok[ok["log10_bcf"].between(*TRIM_LOG10)].copy()
    return kept, int(len(ok) - len(kept))


def qc_report(results_rows: int, family: pd.DataFrame, trimmed: pd.DataFrame,
              trimmed_out: int) -> dict:
    """Counts only. No label statistic of any sealed block (ruling L6)."""
    u = family["unit_norm"] if "unit_norm" in family.columns else pd.Series(dtype=str)
    return {
        "results_rows": int(results_rows),
        "bcf_family_rows": int(len(family)),
        "by_code": {str(k): int(v) for k, v in family["code"].value_counts().items()},
        "value_column_finding": (
            "the BCF VALUE is bcf1_mean/bcf1_unit, not conc1_mean; conc1_* on these rows is the "
            "exposure water concentration. Using conc1_* would have silently modelled the exposure "
            "concentration as the bioconcentration factor, with no error raised."),
        "value_parse": {"numeric": int(family["val"].notna().sum()),
                        "non_numeric_or_blank": int(family["val"].isna().sum()),
                        "censored_op": int(family["censored"].sum())},
        "units_top": {str(k): int(v) for k, v in u.value_counts().head(12).items()},
        "unit_admissible_rows": int(family["unit_ok"].sum()),
        "unit_excluded_top": {str(k): int(v) for k, v in
                              u[~family["unit_ok"]].value_counts().head(8).items()},
        "basis_note": ("BCFD/BAFD is dry weight; dry and wet factors differ by roughly 4-5x, so "
                       "basis is a categorical feature and is never pooled silently"),
        "cas_join": {"rows_with_smiles": int(family["smiles"].notna().sum()),
                     "distinct_cas": int(family["cas_norm"].nunique()),
                     "distinct_cas_with_smiles": int(
                         family.loc[family["smiles"].notna(), "cas_norm"].nunique())},
        "trim_range_log10": list(TRIM_LOG10),
        "unfiltered_usable": {
            "rows": int(len(trimmed)), "rows_trimmed_out": int(trimmed_out),
            "chemicals": int(trimmed["cas_norm"].nunique()),
            "species": int(trimmed["latin_name"].nunique()),
            "by_basis": {str(k): int(v) for k, v in trimmed["basis"].value_counts().items()},
            "note": "this is the pre-stamp figure and may NOT be quoted as a training-row count; "
                    "the admissible count is the AUX-LEAK-filtered one"},
    }
