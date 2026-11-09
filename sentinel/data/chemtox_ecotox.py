# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""ECOTOX ASCII tables -> tidy aquatic endpoint table for spill severity and SSDs.

Scope: ``tests.organism_habitat == 'Water'``; taxa fish, crustaceans, algae, insects, molluscs,
amphibians (from ``species.ecotox_group`` + taxonomy); endpoints L(E)C50/IC50, ECx/LCx (x <= 25),
NOEC/LOEC/NOEL/LOEL/MATC. One row per ECOTOX result. This is *not* a re-derivation of ADORE
(ADORE keeps its own curated subset and splits).

Standardisation (all factors exact, documented in UNIT tables below and unit-tested):
  * concentration: the unit string is split into a basis prefix (``AI`` active ingredient,
    ``ae`` acid equivalent, ...) and a dimensional core; mass/volume cores convert to mg/L,
    molar cores to mol/L; ``ppm``/``ppb`` are read as mg/L / ug/L (w/v, dilute aqueous media);
    cross conversion mg/L <-> mol/L uses the molecular weight of the substance as expressed:
    the ion's atomic mass when ``ion1`` names a single element (e.g. Cu), otherwise the MW of
    the tested chemical from the ``chemicals`` identity table. Anything else (v/v, %, per kg,
    per area, ambiguous ``ppt``) is kept with ``conc_conversion='not_convertible'``.
  * value: ``conc1_mean`` only (min/max are kept, never averaged into a value); a trailing ``*``
    (an ECOTOX annotation whose meaning the release does not document) is stripped and flagged
    ``asterisk``; ``NR``/``NC``/``--`` -> NaN. Asterisked endpoint codes (``LC50*``, ``EC50*``:
    separate codes with the same description) map to the base endpoint with ``endpoint_flag``.
    Log-transformed endpoints (``(log)LC50``, ``log (1/EC50)``) are excluded.
  * operators ``<``/``<=`` = left-censored, ``>``/``>=`` = right-censored, ``~``/``ca`` =
    approximate; values are never substituted.
  * duration: ``obs_duration_mean`` + unit -> hours for clock units (mi, h, d, wk, mo, yr);
    developmental units (hpf, dph, ...) keep NaN hours.
  * ``exposure_class``: acute = L(E)C50/IC50 with duration <= 96 h; prolonged = L(E)C50/IC50
    > 96 h; chronic = NOEC/LOEC/NOEL/LOEL/MATC/ECx/LCx (x <= 25); unknown = duration missing.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from sentinel.data.sources.chem_ids import normalize_cas

TAXON_GROUPS = ("fish", "crustaceans", "algae", "insects", "molluscs", "amphibians")
MEDIAN_ENDPOINTS = ("LC50", "EC50", "IC50")
LOW_EFFECT_ENDPOINTS = (
    "EC10",
    "EC20",
    "EC25",
    "EC05",
    "LC10",
    "LC20",
    "LC25",
    "LC05",
    "IC10",
    "IC20",
    "IC25",
)
THRESHOLD_ENDPOINTS = ("NOEC", "LOEC", "NOEL", "LOEL", "MATC")
ENDPOINTS = MEDIAN_ENDPOINTS + LOW_EFFECT_ENDPOINTS + THRESHOLD_ENDPOINTS
MISSING_CODES = {"NR", "NC", "--", "NA", "", "/"}

# ----------------------------------------------------------------------------- units

MASS_MG = {"kg": 1e6, "g": 1e3, "mg": 1.0, "ug": 1e-3, "ng": 1e-6, "pg": 1e-9, "fg": 1e-12}
MOL = {"mol": 1.0, "mmol": 1e-3, "umol": 1e-6, "nmol": 1e-9, "pmol": 1e-12, "fmol": 1e-15}
MOLAR = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12, "fM": 1e-15}
VOLUME_L = {
    "L": 1.0,
    "l": 1.0,
    "ml": 1e-3,
    "mL": 1e-3,
    "ul": 1e-6,
    "uL": 1e-6,
    "dm3": 1.0,
    "cm3": 1e-3,
    "cc": 1e-3,
    "m3": 1e3,
    "dl": 0.1,
    "dL": 0.1,
    "nl": 1e-9,
}
PARTS = {"ppm": ("mg/L", 1.0), "ppb": ("mg/L", 1e-3), "pptr": ("mg/L", 1e-6)}
BASIS_PREFIXES = {
    "AI": "active_ingredient",
    "ai": "active_ingredient",
    "ae": "acid_equivalent",
    "AE": "acid_equivalent",
    "ACT": "active_ingredient",
}
_CORE_RE = re.compile(r"^(?P<num>[A-Za-z]+)/(?P<den>[A-Za-z]+\d?)$")


def parse_conc_unit(unit: str | None) -> dict:
    """Unit string -> {basis, core, kind ('mass' | 'molar' | None), to_mg_L | to_mol_L factor}."""
    out = {
        "conc_basis": None,
        "conc_unit_core": None,
        "conc_unit_kind": None,
        "factor_mg_L": np.nan,
        "factor_mol_L": np.nan,
    }
    if unit is None or (not isinstance(unit, str) and pd.isna(unit)):
        return out
    u = " ".join(str(unit).split())
    toks = u.split(" ")
    if len(toks) >= 2 and toks[0] in BASIS_PREFIXES:
        out["conc_basis"] = BASIS_PREFIXES[toks[0]]
        toks = toks[1:]
    if len(toks) != 1:  # trailing qualifiers (e.g. "mg/L CaCO3", "ug/L N") change the meaning
        out["conc_unit_core"] = " ".join(toks)
        return out
    core = toks[0]
    out["conc_unit_core"] = core
    if core in PARTS:
        _, f = PARTS[core]
        out.update({"conc_unit_kind": "mass", "factor_mg_L": f})
        return out
    if core in MOLAR:
        out.update({"conc_unit_kind": "molar", "factor_mol_L": MOLAR[core]})
        return out
    m = _CORE_RE.match(core)
    if not m:
        return out
    num, den = m.group("num"), m.group("den")
    if den not in VOLUME_L:
        return out
    if num in MASS_MG:
        out.update({"conc_unit_kind": "mass", "factor_mg_L": MASS_MG[num] / VOLUME_L[den]})
    elif num in MOL:
        out.update({"conc_unit_kind": "molar", "factor_mol_L": MOL[num] / VOLUME_L[den]})
    return out


#: ECOTOX media_type codes -> water type (unlisted codes such as SED, AGR, MIX -> 'other')
WATER_TYPE = {
    "FW": "freshwater",
    "SW": "saltwater",
    "CUL": "culture",
    "AQU": "aqueous",
    "SED": "sediment",
    "MIX": "other",
    "AGR": "other",
    "HYP": "other",
    "FLT": "other",
    "UKN": None,
    "NR": None,
    "NC": None,
    "--": None,
    "NONE": None,
}

DURATION_H = {
    "mi": 1.0 / 60.0,
    "h": 1.0,
    "d": 24.0,
    "wk": 168.0,
    "mo": 730.5,
    "yr": 8766.0,
    "s": 1.0 / 3600.0,
}


def duration_hours(value, unit) -> pd.Series:
    v = pd.to_numeric(pd.Series(value, dtype="string").str.replace("*", "", regex=False), errors="coerce")
    f = pd.Series(unit, dtype="string").map(DURATION_H).astype("float64")
    return (v * f).astype("float64")


def parse_value(s) -> tuple[pd.Series, pd.Series]:
    """ECOTOX numeric text -> (float, flag) with flag in {'', 'asterisk', 'missing_code', 'non_numeric'}."""
    t = pd.Series(s, dtype="string").str.strip()
    star = t.str.endswith("*").fillna(False)
    core = t.str.rstrip("*").str.strip()
    val = pd.to_numeric(core, errors="coerce")
    flag = pd.Series("", index=t.index, dtype="object")
    flag[star.to_numpy()] = "asterisk"
    miss = (t.isna() | t.isin(list(MISSING_CODES))).to_numpy()
    flag[miss] = "missing_code"
    flag[(~miss) & val.isna().to_numpy()] = "non_numeric"
    return val.astype("float64"), flag.astype("string")


CENSOR = {
    "<": "left",
    "<=": "left",
    "<~": "left",
    "~<": "left",
    ">": "right",
    ">=": "right",
    ">~": "right",
    "~>": "right",
    "~": "approximate",
    "ca": "approximate",
    "ca.": "approximate",
    "=": "",
    "": "",
}


def censoring(op) -> tuple[pd.Series, pd.Series]:
    o = pd.Series(op, dtype="string").str.strip().fillna("")
    kind = o.map(lambda x: CENSOR.get(x, "other")).astype("string")
    censored = kind.isin(["left", "right"])
    return censored.astype(bool), kind


# ----------------------------------------------------------------------------- taxa / endpoints


def taxon_group(species: pd.DataFrame) -> pd.Series:
    """Map ECOTOX species rows to the six SSD groups (else 'other')."""
    g = species["ecotox_group"].astype("string").fillna("")
    cls = species["class"].astype("string").fillna("")
    out = pd.Series("other", index=species.index, dtype="object")
    out[g.str.contains("Fish", regex=False).to_numpy()] = "fish"
    out[g.str.contains("Crustaceans", regex=False).to_numpy()] = "crustaceans"
    out[g.str.contains("Algae", regex=False).to_numpy()] = "algae"
    out[(g.str.contains("Insects", regex=False) & cls.eq("Insecta")).to_numpy()] = "insects"
    out[g.str.contains("Molluscs", regex=False).to_numpy()] = "molluscs"
    out[g.str.contains("Amphibians", regex=False).to_numpy()] = "amphibians"
    return out.astype("string")


def code_norm(v) -> pd.Series:
    """Strip ECOTOX comment/approximation markers from a code: 'FW/' -> 'FW', '~MOR' -> 'MOR'."""
    return pd.Series(v, dtype="string").str.strip().str.strip("/~*").str.strip()


def normalize_endpoint(ep) -> pd.Series:
    """'LC50*' / 'LC50/' / ' LC50 ' -> 'LC50' (the raw value is kept separately)."""
    return pd.Series(ep, dtype="string").str.strip().str.rstrip("*/").str.strip().str.upper()


def endpoint_group(ep: pd.Series) -> pd.Series:
    e = pd.Series(ep, dtype="string")
    out = pd.Series(pd.NA, index=e.index, dtype="string")
    out[e.isin(MEDIAN_ENDPOINTS).fillna(False).to_numpy()] = "median_effect"
    out[e.isin(LOW_EFFECT_ENDPOINTS).fillna(False).to_numpy()] = "low_effect"
    out[e.isin(THRESHOLD_ENDPOINTS).fillna(False).to_numpy()] = "threshold"
    return out


def exposure_class(group: pd.Series, hours: pd.Series) -> pd.Series:
    out = pd.Series("unknown", index=group.index, dtype="object")
    med = group.eq("median_effect").fillna(False).to_numpy()
    h = hours.to_numpy(dtype=float)
    out[med & (h <= 96)] = "acute"
    out[med & (h > 96)] = "prolonged"
    out[group.isin(["low_effect", "threshold"]).fillna(False).to_numpy()] = "chronic"
    return out.astype("string")


def element_mass(symbol: str | None) -> float | None:
    """Standard atomic weight for a bare element symbol ('Cu', 'Cd'); None otherwise."""
    if symbol is None or (not isinstance(symbol, str) and pd.isna(symbol)):
        return None
    s = str(symbol).strip()
    if not re.fullmatch(r"[A-Z][a-z]?", s):
        return None
    from rdkit import Chem

    try:
        return float(Chem.GetPeriodicTable().GetAtomicWeight(s))
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------- builder


def build_aquatic(
    tests: pd.DataFrame,
    results: pd.DataFrame,
    species: pd.DataFrame,
    chemicals: pd.DataFrame,
    references: pd.DataFrame | None = None,
    media: pd.DataFrame | None = None,
    chem_ids: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join + filter + standardise. All inputs are the all-string raw tables."""
    t = tests[tests["organism_habitat"].eq("Water").fillna(False).to_numpy(dtype=bool)]
    sp = species.copy()
    sp["taxon_group"] = taxon_group(sp)
    sp = sp[sp["taxon_group"].isin(TAXON_GROUPS)]
    t = t.merge(
        sp[
            [
                "species_number",
                "taxon_group",
                "latin_name",
                "common_name",
                "kingdom",
                "phylum_division",
                "class",
                "tax_order",
                "family",
                "genus",
                "species",
                "ecotox_group",
                "ncbi_taxid",
            ]
        ],
        on="species_number",
        how="inner",
    )
    r = results.copy()
    r["endpoint_norm"] = normalize_endpoint(r["endpoint"])
    r = r[r["endpoint_norm"].isin(ENDPOINTS).fillna(False)]
    keep_r = [
        "result_id",
        "test_id",
        "endpoint",
        "endpoint_norm",
        "effect",
        "measurement",
        "response_site",
        "obs_duration_mean_op",
        "obs_duration_mean",
        "obs_duration_unit",
        "conc1_type",
        "ion1",
        "conc1_mean_op",
        "conc1_mean",
        "conc1_min_op",
        "conc1_min",
        "conc1_max_op",
        "conc1_max",
        "conc1_unit",
        "significance_code",
        "chem_analysis_method",
        "endpoint_assigned",
    ]
    r = r[[c for c in keep_r if c in r.columns]]
    keep_t = [
        "test_id",
        "reference_number",
        "test_cas",
        "species_number",
        "organism_lifestage",
        "exposure_type",
        "media_type",
        "test_location",
        "test_type",
        "study_type",
        "test_method",
        "control_type",
        "exposure_duration_mean_op",
        "exposure_duration_mean",
        "exposure_duration_unit",
        "taxon_group",
        "latin_name",
        "common_name",
        "kingdom",
        "phylum_division",
        "class",
        "tax_order",
        "family",
        "genus",
        "species",
        "ecotox_group",
        "ncbi_taxid",
    ]
    df = r.merge(t[[c for c in keep_t if c in t.columns]], on="test_id", how="inner")
    df["cas"] = df["test_cas"].map(normalize_cas).astype("string")
    ch = chemicals.copy()
    ch["cas"] = ch["cas_number"].map(normalize_cas)
    ch = ch.dropna(subset=["cas"]).drop_duplicates("cas")
    df = df.merge(
        ch[["cas", "chemical_name", "dtxsid", "ecotox_group"]].rename(
            columns={"dtxsid": "ecotox_dtxsid", "ecotox_group": "ecotox_chem_group"}
        ),
        on="cas",
        how="left",
    )
    # values, censoring, units
    df["conc_value"], df["conc_value_flag"] = parse_value(df["conc1_mean"])
    df["conc_min"], _ = parse_value(df["conc1_min"])
    df["conc_max"], _ = parse_value(df["conc1_max"])
    df["censored"], df["censor_kind"] = censoring(df["conc1_mean_op"])
    utab = {u: parse_conc_unit(u) for u in df["conc1_unit"].dropna().unique()}
    for k in ("conc_basis", "conc_unit_core", "conc_unit_kind", "factor_mg_L", "factor_mol_L"):
        col = df["conc1_unit"].map(lambda u, k=k: utab[u][k] if isinstance(u, str) and u in utab else None)
        df[k] = col.astype("float64") if k.startswith("factor") else col.astype("string")
    # molecular weight of the substance as expressed
    itab = {i: element_mass(i) for i in df["ion1"].dropna().unique()}
    ion_mass = df["ion1"].map(lambda i: itab.get(i) if isinstance(i, str) else None).astype("float64")
    mw = pd.Series(np.nan, index=df.index)
    mw_source = pd.Series(pd.NA, index=df.index, dtype="string")
    has_ion = ion_mass.notna().to_numpy()
    mw[has_ion] = ion_mass[has_ion]
    mw_source[has_ion] = "ion_atomic_weight"
    if chem_ids is not None and len(chem_ids):
        cm = chem_ids.drop_duplicates("cas").set_index("cas")
        cmw = df["cas"].map(cm["mw_g_mol"]).astype("float64")
        ion_codes = df["ion1"].astype("string").fillna("NR")
        as_substance = ion_codes.isin(list(MISSING_CODES)).to_numpy()
        use = (~has_ion) & as_substance & cmw.notna().to_numpy()
        mw[use] = cmw[use]
        mw_source[use] = "chemical_mw"
        for col, src in (
            ("dtxsid_final", "chem_dtxsid"),
            ("smiles_canonical", "chem_smiles"),
            ("inchikey", "chem_inchikey"),
            ("structure_class", "chem_structure_class"),
        ):
            df[src] = df["cas"].map(cm[col]).astype("string")
    df["mw_used_g_mol"] = mw
    df["mw_source"] = mw_source
    val, vmin, vmax = df["conc_value"], df["conc_min"], df["conc_max"]
    is_mass = df["conc_unit_kind"].eq("mass").fillna(False).to_numpy()
    is_mol = df["conc_unit_kind"].eq("molar").fillna(False).to_numpy()
    fm, fmol = df["factor_mg_L"].to_numpy(dtype=float), df["factor_mol_L"].to_numpy(dtype=float)
    mwv = mw.to_numpy(dtype=float)
    for name, v in (("", val), ("_min", vmin), ("_max", vmax)):
        v = v.to_numpy(dtype=float)
        mg = np.where(is_mass, v * fm, np.where(is_mol, v * fmol * mwv * 1e3, np.nan))
        mol = np.where(is_mol, v * fmol, np.where(is_mass, v * fm / 1e3 / mwv, np.nan))
        df[f"conc{name}_mg_L"] = mg
        df[f"conc{name}_mol_L"] = mol
    conv = np.where(
        ~(is_mass | is_mol),
        "not_convertible",
        np.where(np.isnan(mwv), np.where(is_mass, "mass_only", "molar_only"), "mass_and_molar"),
    )
    df["conc_conversion"] = pd.Series(conv, index=df.index, dtype="string")
    df["log10_conc_mol_L"] = np.log10(df["conc_mol_L"].where(df["conc_mol_L"] > 0))
    df["log10_conc_mg_L"] = np.log10(df["conc_mg_L"].where(df["conc_mg_L"] > 0))
    # durations
    df["obs_duration_value"], df["obs_duration_flag"] = parse_value(df["obs_duration_mean"])
    df["duration_h"] = duration_hours(df["obs_duration_mean"], df["obs_duration_unit"])
    df["exposure_duration_h"] = duration_hours(df["exposure_duration_mean"], df["exposure_duration_unit"])
    df["endpoint_group"] = endpoint_group(df["endpoint_norm"])
    df["exposure_class"] = exposure_class(df["endpoint_group"], df["duration_h"])
    # ECOTOX appends '/' to a code when a comment exists ('FW/'), '~' marks approximate effects
    for c in (
        "media_type",
        "exposure_type",
        "test_location",
        "organism_lifestage",
        "effect",
        "measurement",
        "conc1_type",
    ):
        if c in df.columns:
            df[c + "_norm"] = code_norm(df[c])
    df["endpoint_flag"] = pd.Series(
        np.where(
            df["endpoint"].astype("string").str.contains("*", regex=False).fillna(False), "asterisk", ""
        ),
        index=df.index,
        dtype="string",
    )
    df["water_type"] = df["media_type_norm"].map(WATER_TYPE).astype("string")
    if references is not None:
        ref = references[["reference_number", "publication_year", "doi", "reference_type"]].drop_duplicates(
            "reference_number"
        )
        df = df.merge(ref.rename(columns={"doi": "reference_doi"}), on="reference_number", how="left")
    if media is not None:
        mcols = [
            c
            for c in (
                "result_id",
                "media_ph_mean",
                "media_temperature_mean",
                "media_temperature_unit",
                "media_hardness_mean",
                "media_hardness_unit",
                "dissolved_oxygen_mean",
                "dissolved_oxygen_unit",
                "media_salinity_mean",
                "media_salinity_unit",
            )
            if c in media.columns
        ]
        df = df.merge(media[mcols], on="result_id", how="left")
        for c in (
            "media_ph_mean",
            "media_temperature_mean",
            "media_hardness_mean",
            "dissolved_oxygen_mean",
            "media_salinity_mean",
        ):
            if c in df.columns:
                df[c + "_value"], _ = parse_value(df[c])
    df = df.drop(columns=["factor_mg_L", "factor_mol_L"])
    return df.sort_values(["cas", "taxon_group", "latin_name", "result_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- QC against EPA


def crosscheck_epa_standardised(tidy: pd.DataFrame, pull: pd.DataFrame, *, rtol: float = 1e-3) -> dict:
    """Compare SENTINEL mg/L and hours with EPA's own standardised values for the same result ids.

    ``pull`` = subset of EPA's aquatic data pull (RESULT_NUMBER, CONC1_MEAN_STD, CONC1_UNITS_STD,
    TEST_DURATION_STD, DURATION_UNITS_STD). Only results present in both are compared; the pull can
    be an older release, so a few differences can be genuine data revisions.
    """
    p = pull.rename(columns={"RESULT_NUMBER": "result_id"})
    m = tidy.merge(p, on="result_id", how="inner")
    out: dict = {"n_matched_results": int(len(m))}
    std_unit = m["CONC1_UNITS_STD"].astype("string").fillna("")
    core = std_unit.str.replace(r"^(AI|ae|AE)\s+", "", regex=True)
    is_mgl = core.eq("mg/L").to_numpy()
    std_val = pd.to_numeric(
        m["CONC1_MEAN_STD"].astype("string").str.replace(r"[*/,]", "", regex=True), errors="coerce"
    ).to_numpy(dtype=float)
    ours = m["conc_mg_L"].to_numpy(dtype=float)
    both = is_mgl & np.isfinite(std_val) & np.isfinite(ours) & (np.abs(std_val) > 0)
    rel = np.abs(ours - std_val) / np.abs(std_val)
    match = both & (rel <= rtol)
    mism = both & (rel > rtol)
    molar_src = m["conc_unit_kind"].eq("molar").fillna(False).to_numpy()
    ion_src = m["mw_source"].eq("ion_atomic_weight").fillna(False).to_numpy()
    out["concentration"] = {
        "epa_mg_L_rows": int(is_mgl.sum()),
        "compared": int(both.sum()),
        "match": int(match.sum()),
        "match_frac": float(match.sum() / both.sum()) if both.sum() else float("nan"),
        "mismatch": int(mism.sum()),
        "mismatch_molar_with_ion_basis": int((mism & molar_src & ion_src).sum()),
        "mismatch_molar_other_mw": int((mism & molar_src & ~ion_src).sum()),
        "mismatch_mass_units": int((mism & ~molar_src).sum()),
        "mismatch_mass_units_examples": m.loc[
            mism & ~molar_src, ["result_id", "conc1_mean", "conc1_unit", "CONC1_MEAN_STD", "CONC1_UNITS_STD"]
        ]
        .head(10)
        .astype(str)
        .to_dict("records"),
        "epa_value_ours_missing": {
            str(k): int(v)
            for k, v in m.loc[is_mgl & np.isfinite(std_val) & ~np.isfinite(ours), "conc_conversion"]
            .value_counts(dropna=False)
            .items()
        },
    }
    du = m["DURATION_UNITS_STD"].astype("string").fillna("")
    dv = pd.to_numeric(m["TEST_DURATION_STD"], errors="coerce").to_numpy(dtype=float)
    fac = du.map({"h": 1.0, "d": 24.0}).astype("float64").to_numpy()
    epa_h = dv * fac
    oh = m["duration_h"].to_numpy(dtype=float)
    okd = np.isfinite(epa_h) & np.isfinite(oh)
    out["duration"] = {
        "compared": int(okd.sum()),
        "match": int((okd & (np.abs(epa_h - oh) <= 1e-6 + 1e-3 * np.abs(oh))).sum()),
        "epa_units": {str(k): int(v) for k, v in du.value_counts().head(8).items()},
    }
    out["duration"]["match_frac"] = (
        out["duration"]["match"] / out["duration"]["compared"]
        if out["duration"]["compared"]
        else float("nan")
    )
    return out
