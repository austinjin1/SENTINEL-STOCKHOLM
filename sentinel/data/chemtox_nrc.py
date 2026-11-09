# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""NRC incident dataset builder: reports x materials flat table, chemical join and coverage stats.

The flat table has one row per (report, material); reports without a MATERIAL_INVOLVED row are
kept once with empty material fields (``material_seq`` = -1). Chemical identities come from the
``chemicals`` dataset (CAS -> DTXSID / SMILES / structure class); unknown CAS values are kept.
"""

from __future__ import annotations

import pandas as pd

CHEM_JOIN_COLS = [
    "cas",
    "dtxsid_final",
    "name",
    "smiles_canonical",
    "inchikey",
    "mw_g_mol",
    "structure_class",
]
CHEM_RENAME = {
    "dtxsid_final": "chem_dtxsid",
    "name": "chem_name",
    "smiles_canonical": "chem_smiles",
    "inchikey": "chem_inchikey",
    "mw_g_mol": "chem_mw_g_mol",
    "structure_class": "chem_structure_class",
}


def flat_table(
    reports: pd.DataFrame, materials: pd.DataFrame, chem: pd.DataFrame | None = None
) -> pd.DataFrame:
    m = materials.drop(columns=["source_file"], errors="ignore")
    flat = reports.merge(m, on="report_id", how="left", validate="one_to_many")
    flat["material_seq"] = flat["material_seq"].fillna(-1).astype("int32")
    if chem is not None and len(chem):
        c = chem[[x for x in CHEM_JOIN_COLS if x in chem.columns]].drop_duplicates("cas")
        flat = flat.merge(c.rename(columns=CHEM_RENAME), on="cas", how="left", validate="many_to_one")
    return flat


def report_level_flags(
    reports: pd.DataFrame, materials: pd.DataFrame, chem: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per report: any material reached water, any valid CAS, any CAS with a resolved structure."""
    m = materials[["report_id", "material_seq", "cas", "if_reached_water"]].copy()
    m["_reached"] = m["if_reached_water"].fillna(False).astype(bool)
    m["_cas"] = m["cas"].notna()
    if chem is not None and len(chem):
        ok = set(chem.loc[chem["smiles_canonical"].notna(), "cas"])
        m["_struct"] = m["cas"].isin(ok)
    else:
        m["_struct"] = False
    g = (
        m.groupby("report_id")
        .agg(
            any_material_reached_water=("_reached", "any"),
            any_valid_cas=("_cas", "any"),
            any_cas_with_structure=("_struct", "any"),
            n_materials=("material_seq", "size"),
        )
        .reset_index()
    )
    out = reports[["report_id"]].merge(g, on="report_id", how="left")
    for c in ("any_material_reached_water", "any_valid_cas", "any_cas_with_structure"):
        out[c] = out[c].astype("boolean").fillna(False).astype(bool)
    out["n_materials"] = out["n_materials"].fillna(0).astype("int32")
    return out


def coverage_stats(r: pd.DataFrame) -> dict:
    """Fractions of water-reaching reports with coordinates and with a CAS number.

    ``r`` = reports merged with :func:`report_level_flags`. Two definitions of "water-reaching":
    ``medium_desc == 'WATER'`` (incident level) and ``IF_REACHED_WATER == YES`` for at least one
    material (material level).
    """
    out = {"n_reports": int(len(r))}
    for name, mask in [
        ("medium_water", r["medium_desc"].fillna("").str.upper().eq("WATER")),
        ("material_reached_water", r["any_material_reached_water"].fillna(False)),
    ]:
        sub = r[mask.to_numpy(dtype=bool)]
        n = int(len(sub))
        cas = sub["any_valid_cas"].astype(bool)
        xy = sub["has_coordinates"].astype(bool)
        given = sub["coordinates_given"].astype(bool) if "coordinates_given" in sub.columns else xy
        st = sub["any_cas_with_structure"].astype(bool)
        out[name] = {
            "n": n,
            "with_coordinates_given": int(given.sum()),
            "frac_with_coordinates_given": float(given.mean()) if n else float("nan"),
            "with_coordinates": int(xy.sum()),
            "frac_with_coordinates": float(xy.mean()) if n else float("nan"),
            "with_cas": int(cas.sum()),
            "frac_with_cas": float(cas.mean()) if n else float("nan"),
            "with_cas_and_coordinates": int((cas & xy).sum()),
            "with_structure": int(st.sum()),
            "frac_with_structure": float(st.mean()) if n else float("nan"),
        }
    return out


def coverage_by_file_year(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, sub in r.groupby("file_year"):
        st = coverage_stats(sub)
        rows.append(
            {
                "file_year": int(y),
                "n_reports": st["n_reports"],
                **{
                    f"{k}_{kk}": vv
                    for k in ("medium_water", "material_reached_water")
                    for kk, vv in st[k].items()
                },
            }
        )
    return pd.DataFrame(rows)
