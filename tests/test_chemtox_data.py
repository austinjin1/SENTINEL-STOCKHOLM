# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for the ChemTox data foundation (ADORE, ECOTOX, chemical identities, NRC).

No network access. Fixtures under tests/fixtures/{adore,ecotox,nrc}/ are small excerpts of the
real releases (ADORE t-F2F rows; ECOTOX 2026-09-15 rows; NRC CY90-CY24 rows; recorded CompTox /
PubChem responses). Synthetic frames appear only where a rule needs an explicit edge case.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data import chemtox_baselines as B
from sentinel.data import chemtox_ecotox as CE
from sentinel.data import chemtox_nrc as CN
from sentinel.data.sources import adore as A
from sentinel.data.sources import chem_ids as C
from sentinel.data.sources import ecotox as E
from sentinel.data.sources import nrc as N

FIX = Path(__file__).parent / "fixtures"


# =========================================================================== CAS numbers


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("50-00-0", "50-00-0"),
        ("000050-00-0", "50-00-0"),
        ("50000", "50-00-0"),  # ECOTOX digit form
        ("7732-18-5", "7732-18-5"),
        ("000074-90-8", "74-90-8"),
        ("68476-34-6", "68476-34-6"),
        ("000000-00-0", None),  # NRC placeholder
        ("7732-18-4", None),  # wrong check digit
        ("10699-0", None),
        ("D007", None),
        ("", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_normalize_cas(raw, expected):
    assert C.normalize_cas(raw) == expected


def test_cas_check_digit():
    assert C.cas_check_digit_ok("7440-50-8")  # copper
    assert not C.cas_check_digit_ok("7440-50-9")


# =========================================================================== chemical identities


def _json(name):
    return json.loads((FIX / "ecotox" / name).read_text())


def test_parse_comptox_single_compound():
    rec = C.parse_comptox_hits(_json("chemid_comptox_50-00-0.json"), "50-00-0")
    assert rec["dtxsid"] == "DTXSID7020637" and rec["comptox_smiles"] == "C=O"
    assert rec["comptox_match"] == "CASRN"


def test_uvcb_without_structure_is_no_structure():
    ct = C.parse_comptox_hits(_json("chemid_comptox_68476-34-6.json"), "68476-34-6")
    row = C.resolve_identity("68476-34-6", ct, None, {})
    assert row["dtxsid_final"] == "DTXSID5028530"
    assert row["smiles_canonical"] is None and row["structure_class"] == "no_structure"


def test_no_structure_hints():
    assert C.no_structure_hint("Diesel Fuel No. 2") == "petroleum_or_uvcb"
    assert C.no_structure_hint("Polyvinyl chloride") == "polymer"
    assert C.no_structure_hint("Aroclor 1254") == "mixture_or_uvcb"
    assert C.no_structure_hint("Lead arsenate (n:1)") == "variable_stoichiometry"
    assert C.no_structure_hint("Dichlorobenzene", "Cl*.ClC1=CC=CC=C1 |m:1:6.7.8|") == "isomer_or_markush"
    assert C.no_structure_hint("Polyvinyl alcohol", "OC(-*)C-* |Sg:n:0,1,3::ht|") == "polymer"
    row = C.resolve_identity(
        "25321-22-6",
        {
            "dtxsid": "DTXSID1",
            "comptox_smiles": "Cl*.ClC1=CC=CC=C1 |m:1:6.7.8|",
            "comptox_name": "Dichlorobenzene",
        },
        None,
        {},
    )
    assert row["structure_class"] == "no_structure" and row["smiles_canonical"] is None
    assert C.no_structure_hint("Asbestos") == "unspecified" and C.no_structure_hint(None) is None
    ct = C.parse_comptox_hits(_json("chemid_comptox_68476-34-6.json"), "68476-34-6")
    assert C.resolve_identity("68476-34-6", ct, None, {})["no_structure_hint"] == "petroleum_or_uvcb"


def test_salt_is_inorganic_multicomponent_flagged():
    ct = C.parse_comptox_hits(_json("chemid_comptox_7758-98-7.json"), "7758-98-7")
    row = C.resolve_identity("7758-98-7", ct, None, {})
    assert row["structure_class"] == "inorganic" and row["n_components"] == 2
    assert row["mw_g_mol"] == pytest.approx(159.6, abs=0.1)


def test_parse_pubchem_and_unresolved():
    pc = C.parse_pubchem(_json("chemid_pubchem_50-00-0.json"))
    assert pc["pubchem_cid"] == "712" and pc["pubchem_inchikey"] == "WSFSSNUMVMOOMR-UHFFFAOYSA-N"
    empty = C.parse_comptox_hits([], "99999-99-9")
    row = C.resolve_identity("99999-99-9", empty, C.parse_pubchem(None), {})
    assert row["structure_class"] == "unresolved" and row["dtxsid_final"] is None


@pytest.mark.parametrize(
    "smiles,cls",
    [
        ("CCO", "organic"),
        ("[Na+].[Cl-]", "inorganic"),
        ("O=C([O-])[O-].[Ca+2]", "inorganic"),
        ("CC(=O)[O-].[Na+]", "organic_multicomponent"),
        ("C[Hg]Cl", "organometallic"),
        ("[C-]#N.[K+]", "inorganic"),
    ],
)
def test_structure_classes(smiles, cls):
    assert C.rdkit_identity(smiles)["structure_class"] == cls


def test_rdkit_identity_recomputes_inchikey_and_mw():
    r = C.rdkit_identity("OCC")
    assert r["smiles_canonical"] == "CCO" and r["inchikey_rdkit"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert r["mw_rdkit"] == pytest.approx(46.069, abs=1e-3)


def test_jsonl_cache_roundtrip(tmp_path):
    c = C.JsonlCache(tmp_path / "c.jsonl")
    c.put("50-00-0", 200, [{"dtxsid": "X"}], "2026-10-22T00:00:00+00:00")
    c2 = C.JsonlCache(tmp_path / "c.jsonl")
    assert "50-00-0" in c2 and c2.get("50-00-0")["payload"][0]["dtxsid"] == "X"


# =========================================================================== ADORE


@pytest.fixture(scope="module")
def adore_excerpt() -> pd.DataFrame:
    raw = (FIX / "adore" / "t-F2F_excerpt.csv").read_bytes()
    df = A.read_member_table(raw, "processed/t-F2F_mortality.csv")
    df.attrs["challenge"] = "t-F2F"
    return df


def test_adore_conversion_types(adore_excerpt):
    df = adore_excerpt
    assert pd.api.types.is_string_dtype(df["test_cas"]) or df["test_cas"].dtype == object
    assert df["split_occurrence"].isin(["0", "1", "2", "3", "4", "test"]).all()
    assert df["chem_MACCS_fp"].str.fullmatch(r"[01]+").all()  # bit string kept verbatim
    assert df["result_id"].map(type).eq(str).all()  # IDs stay strings
    assert pd.api.types.is_float_dtype(df[A.TARGET])
    assert df["tax_pdm_available"].dtype == "boolean"


def test_adore_member_roundtrip_is_lossless(tmp_path):
    raw = (FIX / "adore" / "t-F2F_excerpt.csv").read_bytes()
    zp = tmp_path / "adore_dataset.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("processed/t-F2F_mortality.csv", raw)
        z.writestr("chemicals/pubchem_bits.csv", "bit\tdescription\n0\t>= 4 H\n1\t>= 8 H\n")
        z.writestr("README.md", "readme")
    members = A.convert_release_zip(zp, tmp_path / "release")
    checks = A.verify_release(zp, members)
    assert all(c["mismatches"] == 0 and c["shape_ok"] for c in checks)
    bits = pd.read_parquet(tmp_path / "release" / "chemicals" / "pubchem_bits.parquet")
    assert list(bits.columns) == ["bit", "description"]  # tab sniffed despite .csv
    df = A.load_challenge("t-F2F", root=tmp_path / "release")
    assert len(df) == 37 and "chem_MACCS_fp" not in df.columns  # core columns only


def test_adore_official_splits_are_not_resplit(adore_excerpt):
    df = adore_excerpt
    tr, te = A.train_test(df, A.PRIMARY_SPLIT)
    assert len(tr) + len(te) == len(df) and (te[A.PRIMARY_SPLIT] == "test").all()
    ov = A.chemical_overlap(df, A.PRIMARY_SPLIT)
    assert ov["chem_overlap"] == 0 and ov["n_test"] == 10 and ov["n_chem_test"] == 2
    folds = list(A.cv_folds(df, A.PRIMARY_SPLIT))
    assert [k for k, _, _ in folds] == [0, 1, 2, 3, 4]
    for _, trn, val in folds:
        assert not set(trn) & set(val)
        assert set(df.iloc[trn]["test_cas"]).isdisjoint(df.iloc[val]["test_cas"])
    sa = A.split_assignments(df)
    occ = sa[sa["split"] == A.PRIMARY_SPLIT]
    assert len(occ) == len(df) and occ["fold"].isna().sum() == 10 and set(occ["role"]) == {"train", "test"}


def test_trainvalid_rows_count_as_training():
    df = pd.DataFrame(
        {
            "result_id": ["1", "2", "3"],
            "test_cas": ["a", "b", "c"],
            "tax_gs": ["x"] * 3,
            "split_scaffold-murcko-llo": ["trainvalid", "test", "trainvalid"],
        }
    )
    tr, te = A.train_test(df, "split_scaffold-murcko-llo")
    assert len(tr) == 2 and len(te) == 1
    assert list(A.cv_folds(df, "split_scaffold-murcko-llo")) == []


def test_feature_groups():
    g = A.feature_groups(
        [
            "test_cas",
            "chem_MACCS_fp",
            "chem_mol2vec0",
            "chem_mordred_ABC",
            "split_random",
            "result_conc1_mean_mol_log",
        ]
    )
    assert g["core"] == ["test_cas", "result_conc1_mean_mol_log"] and g["split"] == ["split_random"]
    reps = [c for k, v in g.items() if k not in ("core", "split") for c in v]
    assert (
        "chem_MACCS_fp" in reps
        and g["fp_MACCS"] == ["chem_MACCS_fp"]
        and g["mordred"] == ["chem_mordred_ABC"]
    )


def test_target_is_log10_molar(adore_excerpt):
    df = adore_excerpt
    assert np.allclose(df[A.TARGET], np.log10(df["result_conc1_mean_mol"]), atol=1e-9)


# --------------------------------------------------------------------------- baselines


def test_trivial_baselines_fallbacks():
    tr = pd.DataFrame(
        {
            "tax_gs": ["a", "a", "b", "b"],
            "result_obs_duration_mean": [96.0, 48.0, 96.0, 96.0],
            "y": [1.0, 3.0, 5.0, 7.0],
        }
    )
    te = pd.DataFrame(
        {"tax_gs": ["a", "b", "c"], "result_obs_duration_mean": [96.0, 48.0, 96.0], "y": [0.0, 0.0, 0.0]}
    )
    f = B.fit_predict(tr, te, "y")
    assert np.allclose(f["preds"]["global_mean"], 4.0)
    assert np.allclose(f["preds"]["species_mean"], [2.0, 6.0, 4.0])  # c unseen -> global
    assert np.allclose(f["preds"]["species_duration_mean"], [1.0, 6.0, 4.0])  # (b,48) unseen -> species
    assert f["fallbacks"] == {
        "species_mean_to_global": 1,
        "species_duration_to_species": 1,
        "species_duration_to_global": 1,
    }


def test_evaluate_split_uses_training_rows_only(adore_excerpt):
    res = B.evaluate_split(adore_excerpt, A.PRIMARY_SPLIT, A.TARGET)
    df = adore_excerpt
    tr = df[df[A.PRIMARY_SPLIT] != "test"]
    te = df[df[A.PRIMARY_SPLIT] == "test"]
    gm = tr[A.TARGET].mean()
    expect = math.sqrt(((te[A.TARGET] - gm) ** 2).mean())
    assert res["test"]["global_mean"]["rmse"] == pytest.approx(expect)
    assert res["n_train"] == 27 and res["n_test"] == 10 and res["cv_folds"] == [0, 1, 2, 3, 4]
    assert len(res["cv"]["species_mean"]["rmse_per_fold"]) == 5


def test_replicate_noise_floor_pooling():
    df = pd.DataFrame(
        {
            "test_cas": ["a", "a", "a", "b", "b", "c"],
            "tax_gs": ["s"] * 6,
            "result_obs_duration_mean": [96.0] * 6,
            "result_conc1_type": ["A"] * 6,
            "test_media_type": ["FW"] * 6,
            "test_exposure_type": [None] * 6,
            "y": [0.0, 1.0, 2.0, 5.0, 7.0, 3.0],
        }
    )
    nf = B.replicate_noise_floor(df, "y")
    # SS = (1+0+1) + (1+1) = 4 ; dof = 2 + 1 = 3  (missing exposure type groups together)
    assert nf["pooled_sd"] == pytest.approx(math.sqrt(4 / 3)) and nf["dof"] == 3
    assert nf["n_keys"] == 3 and nf["n_keys_with_replicates"] == 2


# =========================================================================== ECOTOX


@pytest.mark.parametrize(
    "unit,kind,mg,mol,basis",
    [
        ("mg/L", "mass", 1.0, None, None),
        ("ug/L", "mass", 1e-3, None, None),
        ("ng/L", "mass", 1e-6, None, None),
        ("g/L", "mass", 1e3, None, None),
        ("ug/ml", "mass", 1.0, None, None),
        ("mg/dm3", "mass", 1.0, None, None),
        ("ppm", "mass", 1.0, None, None),
        ("ppb", "mass", 1e-3, None, None),
        ("AI mg/L", "mass", 1.0, None, "active_ingredient"),
        ("ae ug/L", "mass", 1e-3, None, "acid_equivalent"),
        ("uM", "molar", None, 1e-6, None),
        ("mM", "molar", None, 1e-3, None),
        ("umol/L", "molar", None, 1e-6, None),
        ("nmol/ml", "molar", None, 1e-6, None),
        ("ul/L", None, None, None, None),
        ("%", None, None, None, None),
        ("mg/kg", None, None, None, None),
        ("ppt", None, None, None, None),
        ("mg/L CaCO3", None, None, None, None),
    ],
)
def test_parse_conc_unit(unit, kind, mg, mol, basis):
    p = CE.parse_conc_unit(unit)
    assert p["conc_unit_kind"] == kind and p["conc_basis"] == basis
    if mg is not None:
        assert p["factor_mg_L"] == pytest.approx(mg)
    if mol is not None:
        assert p["factor_mol_L"] == pytest.approx(mol)


def test_parse_value_and_censoring():
    v, f = CE.parse_value(["1.04", "0.5*", "NR", None, "abc"])
    assert v[0] == 1.04 and v[1] == 0.5 and np.isnan(v[2]) and np.isnan(v[3]) and np.isnan(v[4])
    assert list(f) == ["", "asterisk", "missing_code", "missing_code", "non_numeric"]
    cen, kind = CE.censoring([">", "<=", "~", None, "ca"])
    assert list(cen) == [True, True, False, False, False]
    assert list(kind) == ["right", "left", "approximate", "", "approximate"]


def test_duration_hours():
    h = CE.duration_hours(["96", "2", "1", "30", "18"], ["h", "d", "wk", "mi", "hph"])
    assert list(h[:4]) == [96.0, 48.0, 168.0, 0.5] and np.isnan(h[4])


def test_endpoints_and_exposure_class():
    ep = CE.normalize_endpoint(["LC50*", "NOEC/", " EC50 ", "LD50"])
    assert list(ep) == ["LC50", "NOEC", "EC50", "LD50"]
    g = CE.endpoint_group(ep)
    assert list(g.fillna("-")) == ["median_effect", "threshold", "median_effect", "-"]
    cls = CE.exposure_class(g, pd.Series([96.0, 504.0, 240.0, 24.0]))
    assert list(cls) == ["acute", "chronic", "prolonged", "unknown"]


def test_element_mass():
    assert CE.element_mass("Cu") == pytest.approx(63.546, abs=1e-3)
    assert CE.element_mass("NR") is None and CE.element_mass("NH3") is None


def test_release_index_and_notes():
    html = '<a href="ecotox_ascii_06_11_2026.zip">x</a><a href="ecotox_ascii_09_15_2026.zip">y</a>'
    assert E.latest_release(html) == ("2026-09-15", "ecotox_ascii_09_15_2026.zip")
    notes = "FILE_NAME   RECORD_COUNT COLUMN_COUNT\n---\nchemicals.txt   18551   4\ntests.txt  725635  131\n"
    assert E.parse_release_notes(notes) == {
        "chemicals.txt": {"records": 18551, "columns": 4},
        "tests.txt": {"records": 725635, "columns": 131},
    }


def test_read_pipe_table_counts_bad_lines():
    raw = "a|b|c\n1|2|3\n4||6\n7|8\n9|10|11|\n".encode()
    df, meta = E.read_pipe_table(raw)
    assert meta["rows"] == 3 and meta["bad_lines"] == 1 and pd.isna(df.loc[1, "b"])
    assert df.loc[2, "c"] == "11"


# =========================================================================== NRC


@pytest.fixture(scope="module")
def nrc_book() -> dict[str, pd.DataFrame]:
    book = {}
    for sh in ("CALLS", "INCIDENT_COMMONS", "INCIDENT_DETAILS", "MATERIAL_INVOLVED"):
        df = pd.read_csv(FIX / "nrc" / f"{sh}.csv", dtype=str, keep_default_na=False)
        book[sh] = df.drop(columns=["source_file"]).replace("", None)
    return book


def test_nrc_dms_and_quadrants():
    lat, f = N.dms_to_decimal(
        ["31", "29", None, "0", "95", "28"],
        ["1", "11", None, "0", "0", "56"],
        ["54", "30", None, "0", "0", "10"],
        ["N", None, None, None, "N", "W"],
        kind="lat",
        default_quad=["N"] * 6,
    )
    assert lat[0] == pytest.approx(31 + 1 / 60 + 54 / 3600)
    assert list(f) == ["ok", "quadrant_inferred", "missing", "zero", "out_of_range", "bad_quadrant"]
    lon, _ = N.dms_to_decimal(
        ["144", "90"], ["24", "41"], [None, "0"], [None, None], kind="lon", default_quad=["E", "W"]
    )
    assert lon[0] == pytest.approx(144.4) and lon[1] == pytest.approx(-(90 + 41 / 60))


def test_nrc_amount_units():
    a = N.amount_with_unit([1200, 0, 0, 2, 3], ["GALLON(S)", "UNKNOWN AMOUNT", "NONE", "BARREL(S)", "MCF"])
    assert a["litres"][0] == pytest.approx(1200 * 3.785411784)
    assert np.isnan(a["value"][1]) and a["qualifier"][1] == "UNKNOWN AMOUNT" and a["qualifier"][2] == "NONE"
    assert (
        a["litres"][3] == pytest.approx(2 * 158.987294928) and np.isnan(a["litres"][4]) and a["value"][4] == 3
    )


def test_nrc_tidy_reports_and_materials(nrc_book):
    rep = N.tidy_reports(nrc_book, source_file="CY24.xlsx")
    mat = N.tidy_materials(nrc_book, source_file="CY24.xlsx")
    assert rep["report_id"].is_unique and len(rep) == 12
    r = rep.set_index("report_id")
    assert r.loc["1387970", "has_coordinates"] and r.loc["1387970", "lon"] < 0
    assert r.loc["1403658", "lon_flag"] == "bad_quadrant" and not r.loc["1403658", "has_coordinates"]
    assert r.loc["6155", "lon_flag"] == "quadrant_inferred" and r.loc["6155", "lon"] < 0
    assert r.loc["37862", "lon"] > 140  # Guam: east longitude
    assert r.loc["1387992", "water_supply_contaminated"] is True or bool(
        r.loc["1387992", "water_supply_contaminated"]
    )
    assert r.loc["1387964", "incident_datetime_local"] == pd.Timestamp("2023-12-31 20:40")
    assert r.loc["1", "incident_datetime_local"] == pd.Timestamp("1989-12-31 20:03")
    m = mat.set_index(["report_id", "material_seq"])
    assert m.loc[("521217", 1), "cas"] == "71-43-2"  # benzene, zeros stripped
    assert pd.isna(m.loc[("521217", 0), "cas"])  # 000000-00-0 placeholder
    assert pd.isna(m.loc[("521217", 2), "cas"])  # '10699-0' invalid
    assert m.loc[("1", 0), "amount_released_litres"] == pytest.approx(1200 * 3.785411784)
    assert m.loc[("2", 0), "amount_in_water_qualifier"] == "NONE" and pd.isna(
        m.loc[("2", 0), "amount_in_water"]
    )
    flags = CN.report_level_flags(rep, mat)
    full = rep.merge(flags, on="report_id")
    st = CN.coverage_stats(full)
    assert st["medium_water"]["n"] >= 1 and 0 <= st["medium_water"]["frac_with_cas"] <= 1
    flat = CN.flat_table(full, mat)
    assert len(flat) == len(mat) + (~rep["report_id"].isin(mat["report_id"])).sum()


def test_nrc_flag_parsing():
    f = N.yn_flag(["Y", "N", "U", None, "YES", "no"])
    assert list(f.astype("object")) == [True, False, pd.NA, pd.NA, True, False]
    assert N.year_from_filename("CY90.xlsx") == 1990 and N.year_from_filename("CY24.xlsx") == 2024
    assert list(N.zip_code(["2134.0", "70602", None]).fillna("-")) == ["02134", "70602", "-"]


# --------------------------------------------------------------------------- ECOTOX build on real rows


@pytest.fixture(scope="module")
def ecotox_tables() -> dict[str, pd.DataFrame]:
    out = {}
    for t in ("tests", "results", "species", "chemicals", "references", "media_characteristics"):
        df, meta = E.read_pipe_table((FIX / "ecotox" / f"{t}.txt").read_bytes())
        assert meta["bad_lines"] == 0
        out[t] = df
    return out


def test_ecotox_build_aquatic_on_release_rows(ecotox_tables):
    T = ecotox_tables
    mw = C.rdkit_identity("O=[N+]([O-])c1ccc(Cl)cc1")["mw_rdkit"]  # CAS 100-00-5
    chem = pd.DataFrame(
        {
            "cas": ["100-00-5"],
            "mw_g_mol": [mw],
            "dtxsid_final": ["DTXSID5020281"],
            "smiles_canonical": ["O=[N+]([O-])c1ccc(Cl)cc1"],
            "inchikey": [None],
            "structure_class": ["organic"],
        }
    )
    df = CE.build_aquatic(
        T["tests"],
        T["results"],
        T["species"],
        T["chemicals"],
        references=T["references"],
        media=T["media_characteristics"],
        chem_ids=chem,
    ).set_index("result_id")
    # terrestrial test, worm species and BCF endpoint are excluded
    assert {"2076576", "14871", "2482143"}.isdisjoint(df.index) and len(df) == 17
    r = df.loc["400906"]  # Danio rerio LC50 15 mg/L 96 h
    assert r["conc_mg_L"] == pytest.approx(15.0) and r["conc_mol_L"] == pytest.approx(15.0 / 1e3 / mw)
    assert r["exposure_class"] == "acute" and r["duration_h"] == 96 and r["taxon_group"] == "fish"
    assert r["cas"] == "100-00-5" and r["mw_source"] == "chemical_mw"
    r = df.loc["95474"]  # '>' 1000 ug/L
    assert r["conc_mg_L"] == pytest.approx(1.0) and bool(r["censored"]) and r["censor_kind"] == "right"
    assert (
        df.loc["13819", "conc_mg_L"] == pytest.approx(20.0)
        and df.loc["13819", "taxon_group"] == "crustaceans"
    )
    r = df.loc["739926"]  # algae, AI mg/L
    assert r["conc_basis"] == "active_ingredient" and r["conc_mg_L"] == pytest.approx(0.19)
    r = df.loc["127828"]  # NOEC 14 d, salt water
    assert r["exposure_class"] == "chronic" and r["duration_h"] == 336 and r["water_type"] == "saltwater"
    r = df.loc["2727520"]  # '>' 100 uM, no MW -> molar only
    assert (
        r["conc_mol_L"] == pytest.approx(1e-4)
        and np.isnan(r["conc_mg_L"])
        and r["conc_conversion"] == "molar_only"
    )
    assert np.isnan(df.loc["757362", "conc_value"]) and df.loc["757362", "conc_value_flag"] == "missing_code"
    assert (
        df.loc["794433", "conc_mg_L"] == pytest.approx(8.66) and df.loc["794433", "water_type"] == "saltwater"
    )
    r = df.loc["83061"]  # '9900*' ug/L as Cr
    assert r["conc_value_flag"] == "asterisk" and r["conc_mg_L"] == pytest.approx(9.9)
    assert r["mw_source"] == "ion_atomic_weight" and r["conc_mol_L"] == pytest.approx(
        9.9e-3 / 51.9961, rel=1e-4
    )
    assert df.loc["239383", "conc_conversion"] == "not_convertible"  # mg/kg
    assert df.loc["2108661", "conc_conversion"] == "not_convertible"  # ppt is ambiguous
    assert df.loc["1881", "taxon_group"] == "insects" and df.loc["2239632", "taxon_group"] == "molluscs"
    assert df.loc["2171615", "taxon_group"] == "amphibians"
    assert df.loc["85454", "endpoint_norm"] == "LC50" and df.loc["85454", "endpoint_flag"] == "asterisk"
    assert df.loc["29225", "water_type"] == "freshwater"  # 'FW/' comment marker
    assert df["publication_year"].notna().all()


def test_epa_crosscheck_counts():
    tidy = pd.DataFrame(
        {
            "result_id": ["1", "2", "3", "4"],
            "conc_mg_L": [1.0, 0.5, np.nan, 2.0],
            "conc_unit_kind": ["mass", "molar", "molar", "mass"],
            "mw_source": [None, "ion_atomic_weight", None, None],
            "conc_conversion": ["mass_only", "mass_and_molar", "molar_only", "mass_only"],
            "conc1_mean": ["1000", "10", "3", "2"],
            "conc1_unit": ["ug/L", "uM", "uM", "mg/L"],
            "duration_h": [96.0, 48.0, 24.0, np.nan],
        }
    )
    pull = pd.DataFrame(
        {
            "RESULT_NUMBER": ["1", "2", "3", "4", "9"],
            "CONC1_MEAN_STD": ["1*", "0.9", "0.4", "2", "5"],
            "CONC1_UNITS_STD": ["mg/L", "AI mg/L", "mg/L", "mg/L", "mg/L"],
            "TEST_DURATION_STD": ["4", "48", "1", "2"] + ["1"],
            "DURATION_UNITS_STD": ["d", "h", "d", "d", "d"],
        }
    )
    r = CE.crosscheck_epa_standardised(tidy, pull)
    c = r["concentration"]
    assert r["n_matched_results"] == 4 and c["compared"] == 3 and c["match"] == 2
    assert c["mismatch_molar_with_ion_basis"] == 1 and c["epa_value_ours_missing"] == {"molar_only": 1}
    assert r["duration"]["compared"] == 3 and r["duration"]["match"] == 3


def test_fingerprint_matrix(adore_excerpt):
    fp = A.fingerprint_matrix(adore_excerpt, "chem_MACCS_fp")
    assert fp.shape[0] == 37 and set(np.unique(fp)) <= {0, 1}
    assert "".join(map(str, fp[0])) == adore_excerpt["chem_MACCS_fp"].iloc[0]
