# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for the biological-community data layer (NARS, UK EA, taxonomy, RF baseline helpers).

Fixtures under tests/fixtures/{nars,uk_ea}/ are tiny excerpts of the real public files (first
rows / a few visits). Small synthetic frames are used only to exercise code paths; no test
output is a research result. No network access.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data import bio_common as BC
from sentinel.data import bio_nars as BN
from sentinel.data import bio_taxonomy as T
from sentinel.data import bio_uk_ea as BU
from sentinel.data.sources import nars as N
from sentinel.data.sources import uk_ea as U

FX = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------- common helpers

def test_parse_censored_tokens():
    r = BC.parse_censored(pd.Series(["<0.03", "1.5", "No flow/discharge at sampling point", "", ">100", "<= 2"]))
    assert r.loc[0, "value"] == pytest.approx(0.03) and r.loc[0, "censored"] and r.loc[0, "detection_limit"] == 0.03
    assert r.loc[1, "value"] == 1.5 and not r.loc[1, "censored"] and np.isnan(r.loc[1, "detection_limit"])
    assert np.isnan(r.loc[2, "value"]) and r.loc[2, "qualifier"].startswith("No flow")
    assert np.isnan(r.loc[3, "value"]) and r.loc[3, "qualifier"] == ""
    assert r.loc[4, "value"] == 100 and r.loc[4, "censor_op"] == ">" and not r.loc[4, "censored"]
    assert r.loc[5, "censored"] and r.loc[5, "value"] == 2


def test_missing_codes_become_nan_with_flag():
    v, flag = BC.mask_missing(pd.Series(["-9999", "-999999", "3.2", "ND", "Ice", ""]))
    assert np.isnan(v[0]) and flag[0] == "-9999"
    assert np.isnan(v[1]) and flag[1] == "-999999"
    assert v[2] == 3.2 and flag[2] == ""
    assert np.isnan(v[3]) and flag[3] == "ND"
    assert np.isnan(v[4]) and flag[4] == "Ice"
    assert np.isnan(v[5]) and flag[5] == ""


def test_ids_are_strings_with_leading_zeros():
    assert list(N.huc8_str(pd.Series(["H11130301", "1010001", "10260010", "", "H03W"]))) == \
        ["11130301", "01010001", "10260010", None, None]
    assert list(N.huc_fixed(pd.Series(["314020102", "31402010205"]), 10)) == ["0314020102", None]
    assert list(BC.id_str(pd.Series(["3142124.0", "0012", "", "nan"]))) == ["3142124", "0012", None, None]


# ----------------------------------------------------------------------------- NARS

def test_parse_data_page_excerpt():
    cat = N.parse_data_page((FX / "nars" / "data_page_excerpt.html").read_text(encoding="utf-8"))
    assert {"survey", "survey_code", "data_url", "meta_url", "data_name"} <= set(cat.columns)
    assert (cat["data_url"].str.startswith("https://www.epa.gov/")).all()
    assert "nrsa1819" in set(cat["survey_code"])
    sel = N.select_files(cat, ["nrsa1819", "nrsa0809"])
    assert not sel["data_name"].str.endswith(".zip").any()          # shapefiles excluded
    assert "nrsa_1819_benthic_macroinvertebrate_count_-_data.csv" in set(cat["data_name"])


def test_parse_date_mixed_formats():
    d = N.parse_date(pd.Series(["6/20/2018", "5-Aug-08", "2022-01-31", "06/20/2018", ""]))
    assert [x.strftime("%Y-%m-%d") if pd.notna(x) else None for x in d] == \
        ["2018-06-20", "2008-08-05", "2022-01-31", "2018-06-20", None]


def test_parse_visits_nrsa1819_fixture():
    raw = N.read_raw(FX / "nars" / "nrsa1819_siteinfo_sample.csv")
    v = N.parse_visits(raw, "nrsa1819")
    assert v["visit_id"].str.startswith("nrsa1819-").all() and v["visit_id"].is_unique
    assert v["huc8"].map(lambda s: isinstance(s, str) and len(s) == 8 and s.isdigit()).all()
    assert v["comid"].map(lambda s: isinstance(s, str)).all()
    assert v["huc2"].eq(v["huc8"].str[:2]).all()
    assert v["within_cycle_revisit"].sum() == (raw["VISIT_NO"] != "1").sum() >= 1
    assert v["date"].notna().all() and set(v["ag_eco9"]) <= {"CPL", "NAP", "NPL", "SAP", "SPL", "TPL", "UMW", "WMT", "XER"}


def test_parse_visits_mangled_reachcode_flagged():
    v = N.parse_visits(N.read_raw(FX / "nars" / "nrsa2324_siteinfo_sample.csv"), "nrsa2324")
    assert v["reachcode_mangled"].all() and v["reachcode"].isna().all()
    assert v["huc10"].map(lambda s: s is None or len(s) == 10).all()


def test_nla07_visit_key_without_uid():
    v = N.parse_visits(N.read_raw(FX / "nars" / "nla07_siteinfo_sample.csv"), "nla07")
    assert v["visit_id"].str.match(r"^nla07-NLA\d+-\d+-\d+$").all()
    assert v["reference_class"].isin(["R", "S", "T"]).all()          # REF / SO-SO / TRASH mapped


def test_parse_counts_keeps_missing_300_count_as_nan():
    raw = N.read_raw(FX / "nars" / "nrsa1819_benthic_sample.csv")
    c = N.parse_counts(raw, "nrsa1819", "benthic")
    blank = raw["TOTAL300"].eq("").to_numpy()
    assert blank.any() and c.loc[blank, "count_300"].isna().all()
    assert (c.loc[~blank, "count_300"] >= 0).all() and (c["count"] > 0).all()
    assert c["taxon_name"].str.isupper().all() and "src_family" in c.columns


def test_chem_suffix_wide_nondetect_and_units():
    raw = N.read_raw(FX / "nars" / "nrsa1819_chem_sample.csv")
    c = N.parse_chem_suffix_wide(raw, "nrsa1819", source_file="fixture")
    nh4 = c[c["analyte"] == "nh4_n"]
    nd = nh4[nh4["censored"]]
    assert len(nd) >= 1
    assert nd["value"].isna().all()                                  # no substitution at ingest
    assert (nd["detection_limit"] == nd["mdl"]).all()
    assert (nd["unit_status"] == "assumed_unit").all() and (nd["unit"] == "mg/L").all()
    tp = c[c["analyte"] == "tp"]
    assert (tp["unit"] == "ug/L").all() and tp["unit_status"].isin(["ok"]).all()
    cond = c[c["analyte"] == "cond"]
    assert cond["unit"].eq("uS/cm").all() and cond["unit_source"].str.contains("uS/cm").all()


def test_chem_plain_wide_converts_ug_to_mg_with_factor():
    raw = N.read_raw(FX / "nars" / "nrsa0809_chem_sample.csv")
    c = N.parse_chem_plain_wide(raw, "nrsa0809", N.NRSA0809_CHEM_COLUMNS, flag_suffix="_ALERT")
    tn = c[c["analyte"] == "tn"].reset_index(drop=True)
    assert (tn["unit_status"] == "converted").all()
    np.testing.assert_allclose(tn["value_canonical"], tn["value"] * 0.001)
    no2 = c[c["analyte"] == "no2_n"]
    assert (no2["flag"].isin(["", "R"])).all() and not no2["censored"].any()   # 'R' = below RL, not censored


def test_chem_long_ppb_is_ug_per_l_and_blank_units_filled():
    raw = N.read_raw(FX / "nars" / "nla17_atrazine_sample.csv")
    c = N.parse_chem_long(raw, "nla17")
    assert set(c["unit"]) == {"ug/L"}
    assert set(c["unit_status"]) <= {"ok", "assumed_unit"}


def test_unknown_unit_not_converted():
    v, st = N.to_canonical(pd.Series([1.0, 2.0]), pd.Series(["g", "mg/L"]), "tss", "mg/L")
    assert np.isnan(v[0]) and st[0] == "unknown_unit" and v[1] == 2.0 and st[1] == "ok"


def test_uid_aligner_remaps_superseded_uids():
    si = N.read_raw(FX / "nars" / "nrsa1819_siteinfo_sample.csv")
    fish = N.read_raw(FX / "nars" / "nrsa1819_fish_old_uid_sample.csv")
    assert not fish["UID"].isin(si["UID"]).all()                     # fixture really uses old UIDs
    al = BN.UidAligner(si, "nrsa1819")
    fixed = al(fish, "fish")
    assert fixed["UID"].isin(si["UID"]).all()
    assert al.counts["fish"]["unresolved"] == 0


def test_uid_aligner_does_not_guess_ambiguous_keys():
    si = pd.DataFrame({"UID": ["1", "2"], "SITE_ID": ["A", "A"], "DATE_COL": ["6/1/2018", "6/1/2018"],
                       "VISIT_NO": ["1", "1"]})
    raw = pd.DataFrame({"UID": ["99"], "SITE_ID": ["A"], "DATE_COL": ["6/1/2018"], "VISIT_NO": ["1"]})
    out = BN.UidAligner(si, "nrsa1819")(raw, "x")
    assert out["UID"].iloc[0] == "99"


def test_chemistry_wide_one_row_per_visit():
    raw = N.read_raw(FX / "nars" / "nrsa1819_chem_sample.csv")
    w = BN.chemistry_wide(N.parse_chem_suffix_wide(raw, "nrsa1819"))
    assert w["visit_id"].is_unique
    assert {"tp_ug_l", "tn_mg_l", "cond_us_cm", "ph_lab_su"} <= set(w.columns)
    assert (w["n_censored_analytes"] >= 0).all()


def test_condition_parse_long():
    df = pd.DataFrame({"UID": ["1", "2"], "SITE_ID": ["a", "b"], "BENT_MMI_COND": ["Good", "Not Assessed"],
                       "MMI_BENT": ["55.1", ""]})
    c = N.parse_condition(df, "nrsa1819", source_file="x")
    assert set(c["indicator"]) == {"bent_mmi_cond", "bent_mmi"}
    assert c[(c["indicator"] == "bent_mmi_cond")]["class"].tolist() == ["Good"]


# ---------------------------------------------------------------------------- UK EA

def test_ea_zip_tables_parse():
    z = FX / "uk_ea" / "INV_OPEN_DATA_sample.zip"
    sites = BU.load_sites(z, "invertebrate")
    assert sites["site_id"].map(lambda s: isinstance(s, str)).all()
    assert sites["lat"].between(49, 61).all() and sites["lon"].between(-7, 2.5).all()
    met = BU.load_metrics(z, "invertebrate")
    assert met["sample_date"].notna().all() and "idx_whpt_aspt" in met.columns
    taxa = pd.concat(U.tidy_taxa_chunk(ch, "invertebrate") for ch in U.iter_zip_table(z, "_TAXA.csv"))
    assert taxa["taxon_list_item_key"].str.len().gt(10).all() and (taxa["abundance"] >= 0).all()
    assert taxa["log_abundance_class"].between(1, 5).all()


def test_ea_bng_conversion_known_point():
    lon, lat = U.bng_to_wgs84([530000.0, np.nan], [180000.0, 1.0])     # central London; missing -> NaN
    assert lat[0] == pytest.approx(51.5, abs=0.05) and lon[0] == pytest.approx(-0.13, abs=0.05)
    assert np.isnan(lat[1]) and np.isnan(lon[1])


def test_wq_observations_csv_parse():
    o = U.parse_observations_csv((FX / "uk_ea" / "wq_observations_sample.csv").read_bytes())
    cen = o[o["censored"]]
    assert len(cen) >= 2 and (cen["value"] == cen["detection_limit"]).all()
    assert o["determinand"].str.fullmatch(r"\d{4}").all()
    s = o[o["sample_time_local"].dt.month.isin([6, 7])].iloc[0]
    assert (s["sample_time_local"] - s["time_utc"].tz_localize(None)) == pd.Timedelta(hours=1)  # BST
    assert o["unit"].notna().all()


def test_wq_sampling_points_parse_and_crs_guard():
    doc = json.loads((FX / "uk_ea" / "wq_sampling_points_sample.json").read_text())
    p = U.parse_sampling_points(doc)
    assert len(p) == 2 and p["wq_easting_bng"].between(0, 700000).all() and p["wq_type"].notna().all()
    bad = {"member": [{**doc["member"][0], "geometry": {"asWKT": "POINT(-1.2 52.1) <http://www.opengis.net/def/crs/EPSG/0/4326>"}}]}
    with pytest.raises(ValueError):
        U.parse_sampling_points(bad)


def test_observation_params_limits():
    with pytest.raises(ValueError):
        U.observation_params([f"P{i}" for i in range(101)], ["0061"], "2020-01-01", "2020-12-31")
    with pytest.raises(ValueError):
        U.observation_params(["P1"], ["0061"], "2019-01-01", "2020-12-31")
    p = U.observation_params(["P1", "P2"], ["0061", "0077"], "2020-01-01", "2020-12-31", skip=2500)
    assert p["pointNotation"] == "P1,P2" and p["skip"] == 2500


def test_pairing_rule_nearest_within_1km():
    sites = pd.DataFrame({"site_id": ["s1", "s2", "s3", "s4"], "waterbody_type": ["WBRV", "WBRV", "WBES", "WBRV"],
                          "easting_bng": [1000.0, 5000.0, 1000.0, 9000.0], "northing_bng": [1000.0, 5000.0, 1000.0, 9000.0]})
    pts = pd.DataFrame({"wq_point": ["A", "B", "C", "D"], "wq_type": ["F6", "F6", "FA", "F6"],
                        "wq_status": "O", "wq_point_label": "x",
                        "wq_easting_bng": [1100.0, 1500.0, 5000.0, 20000.0], "wq_northing_bng": [1000.0, 1000.0, 5050.0, 0.0]})
    pr = BU.pair_sites_to_points(sites, pts, max_dist_m=1000)
    assert dict(zip(pr["site_id"], pr["wq_point"])) == {"s1": "A"}      # s2: lake point excluded; s3 estuarine; s4 too far
    assert pr["pair_distance_m"].iloc[0] == pytest.approx(100.0)


def test_window_summaries():
    samples = pd.DataFrame({"sample_id": ["x"], "site_id": ["s1"], "sample_date": [pd.Timestamp("2020-06-15").date()]})
    pairs = pd.DataFrame({"site_id": ["s1"], "wq_point": ["A"], "pair_distance_m": [10.0]})
    obs = pd.DataFrame({"wq_point": "A", "determinand": "0061", "unit": "phunits",
                        "date": pd.to_datetime(["2020-05-01", "2020-06-10", "2020-07-10", "2019-07-01", "2020-09-01"]).date,
                        "value": [7.0, 7.4, 7.8, 6.0, 9.0], "censored": [False, False, True, False, False]})
    s = BU.summarise_windows(samples, pairs, obs).set_index("window")
    assert s.loc["pm30d", "n_obs"] == 2 and s.loc["pm30d", "median"] == pytest.approx(7.6)
    assert s.loc["pm30d", "nearest_lag_days"] == -5 and s.loc["pm30d", "frac_censored"] == 0.5
    assert s.loc["prior365d", "n_obs"] == 3                           # 2019-07-01, 2020-05-01, 2020-06-10


def test_years_needed_covers_windows():
    pairs = pd.DataFrame({"site_id": ["s1"], "wq_point": ["A"]})
    samples = pd.DataFrame({"site_id": ["s1", "s1"], "sample_date": [pd.Timestamp("2001-03-01").date(),
                                                                       pd.Timestamp("1990-05-01").date()]})
    y = BU.years_needed(pairs, samples, last_year=2026)
    assert set(y["year"]) == {2000, 2001}


# ------------------------------------------------------------------------- taxonomy

def test_clean_name_and_rank():
    assert T.clean_name("Sphaeriidae (Pea mussels)") == "SPHAERIIDAE"
    assert T.clean_name("BAETIS SP.") == "BAETIS"
    assert T.clean_name("Cricotopus / Orthocladius") == "CRICOTOPUS/ORTHOCLADIUS"
    lin = {"family": "BAETIDAE", "genus": "BAETIS"}
    assert T.identification_rank("BAETIS", lin) == "genus"
    assert T.identification_rank("BAETIS RHODANI", lin) == "species"
    assert T.identification_rank("BAETIDAE", lin) == "family"
    assert T.identification_rank("CRICOTOPUS/ORTHOCLADIUS", lin) == "complex"
    assert T.gbif_query_name("BAETIS RHODANI") == "Baetis rhodani"


def _src(rows):
    return pd.DataFrame(rows)


def test_harmonise_prefers_gbif_and_records_conflicts():
    src = _src([
        {"source": "nars_streams", "dataset": "nrsa1819", "assemblage": "benthic", "source_taxon_id": "1",
         "source_taxon_name": "BAETIS", "src_phylum": "ARTHROPODA", "src_order": "EPHEMEROPTERA",
         "src_family": "BAETIDAE", "src_genus": "BAETIS", "n_records": 30, "total_count": 300},
        {"source": "nars_lakes", "dataset": "nla22", "assemblage": "benthic", "source_taxon_id": "9",
         "source_taxon_name": "BAETIS", "src_phylum": "ARTHROPODA", "src_order": "EPHEMEROPTERA",
         "src_family": "WRONGIDAE", "src_genus": "BAETIS", "n_records": 3, "total_count": 3},
        {"source": "nars_streams", "dataset": "nrsa1819", "assemblage": "benthic", "source_taxon_id": "2",
         "source_taxon_name": "OLIGOCHAETA", "src_phylum": "ANNELIDA", "src_class": "OLIGOCHAETA",
         "n_records": 50, "total_count": 500},
    ])
    bb = {"macroinvertebrate|BAETIS": {"matchType": "EXACT", "confidence": 99, "rank": "GENUS", "kingdom": "Animalia",
                                       "phylum": "Arthropoda", "class": "Insecta", "order": "Ephemeroptera",
                                       "family": "Baetidae", "genus": "Baetis"},
          "macroinvertebrate|OLIGOCHAETA": {"matchType": "NONE"}}
    tm, cf = T.harmonise(src, bb)
    b = tm[tm["name_clean"] == "BAETIS"]
    # family voted by datasets + GBIF; higher ranks from the family lookup (GBIF and source agree)
    assert (b["h_family"] == "BAETIDAE").all() and (b["lineage_source"] == "family_lookup").all()
    assert (b["h_order"] == "EPHEMEROPTERA").all() and b["has_conflict"].all() and b["has_between_source_conflict"].all()
    assert ((cf["name_clean"] == "BAETIS") & (cf["rank"] == "family") & (cf["conflict_type"] == "between_sources")).any()
    o = tm[tm["name_clean"] == "OLIGOCHAETA"].iloc[0]
    assert o["lineage_source"] == "source_consensus" and o["identification_rank"] == "class"
    tm2, _fam = T.add_analysis_resolution(tm)
    o2 = tm2[tm2["name_clean"] == "OLIGOCHAETA"].iloc[0]
    assert o2["analysis_coarser"] and o2["analysis_taxon"] == "OLIGOCHAETA"   # mixed resolution kept
    b2 = tm2[tm2["name_clean"] == "BAETIS"].iloc[0]
    assert b2["analysis_taxon"] == "BAETIDAE" and not b2["analysis_coarser"]
    # genus-consistency rule: Baetidae records in nars_streams are 100% genus-level (30 >= 20 records)
    assert tm2[(tm2["name_clean"] == "BAETIS") & (tm2["source"] == "nars_streams")]["mixed_otu"].iloc[0] == "BAETIS"
    assert tm2[(tm2["name_clean"] == "BAETIS") & (tm2["source"] == "nars_lakes")]["mixed_otu"].iloc[0] == "BAETIDAE"


def test_ea_taxon_chain_flattening():
    raw = pd.read_csv(FX / "uk_ea" / "taxon_info_sample.csv", dtype=str, keep_default_na=False)
    raw.columns = [c.upper() for c in raw.columns]
    d = U.tidy_taxon_info(raw)
    z = FX / "uk_ea" / "INV_OPEN_DATA_sample.zip"
    counts = pd.concat(U.tidy_taxa_chunk(ch, "invertebrate") for ch in U.iter_zip_table(z, "_TAXA.csv"))
    ea = T.ea_source_taxa(d, {"invertebrate": counts})
    assert len(ea) == counts["taxon_list_item_key"].nunique()
    assert ea["source_taxon_name"].notna().all() and (ea["n_records"] >= 1).all()


# -------------------------------------------------------------------- RF baseline

def _baseline_module():
    p = REPO / "scripts" / "eval" / "baseline_bioassess_rf.py"
    spec = importlib.util.spec_from_file_location("baseline_bioassess_rf", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_loeo_and_group_folds_partition_sites():
    m = _baseline_module()
    sites = pd.Series([f"s{i}" for i in range(30)] + ["s0"])          # one site with two visits
    eco = pd.Series(["A"] * 10 + ["B"] * 12 + ["C"] * 8 + ["A"])
    folds = m.loeo_splits(sites, eco)
    assert len(folds) == 3
    m.check_partition(folds, sites.to_numpy())
    for f in folds:
        held = f.params["held_out"]
        assert set(f.test) == set(sites[eco == held])
    g = m.group_kfold_splits(sites, eco, 2)
    m.check_partition(g, sites.to_numpy())
    r = m.random_kfold_splits(sites, 5, seed=0)
    m.check_partition(r, sites.to_numpy())


def test_targets_use_documented_floors():
    m = _baseline_module()
    df = pd.DataFrame({"tp_ug_l": [0.2, 10.0], "ph_lab_su": [7.0, 8.0]})
    np.testing.assert_allclose(m.target_vector(df, "log10_tp_ug_l"), [0.0, 1.0])
    np.testing.assert_allclose(m.target_vector(df, "ph_lab"), [7.0, 8.0])


def test_nla07_microcystin_half_rl_substitution_undone():
    raw = pd.DataFrame({"SITE_ID": ["NLA1", "NLA2"], "VISIT_NO": ["1", "1"], "MCYST_TL_UGL": ["0.05", "0.8"],
                        "MCYST_PRES": ["1:LESS THAN RL (0.1 ug/L)", "2:PRESENT (> 0.1 ug/L)"]})
    c = N.parse_nla07_microcystin(raw).set_index("visit_id")
    assert np.isnan(c.loc["nla07-NLA1-1", "value"]) and c.loc["nla07-NLA1-1", "censored"]
    assert c.loc["nla07-NLA1-1", "detection_limit"] == 0.1 and "RL/2" in c.loc["nla07-NLA1-1", "qualifier"]
    assert c.loc["nla07-NLA2-1", "value"] == 0.8 and not c.loc["nla07-NLA2-1", "censored"]


def test_below_mdl_flag_without_censoring():
    df = pd.DataFrame({"UID": ["1", "2"], "MICX_RESULT": ["0.08", "0.5"], "MICX_MDL": ["0.1", "0.1"],
                       "MICX_UNITS": ["ppb", "ppb"], "MICX_FLAG": ["", ""]})
    c = N.parse_chem_suffix_wide(df, "nla12").set_index("visit_id")
    assert c.loc["nla12-1", "below_mdl"] and not c.loc["nla12-1", "censored"] and c.loc["nla12-1", "value"] == 0.08
    assert not c.loc["nla12-2", "below_mdl"]


def test_harmonise_rejects_gbif_homonym():
    # beetle genus Sphaeridium (Hydrophilidae) vs a GBIF match in a bivalve family
    src = _src([
        {"source": "uk_ea", "dataset": "ea_invertebrate", "assemblage": "invertebrate", "source_taxon_id": "K1",
         "source_taxon_name": "SPHAERIDIUM", "preferred_name": "SPHAERIDIUM", "ea_rank": "genus",
         "ea_taxon_type": "Other Macroinvertebrates", "src_family": "HYDROPHILIDAE", "src_genus": "SPHAERIDIUM",
         "n_records": 5, "total_count": 5},
        {"source": "uk_ea", "dataset": "ea_invertebrate", "assemblage": "invertebrate", "source_taxon_id": "K2",
         "source_taxon_name": "HYDROPHILIDAE", "preferred_name": "HYDROPHILIDAE", "ea_rank": "family",
         "ea_taxon_type": "Other Macroinvertebrates", "src_family": "HYDROPHILIDAE", "n_records": 9, "total_count": 9},
    ])
    bb = {"macroinvertebrate|SPHAERIDIUM": {"matchType": "EXACT", "confidence": 98, "rank": "GENUS", "kingdom": "Animalia",
                                            "phylum": "Mollusca", "class": "Bivalvia", "order": "Venerida",
                                            "family": "Limnocyrenidae", "genus": "Sphaeridium"},
          "macroinvertebrate|HYDROPHILIDAE": {"matchType": "EXACT", "confidence": 99, "rank": "FAMILY",
                                              "kingdom": "Animalia", "phylum": "Arthropoda", "class": "Insecta",
                                              "order": "Coleoptera", "family": "Hydrophilidae"}}
    tm, cf = T.harmonise(src, bb)
    s1 = tm[tm["name_clean"] == "SPHAERIDIUM"].iloc[0]
    assert s1["h_family"] == "HYDROPHILIDAE" and s1["h_order"] == "COLEOPTERA" and s1["h_phylum"] == "ARTHROPODA"
    assert s1["lineage_source"] == "family_lookup" and s1["gbif_family"] == "LIMNOCYRENIDAE"
    assert ((cf["name_clean"] == "SPHAERIDIUM") & (cf["conflict_type"] == "source_vs_gbif")).any()


def test_spelling_correction_against_gbif_names():
    src = _src([
        {"source": "nars_lakes", "dataset": "nla22", "assemblage": "zooplankton", "source_taxon_id": "1",
         "source_taxon_name": "OLIGOCHAETA", "src_phylum": "ANNILIDA", "src_subclass": "OLIGOCHAETA",
         "n_records": 4, "total_count": 4},
        {"source": "nars_lakes", "dataset": "nla22", "assemblage": "zooplankton", "source_taxon_id": "2",
         "source_taxon_name": "NAIS", "src_phylum": "ANNELIDA", "src_family": "NAIDIDAE", "src_genus": "NAIS",
         "n_records": 4, "total_count": 4},
    ])
    bb = {"zooplankton|NAIS": {"matchType": "EXACT", "confidence": 99, "rank": "GENUS", "kingdom": "Animalia",
                               "phylum": "Annelida", "class": "Clitellata", "family": "Naididae", "genus": "Nais"}}
    tm, cf = T.harmonise(src, bb)                                    # reviewed fix list applies
    o = tm[tm["name_clean"] == "OLIGOCHAETA"].iloc[0]
    assert o["h_phylum"] == "ANNELIDA"
    sc = cf[cf["conflict_type"] == "spelling_corrected"]
    assert len(sc) == 1 and sc["chosen"].iloc[0] == "ANNELIDA"


def test_real_names_are_not_spelling_corrected():
    # Phormidiaceae is a real family (GBIF EXACT at rank family): never merged into Phormidesmiaceae
    src = _src([
        {"source": "nars_lakes", "dataset": "nla12", "assemblage": "phytoplankton", "source_taxon_id": "1",
         "source_taxon_name": "PHORMIDIUM", "src_phylum": "CYANOBACTERIA", "src_family": "PHORMIDIACEAE",
         "src_genus": "PHORMIDIUM", "n_records": 3, "total_count": 3},
        {"source": "nars_lakes", "dataset": "nla12", "assemblage": "phytoplankton", "source_taxon_id": "2",
         "source_taxon_name": "PHORMIDESMIS", "src_phylum": "CYANOBACTERIA", "src_family": "PHORMIDESMIACEAE",
         "src_genus": "PHORMIDESMIS", "n_records": 3, "total_count": 3},
    ])
    bb = {"algae|PHORMIDESMIS": {"matchType": "EXACT", "confidence": 99, "rank": "GENUS", "kingdom": "Bacteria",
                                 "phylum": "Cyanobacteria", "family": "Phormidesmiaceae", "genus": "Phormidesmis"}}
    rv = {"family|PHORMIDIACEAE": {"matchType": "EXACT", "rank": "FAMILY", "canonicalName": "Phormidiaceae"}}
    tm, cf = T.harmonise(src, bb, rv)
    assert tm[tm["name_clean"] == "PHORMIDIUM"].iloc[0]["h_family"] == "PHORMIDIACEAE"
    assert not (cf.get("conflict_type", pd.Series(dtype=object)) == "spelling_corrected").any()


def test_window_summaries_exclude_effluent_and_incident_samples():
    samples = pd.DataFrame({"sample_id": ["x"], "site_id": ["s1"], "sample_date": [pd.Timestamp("2020-06-15").date()]})
    pairs = pd.DataFrame({"site_id": ["s1"], "wq_point": ["A"], "pair_distance_m": [10.0]})
    obs = pd.DataFrame({"wq_point": "A", "determinand": "0111", "unit": "MILLIGRAM PER LITRE",
                        "date": [pd.Timestamp("2020-06-10").date()] * 3, "value": [0.1, 9.0, 5.0], "censored": False,
                        "sample_material": ["RIVER / RUNNING SURFACE WATER", "FINAL SEWAGE EFFLUENT",
                                            "RIVER / RUNNING SURFACE WATER"],
                        "sampling_purpose": ["ENVIRONMENTAL MONITORING (GQA & RE ONLY)", "ENVIRONMENTAL MONITORING (GQA & RE ONLY)",
                                             "UNPLANNED REACTIVE MONITORING (POLLUTION INCIDENTS)"]})
    s = BU.summarise_windows(samples, pairs, obs).set_index("window")
    assert s.loc["pm30d", "n_obs"] == 1 and s.loc["pm30d", "median"] == pytest.approx(0.1)
