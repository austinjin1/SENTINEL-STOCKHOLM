# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for STAC chip infrastructure (no network).

Fixtures in tests/fixtures/stac are trimmed copies of real STAC items / metadata XML retrieved
2026-10-22 (Earth Search C1, Earth Search legacy with ``boa_offset_applied``, Planetary Computer
baselines 04.00 and 02.12). Synthetic arrays below only exercise code paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data import chips as C
from sentinel.data.sources import aquacorr as A
from sentinel.data.sources import stac as S

FIX = Path(__file__).parent / "fixtures" / "stac"


def _item(name: str) -> dict:
    return json.loads((FIX / name).read_text())


# ------------------------------------------------------------------------------ offset logic

def test_offset_earth_search_c1_from_raster_bands():
    it = _item("es_c1_item.json")
    sc = S.parse_s2_item(it, S.ES)
    assert sc.baseline == "05.09"
    assert sc.offset.offset_dn == -1000 and sc.offset.source == "raster:bands"
    assert sc.offset.consistent and not sc.offset.already_applied


def test_offset_already_applied_item_is_not_corrected_twice():
    it = _item("es_legacy_offset_applied_item.json")
    assert it["properties"]["earthsearch:boa_offset_applied"] is True
    # the trap: its raster:bands still advertise offset -0.1
    assert it["assets"]["red"]["raster:bands"][0]["offset"] == pytest.approx(-0.1)
    sc = S.parse_s2_item(it, S.ES)
    assert sc.offset.offset_dn == 0 and sc.offset.already_applied and sc.offset.consistent


def test_offset_planetary_computer_baseline_rule_and_xml():
    it = _item("mpc_b0400_item.json")
    sc = S.parse_s2_item(it, S.MPC)
    assert sc.raster_bands == {}  # MPC assets carry no scale/offset metadata
    assert sc.offset.offset_dn == -1000 and sc.offset.source == "baseline_rule"
    offs = S.parse_boa_offsets((FIX / "mpc_b0400_product_boa.xml").read_text())
    assert set(offs.values()) == {-1000} and len(offs) >= 12
    d = S.s2_offset_decision(it["properties"], None, S.xml_offset_from_bands(offs))
    assert d.offset_dn == -1000 and d.source == "product_xml" and d.consistent

    old = _item("mpc_b0212_item.json")
    offs_old = S.parse_boa_offsets((FIX / "mpc_b0212_product_boa.xml").read_text())
    assert offs_old == {}
    d = S.s2_offset_decision(old["properties"], None, S.xml_offset_from_bands(offs_old))
    assert d.offset_dn == 0 and d.consistent


def test_offset_inconsistent_or_unknown_is_flagged():
    d = S.s2_offset_decision({"s2:processing_baseline": "02.12"}, {"scale": 1e-4, "offset": -0.1})
    assert not d.consistent
    d = S.s2_offset_decision({"s2:processing_baseline": "05.10"}, None, xml_offset_dn=0)
    assert not d.consistent
    d = S.s2_offset_decision({"s2:processing_baseline": "00.01"}, {"scale": 1e-4, "offset": -0.1})
    assert not d.consistent  # legacy non-ESA baseline: raster:bands alone is not trusted
    d = S.s2_offset_decision({}, None)
    assert not d.consistent
    assert S.parse_baseline("05.09") == pytest.approx(5.09)
    assert S.parse_baseline(None) is None and S.parse_baseline("x") is None
    assert S.xml_offset_from_bands({1: -1000, 2: -900}) is None


def test_harmonise_raw_and_precorrected_agree():
    rng = np.random.default_rng(0)
    corrected = rng.integers(1, 3000, size=(8, 8)).astype(np.uint16)  # offset already removed
    raw = corrected + 1000                                              # same pixels, raw DN
    raw[0, 0] = 0
    corrected[0, 0] = 0                                                 # nodata in both
    h_raw = S.harmonise_dn(raw, -1000)
    h_cor = S.harmonise_dn(corrected, 0)
    assert np.array_equal(h_raw, h_cor)
    assert h_raw[0, 0] == S.NODATA_I16 and h_raw.dtype == np.int16
    r = S.to_reflectance(h_raw)
    assert np.isnan(r[0, 0]) and r[1, 1] == pytest.approx(corrected[1, 1] * 1e-4)
    big = np.array([[65535, 1]], np.uint16)
    assert S.harmonise_dn(big, -1000)[0, 0] == 32767
    assert S.harmonise_dn(big, -1000)[0, 1] == -999  # negative reflectance stays representable


def test_offset_sanity_flags_double_and_missing_correction():
    chip = _synthetic_chip(water_value=150)       # harmonised DN 150 -> reflectance 0.015
    f = C.s2_chip_features(chip)
    assert C.s2_offset_suspect(f) == ""
    dbl = chip.copy()
    dbl[:12] = np.where(dbl[:12] == S.NODATA_I16, dbl[:12], dbl[:12] - 1000)
    assert C.s2_offset_suspect(C.s2_chip_features(dbl)) == "double"
    miss = chip.copy()
    miss[:12] = np.where(miss[:12] == S.NODATA_I16, miss[:12], miss[:12] + 1000)
    assert C.s2_offset_suspect(C.s2_chip_features(miss)) == "bright_water"
    cloudy = miss.copy()
    cloudy[C.S2_STORE_LAYERS.index("SCL"), :32] = 9   # half the chip cloud: brightness is explained
    assert C.s2_offset_suspect(C.s2_chip_features(cloudy)) == ""


# ----------------------------------------------------------------------- footprint alignment

def _native_reader(factor_name: str):
    def read(r, c, h, w):
        rows = np.arange(r, r + h)[:, None]
        cols = np.arange(c, c + w)[None, :]
        return (rows * 1000 + cols).astype(np.int64)
    return read


@pytest.mark.parametrize("factor", [1, 2, 6])
@pytest.mark.parametrize("x", [200015.0, 200025.0, 200061.0, 205555.5])
def test_replication_maps_each_10m_pixel_to_its_native_pixel(factor, x):
    g = S.chip_grid(x, 4_690_000.0 - 3.0, 199980.0, 4700040.0, 10.0, 64, "EPSG:32617")
    up = S.replicate_window(_native_reader(str(factor)), g, factor)
    assert up.shape == (64, 64)
    rows10 = g.row0 + np.arange(64)
    cols10 = g.col0 + np.arange(64)
    expect = (rows10[:, None] // factor) * 1000 + (cols10[None, :] // factor)
    assert np.array_equal(up, expect)


def test_all_layers_share_the_chip_footprint():
    g = S.chip_grid(250_123.0, 4_650_456.0, 199980.0, 4700040.0, 10.0, 64, "EPSG:32617")
    left, bottom, right, top = g.bounds
    assert right - left == pytest.approx(640.0) and top - bottom == pytest.approx(640.0)
    assert g.point_row == 32 and g.point_col == 32
    assert left <= g.point_x < right and bottom < g.point_y <= top
    # the point lies inside chip pixel (32, 32)
    assert left + 320 <= g.point_x < left + 330 and top - 330 < g.point_y <= top - 320
    fy, fx = g.subpixel
    assert 0 <= fy < 1 and 0 <= fx < 1
    # v1 defect for contrast: a native 64x64 window of a 20 m band spans 1280 m (4x the area)
    assert (64 * 20) ** 2 / (64 * 10) ** 2 == 4


# ---------------------------------------------------------------------------- SCL / QA_PIXEL

def test_scl_fractions_and_centre_window():
    scl = np.full((64, 64), 6, np.uint8)
    scl[:32] = 4            # top half vegetation
    scl[31:34, 31:34] = 6
    scl[31, 31] = 9         # one cloud pixel in the centre 3x3
    fr = S.scl_fractions(scl)
    assert sum(fr.values()) == pytest.approx(1.0)
    assert fr["vegetation"] == pytest.approx((32 * 64 - 3) / 4096)  # row 31 of the 3x3 overlaps the top half
    sl = S.centre_slice(64, 3)
    assert (sl.start, sl.stop) == (31, 34)
    c3 = S.scl_summary(scl[sl, sl], "c3")
    assert c3["c3_n_water"] == 8 and c3["c3_n_valid"] == 8 and c3["c3_cloud_high"] == pytest.approx(1 / 9)


def test_qa_pixel_bits():
    qa = np.array([[1 << 6 | 1 << 7, 1 << 3, 1, 21824, 21952]], np.uint16)
    f = S.qa_pixel_flags(qa)
    assert f["clear"][0, 0] and f["water"][0, 0] and not f["cloud"][0, 0]
    assert f["cloud"][0, 1] and f["fill"][0, 2]
    assert f["clear"][0, 3] and not f["water"][0, 3]  # 21824 = clear land, low confidences
    assert f["clear"][0, 4] and f["water"][0, 4]      # 21952 = clear water, low confidences
    fr = S.qa_pixel_fractions(qa, "qa")
    assert fr["qa_water"] == pytest.approx(0.4)


# ------------------------------------------------------------------------ STAC item parsing

def test_parse_items_time_tile_geometry():
    es = S.parse_s2_item(_item("es_c1_item.json"), S.ES)
    assert es.tile == "17TKG" and es.platform == "S2B" and es.epsg == 32617
    assert es.sensing_time == es.datetime_item == "2023-08-27T16:31:57.078000Z"
    assert es.relative_orbit == 40 and es.view_zenith is not None and es.sun_zenith is not None
    assert set(S.S2_LAYER_NAMES) <= set(es.assets)
    mp = S.parse_s2_item(_item("mpc_b0400_item.json"), S.MPC)
    assert mp.tile == "19TCH" and mp.platform == "S2A" and mp.relative_orbit == 111
    assert mp.sensing_time is None  # MPC datetime is the datatake start, not the granule time
    g = S.parse_granule_xml((FIX / "mpc_b0400_granule_trim.xml").read_text())
    assert g["sensing_time"] == "2022-07-23T15:41:25.623269Z"
    assert mp.datetime_item.startswith("2022-07-23T15:36:11")
    assert 0 < g["view_zenith"] < 15 and 0 <= g["view_azimuth"] < 360 and 0 < g["sun_zenith"] < 90


def test_dedupe_and_rank():
    base = S.parse_s2_item(_item("mpc_b0400_item.json"), S.MPC)
    import copy
    re_ = copy.deepcopy(base)
    re_.id, re_.baseline = base.id + "_reproc", "05.10"
    assert [s.baseline for s in S.dedupe_mpc_reprocessing([base, re_])] == ["05.10"]
    es = S.parse_s2_item(_item("es_c1_item.json"), S.ES)
    lon, lat = -83.30, 41.70
    ranked = S.rank_scenes_for_point([base, es], lon, lat)
    assert ranked and ranked[0].source == S.ES  # MPC 19TCH does not cover Lake Erie
    assert S.point_interior_distance(es.geometry, lon, lat) > 0
    assert S.point_interior_distance(base.geometry, lon, lat) < 0


# ------------------------------------------------------------------------------------- JRC

def test_jrc_point_stats_and_tile_key():
    from affine import Affine
    occ = np.zeros((21, 21), np.uint8)
    occ[:, 12:] = 90          # water east of column 12
    occ[0, 0] = 255           # nodata
    tr = Affine(0.00025, 0, -93.0, 0, -0.00025, 45.0)
    lon_land = -93.0 + 10.5 * 0.00025   # centre of column 10 (two pixels west of the water)
    lat = 45.0 - 10.5 * 0.00025
    st = S.jrc_point_stats(occ, tr, lon_land, lat)
    assert st["jrc_occ_point"] == 0 and st["jrc_dist_land_m"] == pytest.approx(0, abs=1e-6)
    assert 30 < st["jrc_dist_water_m"] < 45 and st["jrc_water_ok"]
    lon_w = -93.0 + 18.5 * 0.00025
    st = S.jrc_point_stats(occ, tr, lon_w, lat)
    assert st["jrc_occ_point"] == 90 and st["jrc_water_ok"] and st["jrc_dist_land_m"] > 100
    assert S.jrc_tile_key(-93.58, 44.93) == (-100, 50)
    assert S.jrc_tile_key(6.5, 46.4) == (0, 50)


# ------------------------------------------------------------------------- chip store / IO

def _synthetic_chip(water_value: int = 150) -> np.ndarray:
    chip = np.full((len(C.S2_STORE_LAYERS), 64, 64), 400, np.int16)
    L = {n: i for i, n in enumerate(C.S2_STORE_LAYERS)}
    for b in S.S2_REFLECTANCE_LAYERS:
        chip[L[b], 20:44, 20:44] = water_value
    chip[L["SCL"]] = 4
    chip[L["SCL"], 20:44, 20:44] = 6
    chip[L["AOT"]] = 120
    chip[L["JRC"]] = 0
    chip[L["JRC"], 20:44, 20:44] = 95
    chip[: len(S.S2_REFLECTANCE_LAYERS), :2, :] = S.NODATA_I16
    chip[L["SCL"], :2, :] = 0
    return chip


def test_chip_features_centre_medians():
    f = C.s2_chip_features(_synthetic_chip(200))
    assert f["scl_c3_n_water"] == 9 and f["rw3_B04"] == pytest.approx(0.02)
    assert f["scl_c7_n_water"] == 49 and f["aot_c3"] == pytest.approx(0.12)
    assert f["jrc_c3_median"] == 95 and 0 < f["chip_nodata_frac"] < 0.05


def test_chip_writer_roundtrip_and_resume(tmp_path):
    root = tmp_path / "store"
    w = C.ChipWriter(root, layers=C.S2_STORE_LAYERS, dtype="int16", semantics={"t": 1},
                     max_shard_chips=3)
    chips = [_synthetic_chip(100 + i) for i in range(5)]
    for i, ch in enumerate(chips):
        w.add(f"k{i}", ch, {"scene": "s", "i": i})
    w.add("bad", None, {"scene": "s"}, status="no_scene")
    w.close()
    assert len(list((root / "shards").glob("*.npz"))) == 2
    idx = C.consolidate(root)
    assert set(idx["status"]) == {"ok", "no_scene"} and len(idx) == 6
    store = C.ChipStore(root)
    for i in range(5):
        cid = store.index.set_index("request_key").loc[f"k{i}", "chip_id"]
        assert np.array_equal(store.load(cid), chips[i])
    store.close()
    # orphan shard (crash before its index part) is removed on restart; done keys are skipped
    (root / "shards" / "shard-00099.npz").write_bytes(b"x")
    w2 = C.ChipWriter(root, layers=C.S2_STORE_LAYERS, dtype="int16", semantics={"t": 1})
    assert w2.is_done("k0") and w2.is_done("bad") and not w2.is_done("k9")
    assert not (root / "shards" / "shard-00099.npz").exists()
    w3 = C.ChipWriter(root, layers=C.S2_STORE_LAYERS, dtype="int16", semantics={"t": 1},
                      retry_failed=True)
    assert w3.is_done("k0") and not w3.is_done("bad")
    with pytest.raises(ValueError):
        C.ChipWriter(root, layers=["B02"], dtype="int16", semantics={})


def test_index_columns_keep_types():
    df = pd.DataFrame({"flag": [True, None, False], "x": [1, None, 2.5], "s": ["a", None, 3],
                       "d": [{"k": 1}, None, None]})
    out = C.normalise_columns(df)
    assert str(out["flag"].dtype) == "boolean" and out["flag"].tolist()[0] is True
    assert pd.isna(out["flag"][1]) and out["flag"][2] == False  # noqa: E712 — not the string 'False'
    assert out["x"].dtype == np.float64 and out["s"].tolist() == ["a", None, "3"]
    assert json.loads(out["d"][0]) == {"k": 1}


# ------------------------------------------------------------------------------- AquaCorr

def test_aquacorr_srcflag_decoding():
    # extraction/lab (1) | restrictive tier (1<<2) | total fraction (2<<4) | discrete (1<<6) | recommended
    v = 1 | (1 << 2) | (2 << 4) | (1 << 6) | (1 << 9)
    out = A.decode_srcflag(pd.Series([v, 2 | (1 << 8), None, 0]))
    assert out.loc[0, "method_class"] == "extraction_lab" and out.loc[0, "quality_tier"] == "restrictive"
    assert out.loc[0, "fraction"] == "total" and out.loc[0, "depth_type"] == "discrete"
    assert bool(out.loc[0, "recommended"]) and not bool(out.loc[0, "outlier"])
    assert out.loc[1, "method_class"] == "in_vivo_sensor" and bool(out.loc[1, "outlier"])
    assert pd.isna(out.loc[2, "method_class"]) and pd.isna(out.loc[2, "recommended"])
    assert out.loc[3, "method_class"] == "unknown"
    reg = A.assign_region(np.array([44.9, 46.4, -33.0, np.nan]), np.array([-93.6, 6.5, 151.0, 0.0]))
    assert list(reg) == ["US_conus", "Europe", "other", "unknown"]


# --------------------------------------------------------------------------------- GLORIA / SRF

GFIX = Path(__file__).parent / "fixtures" / "gloria"


def _tiny_xlsx() -> bytes:
    """Minimal OOXML workbook (shared + inline strings, numbers) to exercise the reader."""
    import io
    import zipfile
    files = {
        "xl/workbook.xml": '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                           '<sheets><sheet name="Spectral Responses (S2A)" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
                                      'relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
                                      '</Relationships>',
        "xl/sharedStrings.xml": '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                                '<si><t>SR_WL</t></si><si><t>S2A_SR_AV_B4</t></si></sst>',
        "xl/worksheets/sheet1.xml": '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                                    '<sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>'
                                    '<c r="D1" t="inlineStr"><is><t>note</t></is></c></row>'
                                    '<row r="2"><c r="A2"><v>664</v></c><c r="B2"><v>0.5</v></c></row>'
                                    '</sheetData></worksheet>'}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return buf.getvalue()


def test_minimal_xlsx_reader():
    from sentinel.data.sources import gloria as G
    sheets = G.read_xlsx(_tiny_xlsx())
    rows = sheets["Spectral Responses (S2A)"]
    assert rows[0] == ["SR_WL", "S2A_SR_AV_B4", None, "note"]
    assert rows[1] == [664.0, 0.5]


def test_band_simulation_flat_nan_and_real_excerpt():
    from sentinel.data.sources import gloria as G
    wl = G.WL
    srf = pd.read_csv(GFIX / "s2a_srf_b2_b4_excerpt.csv")   # official ESA S2A SRF v3.2 excerpt
    grid_srf = pd.DataFrame({"wavelength_nm": wl})
    for b in ("B2", "B4"):
        r = G._on_grid(srf.wavelength_nm.to_numpy(float), srf[b].to_numpy(float), wl)
        grid_srf[f"S2A_{b}"] = r / r.sum()
    flat = np.full((2, wl.size), 0.01)
    flat[1, wl == 665] = np.nan           # inside B4 support, outside B2 support
    sim = G.simulate_bands(flat, grid_srf)
    assert sim.loc[0, "rrs_S2A_B4"] == pytest.approx(0.01) and sim.loc[0, "rrs_S2A_B2"] == pytest.approx(0.01)
    assert np.isnan(sim.loc[1, "rrs_S2A_B4"]) and sim.loc[1, "rrs_S2A_B2"] == pytest.approx(0.01)
    rrs = pd.read_csv(GFIX / "gloria_rrs_excerpt.csv")     # real GLORIA spectra (3 samples)
    R = G.rrs_matrix(rrs)
    sim = G.simulate_bands(R, grid_srf)
    w = grid_srf["S2A_B4"].to_numpy()
    sup = w >= 0.01 * w.max()
    for i in range(len(R)):
        if np.isfinite(R[i, sup]).all():
            assert sim.loc[i, "rrs_S2A_B4"] == pytest.approx((R[i, sup] * w[sup]).sum() / w[sup].sum())
            # band value lies within the spectrum's range over the band
            assert R[i, sup].min() <= sim.loc[i, "rrs_S2A_B4"] <= R[i, sup].max()


def test_srf_coverage_rule_and_tophats():
    from sentinel.data.sources import gloria as G
    # a band that is 50 % outside 350-900 nm is not simulated
    wl = np.arange(880, 921, dtype=float)
    full = np.arange(880, 921)
    on = G._on_grid(wl, np.ones_like(wl), G.WL)
    assert on.sum() / G._on_grid(wl, np.ones_like(wl), full).sum() < G.COVERAGE_MIN
    for cam, bands in G.CAMERA_TOPHATS.items():
        assert set(bands) == {"blue", "green", "red", "nir"}
        assert all(400 < lo < hi < 900 for lo, hi in bands.values())


def test_gloria_tidy_flags_and_units():
    from sentinel.data.sources import gloria as G
    meta = pd.read_csv(GFIX / "gloria_meta_excerpt.csv")
    qc = pd.read_csv(GFIX / "gloria_qc_excerpt.csv")
    rrs = pd.read_csv(GFIX / "gloria_rrs_excerpt.csv")
    qc = pd.concat([qc, pd.DataFrame({"GLORIA_ID": [np.nan]})], ignore_index=True)  # trailing empty row
    sim = pd.DataFrame({"rrs_X": [0.1, 0.2, 0.3]})
    df = G.tidy_samples({"meta": meta, "qc": qc, "rrs": rrs}, sim)
    assert len(df) == 3 and df.GLORIA_ID.is_unique
    assert df.any_qc_flag.dtype == bool
    assert (df.loc[df.TSS.notna(), "TSS_unit"] == "g/m3").all() and df.loc[df.Chla.isna(), "Chla_unit"].isna().all()
    assert df.set_index("GLORIA_ID").loc["GID_2000", "rrs_X"] == pytest.approx(0.2)
    assert set(df.water_body_type_name) <= set(G.WATER_BODY.values())


def test_offset_group_check_detects_shifted_group():
    rows = []
    for src, bl, off, b11 in (("earth-search", "05.00", -1000, 0.012), ("planetary-computer", "02.12", 0, 0.010),
                              ("planetary-computer", "04.00", -1000, 0.112)):  # last group: offset missing
        for k in range(5):
            rows.append({"status": "ok", "stac_source": src, "processing_baseline": bl, "offset_dn": off,
                         "scl_chip_cloud_medium": 0.0, "scl_chip_cloud_high": 0.0, "scl_chip_thin_cirrus": 0.0,
                         "scl_chip_n_water": 500, "qc_water_b11_median": b11 + 0.001 * k, "qc_refl_p01_min": 0.0})
    rows.append({"status": "no_scene"})
    idx = pd.DataFrame(rows)
    g = C.offset_group_check(idx).set_index("processing_baseline")
    assert g.loc["04.00", "median"] > 0.1 > g.loc["05.00", "median"] and g.loc["02.12", "n"] == 5
    flags = C.recompute_offset_suspect(idx)
    assert pd.isna(flags.iloc[-1]) and (flags.iloc[10:15] == "bright_water").all() and (flags.iloc[:10] == "").all()
    assert C.offset_group_check(pd.DataFrame({"status": ["no_scene"]})).empty


def test_trimmed_catalog_items_keep_offset_and_assets():
    from sentinel.data import matchups as M
    for name, src in (("es_c1_item.json", S.ES), ("es_legacy_offset_applied_item.json", S.ES),
                      ("mpc_b0400_item.json", S.MPC)):
        it = _item(name)
        full, trim = S.parse_s2_item(it, src), S.parse_s2_item(M.trim_item(it, src), src)
        assert trim.offset == full.offset and set(trim.assets) == set(S.S2_LAYER_NAMES)
        assert trim.metadata_hrefs == full.metadata_hrefs and trim.geometry == full.geometry
    cat = M.items_to_catalog([(S.ES, _item("es_c1_item.json"))])
    sc = M.scenes_from_catalog(cat)["S2B_T17TKG_20230827T162957_L2A"]
    assert sc.tile == "17TKG" and sc.offset.offset_dn == -1000


def test_naip_site_join_and_chip_features():
    from sentinel.data import naip as N
    sq = lambda x0, y0, d: json.dumps({"type": "Polygon", "coordinates": [[[x0, y0], [x0 + d, y0], [x0 + d, y0 + d],
                                                                           [x0, y0 + d], [x0, y0]]]})
    cat = pd.DataFrame({"item_id": ["a", "b", "c"], "date": ["2021-06-18", "2021-06-18", "2019-07-27"],
                        "year": [2021, 2021, 2019], "gsd": [0.6, 0.6, 0.6],
                        "geometry": [sq(-93.70, 44.90, 0.10), sq(-93.62, 44.90, 0.10), sq(-93.70, 44.90, 0.10)]})
    sites = pd.DataFrame({"point_key": ["p1", "p2"], "lon": [-93.61, -94.5], "lat": [44.95, 44.95]})
    j = N.sites_in_items(sites, cat)
    assert set(j.point_key) == {"p1"} and len(j) == 2  # one per acquisition date; p2 is outside all DOQQs
    assert j.loc[j.naip_year == 2021, "item_id"].item() == "b"  # most interior of the two overlapping DOQQs
    chip = np.full((4, 256, 256), 100, np.uint8)
    chip[3, 120:137, 120:137] = 20                      # dark NIR (water-like) centre
    chip[:, :10, :] = 0                                 # fill
    f = N.naip_chip_features(chip)
    assert f["c9_NIR_median"] == 20 and f["c9_darknir_frac"] == 1.0 and 0.03 < f["chip_fill_frac"] < 0.05
