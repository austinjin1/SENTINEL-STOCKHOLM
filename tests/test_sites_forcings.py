# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Offline tests: site registry, flow network, NLDI / StreamCat / gridMET / Open-Meteo clients.

Fixtures (tests/fixtures/<source>/) are recorded real responses retrieved 2026-10-22:
* nldi/      NLDI nwissite lookup (USGS-05586300), a 404 body, UM/DM navigation (first 8 features).
* streamcat/ POST /metrics response truncated to 4 COMIDs; 15-row subset of /variable_info.
* gridmet/   tiny real subsets of gridMET pr_2024/pr_2026 yearly files serialised to JSON (packed
             uint16 + attributes; the repo ignores *.nc); slice attributes re-based in the 2026 subset.
* openmeteo/ ERA5 archive daily (Honolulu, 3 days) and Previous Runs hourly (GFS, 2 days, leads 1-2).
Small synthetic frames below (catalog rows, a toy flow network, hourly series) exist only inside
these tests. No network access.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data import sites as S
from sentinel.data.sources import gridmet as G
from sentinel.data.sources import nldi as N
from sentinel.data.sources import openmeteo as OM
from sentinel.data.sources import streamcat as SC

FX = Path(__file__).parent / "fixtures"


# =============================================================================== sites
def test_split_huc_strings_and_padding():
    assert S.split_huc("03130004") == ("03", "0313", "03130004", None)
    assert S.split_huc("070900020504") == ("07", "0709", "07090002", "070900020504")
    assert S.split_huc("") == (None, None, None, None)
    assert S.split_huc(None) == (None, None, None, None)
    assert S.split_huc("3130004") == (None, None, None, None)  # unpadded/odd -> refuse to guess


def _toy_catalog() -> pd.DataFrame:
    # deliberately unsorted site order + several series per site (regression test for the
    # positional-assignment bug that scrambled rows when site_id was an Index)
    rows = []
    for site, name, lat, lon, huc, st, parms in [
        (
            "05586300",
            "ILLINOIS RIVER AT FLORENCE, IL",
            "39.6327",
            "-90.6077",
            "07130011",
            "IL",
            ["00010", "00300", "00095"],
        ),
        ("01646500", "POTOMAC RIVER NEAR WASH, DC", "38.9497", "-77.1276", "02070008", "MD", ["00010"]),
        ("023432415", "CHATTAHOOCHEE R", "31.6222", "-85.0592", "03130004", "AL", ["00010", "00400"]),
    ]:
        for p in parms:
            rows.append(
                {
                    "agency_cd": "USGS",
                    "site_no": site,
                    "station_nm": name,
                    "site_tp_cd": "ST",
                    "dec_lat_va": lat,
                    "dec_long_va": lon,
                    "coord_acy_cd": "S",
                    "dec_coord_datum_cd": "NAD83",
                    "alt_va": "100",
                    "huc_cd": huc,
                    "data_type_cd": "dv",
                    "parm_cd": p,
                    "stat_cd": "00003",
                    "ts_id": "1",
                    "loc_web_ds": "",
                    "begin_date": "2010-01-01",
                    "end_date": "2026-10-01",
                    "count_nu": "100",
                    "state": st,
                }
            )
    return pd.DataFrame(rows)


def test_usgs_sites_from_catalog_rows_stay_aligned():
    u = S.usgs_sites_from_catalog(_toy_catalog(), ["00010", "00300", "00095", "00400"])
    u = u.set_index("site_id")
    assert u.loc["05586300", "name"].startswith("ILLINOIS")
    assert u.loc["01646500", "name"].startswith("POTOMAC")
    assert u.loc["023432415", "lat"] == pytest.approx(31.6222)
    assert u.loc["05586300", "n_wq_series"] == 3
    assert bool(u.loc["05586300", "has_do"]) and not bool(u.loc["01646500", "has_do"])
    assert bool(u.loc["023432415", "has_ph"])


def test_build_registry_schema_keys_and_neon():
    u = S.usgs_sites_from_catalog(_toy_catalog(), ["00010", "00300", "00095", "00400"])
    info = pd.DataFrame(
        {
            "site_no": ["05586300", "01646500", "023432415"],
            "tz_cd": ["CST", "EST", "EST"],
            "alt_va": ["404.5", "", "0.01"],
            "alt_datum_cd": ["NAVD88", "", "NGVD29"],
            "drain_area_va": ["26743", "11560", ""],
        }
    )
    neon = pd.DataFrame(
        [
            {
                "site_id": "CRAM",
                "name": "Crampton Lake NEON",
                "lat": 46.21,
                "lon": -89.47,
                "coord_source": "NEON buoy",
                "coord_datum": "WGS84",
                "habitat": "Aquatic Lake",
                "state": "WI",
                "elev_m": 509.0,
            },
            {
                "site_id": "CARI",
                "name": "Caribou Creek NEON",
                "lat": 65.15,
                "lon": -147.50,
                "coord_source": "NEON downstream sensor station (S2)",
                "coord_datum": "WGS84",
                "habitat": "Aquatic Wadeable Stream",
                "state": "AK",
                "elev_m": 225.4,
            },
        ]
    )
    reg = S.build_registry(
        u, info, neon, wbd_lookup=lambda lon, lat: "190803090603" if lat > 60 else "040201020102"
    )
    assert list(reg.columns) == S.REGISTRY_COLUMNS
    r = reg.set_index("site_key")
    assert set(r.index) == {"USGS-05586300", "USGS-01646500", "USGS-023432415", "NEON-CRAM", "NEON-CARI"}
    assert r.loc["USGS-023432415", "huc8"] == "03130004" and r.loc["USGS-023432415", "huc2"] == "03"
    assert r.loc["NEON-CRAM", "site_type"] == "LK" and r.loc["NEON-CARI", "site_type"] == "ST"
    assert (
        r.loc["NEON-CARI", "huc12"] == "190803090603"
        and r.loc["NEON-CARI", "huc_source"] == "wbd_point_huc12"
    )
    assert r.loc["USGS-05586300", "utc_offset_h"] == -6 and r.loc["NEON-CARI", "utc_offset_h"] == -9
    assert r.loc["USGS-05586300", "elev_m"] == pytest.approx(404.5 * 0.3048, abs=0.01)
    assert r.loc["USGS-05586300", "drain_area_km2"] == pytest.approx(26743 * 2.589988, rel=1e-6)
    assert bool(r.loc["USGS-05586300", "in_conus"]) and not bool(r.loc["NEON-CARI", "in_conus"])
    assert pd.api.types.is_string_dtype(reg["huc8"]) and pd.api.types.is_string_dtype(reg["site_id"])


# =============================================================================== NLDI
def test_parse_nwissite_feature():
    f = N.parse_feature(json.loads((FX / "nldi" / "nwissite_USGS-05586300.json").read_text()))
    assert f["comid"] == "3607116" and isinstance(f["comid"], str)
    assert f["reachcode"] == "07130011002494" and f["measure"] == pytest.approx(74.1997)
    assert f["nldi_type"] == "hydrolocation" and f["nldi_source"] == "nwissite"


def test_parse_position_feature_uses_identifier():
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[-90.6, 39.6], [-90.6, 39.61]]},
                "properties": {
                    "identifier": "3607116",
                    "source": "comid",
                    "sourceName": "NHDPlus comid",
                    "comid": 3607116,
                },
                "id": 3607116,
            }
        ],
    }
    f = N.parse_feature(payload)
    assert f["comid"] == "3607116" and f["measure"] is None
    assert N.parse_feature(None)["comid"] is None


def test_parse_navigation_and_404_detail():
    nav = N.parse_navigation(
        json.loads((FX / "nldi" / "nav_DM_nwissite_USGS-05586300_first8.json").read_text())
    )
    assert len(nav) == 8 and nav["identifier"].str.startswith("USGS-").all()
    assert nav["comid"].map(type).eq(str).all()
    assert json.loads((FX / "nldi" / "nwissite_404_USGS-331915112400601.json").read_text())["status"] == 404


def test_network_position_measure_convention():
    assert N.network_position_km(100.0, 2.0, 0.0) == (100.0, "vaa_pathlength_measure")  # downstream end
    assert N.network_position_km(100.0, 2.0, 100.0)[0] == pytest.approx(102.0)  # upstream end
    assert N.network_position_km(100.0, 2.0, None) == (101.0, "vaa_pathlength_midpoint")


def test_navigation_request_shape():
    url, p = N.navigation_request("nwissite", "USGS-05586300", "DM", "nwissite", 500)
    assert url.endswith("/nwissite/USGS-05586300/navigation/DM/nwissite") and p["distance"] == "500"
    with pytest.raises(ValueError):
        N.navigation_request("nwissite", "USGS-1", "XX")


def test_nldi_client_serves_cache_without_network(tmp_path, monkeypatch):
    url, params = N.feature_request("nwissite", "USGS-05586300")
    key = N._cache_key(url, params)
    cp = tmp_path / key[:2] / f"{key}.json.gz"
    cp.parent.mkdir(parents=True)
    body = json.loads((FX / "nldi" / "nwissite_USGS-05586300.json").read_text())
    with gzip.open(cp, "wt") as fh:
        json.dump({"status": 200, "body": body}, fh)
    c = N.NLDIClient(tmp_path, min_interval=0)
    monkeypatch.setattr(c, "_session", lambda: (_ for _ in ()).throw(AssertionError("network used")))
    feat, st = c.feature("nwissite", "USGS-05586300")
    assert st == 200 and feat["comid"] == "3607116" and c.stats["cache_hits"] == 1


# =============================================================================== flow network
def _toy_network():
    # main stem: hydroseq 1 (outlet) <- 2 <- 3 <- 4 ; tributary 5 -> 3. pathlength = km to outlet
    # at the downstream end of each flowline.
    vaa = pd.DataFrame(
        {
            "comid": [101, 102, 103, 104, 105],
            "hydroseq": [1, 2, 3, 4, 5],
            "dnhydroseq": [0, 1, 2, 3, 3],
            "levelpathi": [1, 1, 1, 1, 5],
            "terminalpa": [1] * 5,
            "pathlength": [0.0, 10.0, 30.0, 60.0, 40.0],
            "lengthkm": [10.0, 20.0, 30.0, 40.0, 25.0],
            "totdasqkm": [500.0, 400, 300, 100, 50],
            "streamorde": [5, 5, 4, 3, 2],
            "slope": [1e-4] * 5,
            "pathtimema": [0.0, 0.5, 1.5, 3.0, 2.0],
            "ftype": ["StreamRiver"] * 5,
            "vpuid": ["07"] * 5,
        }
    )
    links = pd.DataFrame(
        {
            "site_key": ["USGS-A", "USGS-B", "USGS-C", "USGS-T", "NEON-X"],
            "comid": ["104", "102", "102", "105", "101"],
            "measure": [50.0, 80.0, 10.0, 0.0, np.nan],
            "comid_method": ["nldi_nwissite"] * 4 + ["nldi_position"],
        }
    )
    return vaa, S.attach_vaa(links, vaa)


def test_attach_vaa_positions():
    _, nodes = _toy_network()
    p = nodes.set_index("site_key")["pos_km_to_outlet"]
    assert p["USGS-A"] == pytest.approx(60 + 40 * 0.5)
    assert p["USGS-B"] == pytest.approx(10 + 20 * 0.8) and p["USGS-C"] == pytest.approx(10 + 20 * 0.1)
    assert p["NEON-X"] == pytest.approx(5.0)  # no measure -> midpoint
    assert nodes.set_index("site_key").loc["NEON-X", "pos_method"] == "vaa_pathlength_midpoint"


def test_vaa_downstream_edges_and_combine():
    vaa, nodes = _toy_network()
    ve = S.vaa_downstream_edges(nodes, vaa, distance_km=500)
    pairs = set(zip(ve["src"], ve["dst"]))
    assert ("USGS-A", "USGS-B") in pairs and ("USGS-B", "USGS-C") in pairs and ("USGS-T", "USGS-C") in pairs
    assert ("USGS-C", "USGS-B") not in pairs  # same flowline, C is downstream of B
    assert ("USGS-A", "USGS-T") not in pairs and ("USGS-T", "USGS-A") not in pairs  # tributary vs main stem
    short = S.vaa_downstream_edges(nodes, vaa, distance_km=30)
    assert ("USGS-A", "USGS-B") not in set(zip(short["src"], short["dst"]))  # 80 - 26 = 54 km > 30
    nav = pd.DataFrame(
        {"query_site_key": ["USGS-B", "USGS-A"], "identifier": ["USGS-A", "USGS-B"], "mode": ["UM", "DM"]}
    )
    e = S.combine_edges(nodes, nav, ve, set(nodes["site_key"]))
    row = e.set_index(["src", "dst"]).loc[("USGS-A", "USGS-B")]
    assert row["found_by"] == "nldi_DM+nldi_UM+vaa_dm"
    assert row["distance_km"] == pytest.approx(80 - 26) and row["direction_check"] == "ok"
    assert row["travel_time_days"] == pytest.approx(3.0 - 0.5)
    groups = S.flow_components(list(nodes["site_key"]), e)
    assert groups["USGS-A"] == groups["USGS-T"] == groups["NEON-X"]


# =============================================================================== StreamCat
def test_streamcat_parse_and_columns():
    rec = json.loads((FX / "streamcat" / "metrics_post_4comids.json").read_text())
    df = SC.parse_metrics(rec["response"])
    assert len(df) == 4 and df["comid"].map(type).eq(str).all()
    exp = SC.expected_columns(rec["request"]["name"].split(","), "cat,ws")
    assert set(exp) <= set(df.columns)
    assert SC.metric_stem("pctcrop2019wsrp100") == ("pctcrop2019", "wsrp100")
    assert SC.metric_stem("iwi") == ("iwi", "other")
    assert SC.expected_columns(["iwi"], "other") == ["iwi"]


def test_streamcat_dictionary_units_and_years():
    vi = pd.read_csv(FX / "streamcat" / "variable_info_subset.csv")
    d = SC.column_dictionary(
        ["pctcrop2019ws", "rddenscat", "n_tin_2017ws", "precip9120cat", "iwi", "elevws"], vi
    )
    d = d.set_index("column")
    assert d.loc["pctcrop2019ws", "units"] == "Percent" and d.loc["pctcrop2019ws", "year"] == "2019"
    assert d.loc["rddenscat", "units"] == "Kilometer/Square Kilometer"  # non-year (active-data) entry
    assert d.loc["n_tin_2017ws", "units"] == "Kilograms" and d.loc["n_tin_2017ws", "year"] == "2017"
    assert d.loc["precip9120cat", "units"] == "Millimeters" and d.loc["elevws", "units"] == "Meters"
    assert d.loc["iwi", "aoi"] == "other"


def test_streamcat_riparian_requests_omit_area_flags():
    _, d = SC.metrics_request(["pctimp2019"], ["3607116"], "catrp100,wsrp100")
    assert "showareasqkm" not in d and "showpctfull" not in d  # API 403 with these flags (2026-10-22)
    _, d = SC.metrics_request(["pctimp2019"], ["3607116"], "cat,ws")
    assert d["showareasqkm"] == "true" and d["comid"] == "3607116"
    # National Nutrient Inventory metrics fail (ORA-00923) with the flags; 'other' AOI too
    assert not SC.wants_flags(["n_tin_2017"], "cat,ws") and not SC.wants_flags(["iwi"], "other")
    assert SC.wants_flags(["nani", "fert"], "cat,ws")


def test_streamcat_rejects_excluded_metrics():
    with pytest.raises(ValueError):
        SC.metrics_request(["mast2014"], ["1"], "other")
    names = [n for _, _, ns in SC.CURATED for n in ns]
    assert not any(n.startswith(tuple(SC.EXCLUDED)) for n in names)
    n_cols = sum(len(SC.expected_columns(ns, aoi)) for _, aoi, ns in SC.CURATED)
    assert 100 <= n_cols <= 200


# =============================================================================== gridMET
def _nc_from_json(name: str, tmp_path: Path) -> Path:
    import netCDF4

    spec = json.loads((FX / "gridmet" / name).read_text())
    out = tmp_path / name.replace(".json", ".nc")
    with netCDF4.Dataset(out, "w") as d:
        for k, n in spec["dims"].items():
            d.createDimension(k, n)
        for vname, v in spec["variables"].items():
            attrs = dict(v["attrs"])
            fill = attrs.pop("_FillValue", None)
            dt = np.dtype(v["dtype"])
            x = d.createVariable(
                vname, dt, tuple(v["dims"]), fill_value=None if fill is None else dt.type(fill)
            )
            x.set_auto_maskandscale(False)
            for a, val in attrs.items():
                x.setncattr(a, val)
            x[:] = np.asarray(v["data"], dtype=dt)
        for a, val in spec["global_attrs"].items():
            d.setncattr(a, val)
    return out


def test_gridmet_read_applies_packing(tmp_path):
    g = G.read_grid(_nc_from_json("pr_2024_florence_subset.json", tmp_path))
    assert g["var_name"] == "precipitation_amount" and g["units"] == "mm"
    i = int(np.argmin(abs(g["lat"] - 39.65)))
    j = int(np.argmin(abs(g["lon"] + 90.60)))
    day = list(g["dates"].strftime("%Y-%m-%d")).index("2024-01-08")
    assert g["data"][day, i, j] == pytest.approx(21.5, abs=1e-4)  # packed 215 x 0.1 (NCSS CSV returns 215)
    assert (g["status"] == "permanent").all()


def test_gridmet_fill_value_and_nearest_valid(tmp_path):
    g = G.read_grid(_nc_from_json("pr_2024_galveston_subset.json", tmp_path))
    valid = ~np.isnan(g["data"][0])
    assert (~valid).sum() == 1
    mi, mj = map(int, np.argwhere(~valid)[0])
    pts = G.nearest_valid_cells(
        np.array([g["lat"][mi], 10.0]),
        np.array([g["lon"][mj], -150.0]),
        g["lat"],
        g["lon"],
        valid,
        max_radius_cells=2,
    )
    assert (pts.loc[0, "i"], pts.loc[0, "j"]) != (mi, mj) and valid[pts.loc[0, "i"], pts.loc[0, "j"]]
    assert 0 < pts.loc[0, "cell_dist_km"] < 6.5
    assert pts.loc[1, "i"] == -1 and np.isnan(pts.loc[1, "cell_dist_km"])  # outside the grid
    vals = G.extract_cells(g, pts)
    assert np.isnan(vals[:, 1]).all() and np.isfinite(vals[:, 0]).all()


def test_gridmet_status_flags(tmp_path):
    g = G.read_grid(_nc_from_json("pr_2026_status_subset.json", tmp_path))
    assert list(g["status"][:4]) == ["permanent"] * 4 and (g["status"][4:] == "provisional").all()


def test_gridmet_units_and_urls():
    assert G.file_url("tmmx", 2024).endswith("/metdata/data/tmmx_2024.nc")
    np.testing.assert_allclose(G.to_harmonised("tmmx", np.array([273.15, 300.0])), [0.0, 26.85])
    assert G.to_harmonised("pr", np.array([1.5]))[0] == 1.5
    assert {v[2] for v in G.VARIABLES.values()} == {
        "precip_mm",
        "tmax_c",
        "tmin_c",
        "srad_wm2",
        "wind_ms",
        "rhmax_pct",
        "rhmin_pct",
        "sph_kgkg",
        "vpd_kpa",
        "pet_mm",
    }
    with pytest.raises(KeyError):
        G.file_url("tmax", 2024)


# =============================================================================== Open-Meteo
def test_call_weight_matches_pricing_examples():
    assert OM.call_weight(15, 14) == pytest.approx(1.5)  # "2 weeks, 15 variables = 1.5 calls"
    assert OM.call_weight(15, 28) == pytest.approx(3.0)  # "4 weeks ... 3.0 calls"
    assert OM.call_weight(10, 14) == pytest.approx(1.0)
    assert OM.call_weight(3, 3) == pytest.approx(1.0)  # minimum one call
    assert OM.call_weight(9, 365) == pytest.approx(0.9 * 365 / 14)
    assert OM.call_weight(10, 14, n_locations=3) == pytest.approx(3.0)


def test_timezone_and_chunks_and_cells():
    assert (
        OM.etc_timezone(-9) == "Etc/GMT+9"
        and OM.etc_timezone(10) == "Etc/GMT-10"
        and OM.etc_timezone(0) == "GMT"
    )
    ch = OM.year_chunks("2000-03-01", "2002-02-10")
    assert ch == [("2000-03-01", "2000-12-31"), ("2001-01-01", "2001-12-31"), ("2002-01-01", "2002-02-10")]
    assert OM.cell_id(21.3069, -157.8583) == OM.cell_id(21.31, -157.86) != OM.cell_id(21.45, -157.86)


def test_era5_fixture_harmonised_units():
    rec = json.loads((FX / "openmeteo" / "era5_daily_honolulu_2024-01-01_03.json").read_text())
    df, meta = OM.parse_daily(rec["response"])
    assert (
        meta["units"]["shortwave_radiation_sum"] == "MJ/m²" and meta["units"]["wind_speed_10m_mean"] == "m/s"
    )
    h = OM.era5_to_harmonised(df, meta["elevation"])
    np.testing.assert_allclose(h["srad_wm2"], df["shortwave_radiation_sum"] * 1e6 / 86400)
    assert (h["vpd_kpa"] >= 0).all() and h["sph_kgkg"].between(0.003, 0.03).all()
    assert (h["tmax_c"] >= h["tmin_c"]).all() and len(h) == 3


def test_fao56_helpers():
    assert OM.sat_vp_kpa(20.0) == pytest.approx(2.338, abs=1e-3)  # FAO-56 Annex 2 table
    assert OM.pressure_kpa_from_elev(1800) == pytest.approx(81.8, abs=0.1)  # FAO-56 example 2
    assert OM.vpd_fao56(25.0, 15.0, 30.0) == 0.0  # floored at zero
    q = OM.specific_humidity_from_dewpoint(15.0, 101.3)
    assert q == pytest.approx(0.0106, abs=3e-4)


def test_previous_runs_fixture_aggregation():
    rec = json.loads((FX / "openmeteo" / "previous_runs_gfs_florence_2024-07-01_02.json").read_text())
    hourly, meta = OM.parse_hourly(rec["response"])
    assert len(hourly) == 48 and hourly["time_utc"].dt.tz is not None
    agg = OM.aggregate_previous_runs(hourly, utc_offset_h=0, leads=(1, 2))
    a1 = agg[(agg["lead_day"] == 1)].set_index("date_local")
    d = pd.Timestamp("2024-07-01").date()
    t = hourly["temperature_2m_previous_day1"].iloc[:24]
    assert a1.loc[d, "tmax_c"] == pytest.approx(t.max()) and a1.loc[d, "n_hours_t"] == 24
    assert a1.loc[d, "max_init_utc"] == pd.Timestamp("2024-06-30 18:00", tz="UTC")
    # local CST days: the first/last local days are incomplete -> NaN, never partial sums
    loc = OM.aggregate_previous_runs(hourly, utc_offset_h=-6, leads=(1,))
    assert loc["precip_mm"].isna().sum() >= 1 and (loc.loc[loc["n_hours_p"] < 24, "precip_mm"].isna()).all()


def test_previous_runs_issue_time_bound():
    # synthetic hourly series: every lead-N daily value must come from runs initialised before
    # local midnight that starts day (L - N + 1)
    t = pd.date_range("2024-03-01", "2024-03-12", freq="h", tz="UTC")
    cols = {
        f"{v}_previous_day{n}": np.arange(len(t), dtype=float)
        for v in OM.NWP_HOURLY_VARS
        for n in range(1, 8)
    }
    hourly = pd.DataFrame({"time_utc": t, **cols})
    for off in (-5, -8, -10, 0):
        agg = OM.aggregate_previous_runs(hourly, utc_offset_h=off)
        issue_local_midnight = pd.to_datetime(agg["date_local"]) - pd.to_timedelta(
            agg["lead_day"] - 1, unit="D"
        )
        issue_utc = (issue_local_midnight - pd.Timedelta(hours=off)).dt.tz_localize("UTC")
        assert (agg["max_init_utc"] < issue_utc).all()


def test_usable_at_filters_by_init_plus_latency():
    t = pd.date_range("2024-03-01", "2024-03-05", freq="h", tz="UTC")
    cols = {f"{v}_previous_day{n}": np.ones(len(t)) for v in OM.NWP_HOURLY_VARS for n in range(1, 8)}
    agg = OM.aggregate_previous_runs(pd.DataFrame({"time_utc": t, **cols}), utc_offset_h=-6)
    issue = pd.Timestamp("2024-03-03 00:00", tz="UTC")
    ok = OM.usable_at(agg, issue, latency_h=6)
    assert len(ok) and (ok["max_init_utc"] + pd.Timedelta(hours=6) <= issue).all()
    assert len(OM.usable_at(agg, "2024-03-03 00:00", latency_h=6)) == len(ok)  # naive -> UTC


def test_request_builders_and_weights():
    url, p = OM.previous_runs_request(39.6, -90.6, "2024-01-01", "2024-12-31", "gfs_global")
    assert url == OM.PREVIOUS_RUNS_URL and len(p["hourly"].split(",")) == 21 and p["timezone"] == "GMT"
    assert OM.request_weight(url, p) == pytest.approx(2.1 * 366 / 14)
    url, p = OM.era5_daily_request(61.2, -149.9, "2024-01-01", "2024-01-14", -9)
    assert (
        p["timezone"] == "Etc/GMT+9" and p["models"] == "era5_seamless" and OM.request_weight(url, p) == 1.0
    )


def test_budget_ledger_caps(tmp_path):
    led = OM.BudgetLedger(tmp_path / "ledger.jsonl", day_cap=50, hour_cap=40, minute_cap=30)
    led.reserve(25, "a")
    assert led.used()["day"] == pytest.approx(25)
    with pytest.raises(ValueError):
        led.reserve(31, "too big for one minute")
    with open(led.path, "a") as fh:  # pretend older usage inside the 24 h window
        fh.write(json.dumps({"t": __import__("time").time() - 7200, "w": 20, "tag": "old"}) + "\n")
    with pytest.raises(OM.DailyLimitReached):
        led.reserve(10, "b")
    assert 0 < led.seconds_until_day_room(10) <= 86400
