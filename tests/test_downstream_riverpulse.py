# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for D8 RiverPulse (synthetic arrays only; split checks skip without local data)."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from sentinel.models.downstream import common as C
from sentinel.models.downstream import riverpulse as R


@pytest.mark.parametrize("col", ["do_obs_mgl", "temp_water_c", "discharge_m3s", "depth_m", "shortwave_wm2",
                                 "hydgeom_c", "k600", "do_amp_mgl"])
def test_forbidden_label_model_inputs_rejected(col):
    with pytest.raises(AssertionError):
        R.assert_no_forbidden(["doy_sin", col])


def test_allowed_feature_names_pass():
    R.assert_no_forbidden(["doy_sin", "lat", "daylen_h", *[f"wx_{c}" for c in R.WX_KEEP], "sc_pctcrop2019ws"])
    R.assert_no_forbidden(list(R.LIGHT_COLS))


def test_day_length():
    assert abs(R.day_length_h(0.0, 80) - 12.1) < 0.2
    assert 15.0 < R.day_length_h(45.0, 172) < 16.0
    assert 8.5 < R.day_length_h(45.0, 355) < 9.2


def test_doy_climatology_and_site_year_means():
    d = pd.DataFrame({"site_id": "s", "date": pd.date_range("2010-01-01", "2011-12-31", freq="D")})
    d["gpp"] = 2 + np.sin(2 * np.pi * d["date"].dt.dayofyear / 365.25)
    clim = R.doy_climatology_fit(d, "gpp")
    assert clim.notna().all() and abs(clim.loc[91] - 3.0) < 0.1
    sy = R.site_year_means(d, d["gpp"].to_numpy(), "gpp")
    assert len(sy) == 2 and np.allclose(sy["obs"], sy["pred"])


def test_roles_copied_from_sonde_split():
    if not R.SONDE_SPLIT.exists():
        pytest.skip("no sonde split")
    try:
        sites = R.load_sites()
    except FileNotFoundError:
        pytest.skip("no Powell labels locally")
    role, prov = R.site_roles(sites)
    sp = json.loads(R.SONDE_SPLIT.read_text())["site_role"]
    for sid, nwis in zip(sites["site_id"], sites["nwis_id"]):
        if prov[sid].startswith("flow_override"):
            assert role[sid] == "train"
        elif nwis in sp:
            assert role[sid] == sp[nwis] and prov[sid] == "sonde_split_site"
        assert role[sid] in C.ROLES


def test_split_frozen_file_matches_protocol_sha():
    p = C.PREREG / R.SPLIT_FILE
    if not p.exists():
        pytest.skip("split not frozen in this checkout")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha == R.SPLIT_SHA and sha in (C.PREREG / "downstream_riverpulse.md").read_text()
    C.load_split("riverpulse", expect_sha=R.SPLIT_SHA, path=p)


def test_drop_protected_removes_test_sites():
    df = pd.DataFrame({"site_id": ["a", "b", "c"], "role": ["train", "val", "test"]})
    out = C.drop_protected(df)
    assert out["site_id"].tolist() == ["a", "b"]


def test_model_derived_streamcat_columns_excluded():
    try:
        sites = R.load_sites()
        sc = R.streamcat_for_sites(sites)
    except FileNotFoundError:
        pytest.skip("no local data")
    assert not set(R.STREAMCAT_EXCLUDE) & set(sc.columns)


def test_inner_groups_shared_and_deterministic():
    g = np.array([f"h{i % 10}" for i in range(100)])
    assert R.inner_groups(g) == R.inner_groups(g[::-1]) and len(R.inner_groups(g)) == 2
