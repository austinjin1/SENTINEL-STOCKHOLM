# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared physical constants of the M6 architecture-v2 round (``sentinel/hydromet_sota/gso_mt/constants.py``; lead
rulings §2.3, lead note X7): one module, a source per value, and a unit test per value.

Each test compares the module with the number **as printed in the cited source** (read on 2026-11-10) or re-computes
a worked example of that source. The sources and the places are in the module; this file repeats the printed
numbers independently, so a change of the module alone fails here.
"""

import io
import math
import tokenize
from pathlib import Path

import numpy as np
import pytest

from sentinel.hydromet_sota.gso_mt import constants as K
from sentinel.hydromet_sota.gso_mt import solar as S

ROOT = Path(K.__file__).resolve().parents[3]

# name -> value as printed in the source (typed here from the source text, not from the module)
PRINTED = {
    "GSC_MJ_M2_MIN": 0.0820, "DR_AMPLITUDE": 0.033, "DECLINATION_AMPLITUDE": 0.409, "DECLINATION_PHASE": 1.39,
    "DAYS_PER_YEAR": 365, "RSO_A": 0.75, "RSO_B": 2e-5,                                   # FAO-56 eq. 21, 23, 24, 37
    "SVP_A": 0.6108, "SVP_B": 17.27, "SVP_C": 237.3, "SVP_SLOPE": 4098, "PSYCHROMETRIC": 0.665e-3,   # eq. 11, 13, 8
    "EPSILON": 0.622, "WIND_Z_A": 4.87, "WIND_Z_B": 67.8, "WIND_Z_C": 5.42, "WIND_HEIGHT_M": 10,     # eq. 8, 47
    "WIND_FN_A_MJ": 6.43, "WIND_FN_B": 0.536,                                             # Shuttleworth 2007 eq. 1
    "BRUTSAERT_A": 1.24, "BRUTSAERT_EXPONENT": 1 / 7,                                     # Crawford & Duchon eq. 16
    "EMIS_WATER": 0.97, "ALBEDO_WATER": 0.08,                                             # Jobson 1978; McMahon 2013
    "SIGMA": 5.670374419e-8,                                                              # NIST CODATA 2018
    "KPA_PER_ATM": 101.325, "KELVIN_OFFSET": 273.15, "HPA_PER_KPA": 10, "SECONDS_PER_DAY": 86400, "DEG_PER_HOUR": 15,
    "RHO_CP_WATER": 4.184e6,
}


def test_every_constant_has_a_source_a_place_a_status_and_the_printed_value():
    assert set(K.C) == set(PRINTED)
    for name, c in K.C.items():
        assert c.value == pytest.approx(PRINTED[name], rel=1e-12), name
        assert getattr(K, name) == c.value
        assert c.source and c.where and c.printed and c.unit and c.used_by, name
        assert c.status in ("verified", "secondary", "definition", "nominal"), name
    assert len(K.table()) == len(K.C)
    # honest status: what was not read in the original is not called verified
    assert K.C["BRUTSAERT_A"].status == "secondary" and K.C["RHO_CP_WATER"].status == "nominal"
    assert K.FORMULAS["equilibrium temperature and exchange coefficient"][3] == "secondary"
    assert all(len(v) == 4 and v[3] in ("verified", "secondary", "definition") for v in K.FORMULAS.values())
    # nothing released uses a nominal value
    assert [k for k, c in K.C.items() if c.status == "nominal"] == ["RHO_CP_WATER"]
    assert "not released" in K.C["RHO_CP_WATER"].used_by


def test_derived_values():
    assert K.GSC_WM2 == pytest.approx(1366.67, abs=0.01)                    # 0.0820 MJ m-2 min-1
    assert K.WIND_FN_A_W == pytest.approx(74.42, abs=0.01)                  # 6.43 MJ m-2 d-1 kPa-1
    assert K.WIND_10M_TO_2M == pytest.approx(0.748, abs=0.0005)             # FAO-56 Example 14: "= 0.75"
    assert K.WIND_10M_TO_2M * 3.2 == pytest.approx(2.4, abs=0.01)           # FAO-56 Example 14: u2 = 2.4 m/s
    assert K.HOURS_PER_RADIAN * 1.527 == pytest.approx(11.7, abs=0.05)      # FAO-56 Example 9: N = 11.7 h
    assert 1.0 - K.EPSILON == pytest.approx(0.378)


def test_fao56_example_8_and_9_extraterrestrial_radiation_and_daylight_hours():
    """FAO-56 Example 8: 3 September (J = 246) at 20 deg S: dr = 0.985, delta = 0.120 rad, omega_s = 1.527 rad,
    Ra = 32.2 MJ m-2 d-1; Example 9: N = 11.7 h."""
    assert 1.0 + K.DR_AMPLITUDE * math.cos(2 * math.pi * 246 / K.DAYS_PER_YEAR) == pytest.approx(0.985, abs=0.0005)
    assert float(S.declination(246)) == pytest.approx(0.120, abs=0.0005)
    ch = S.solar_channels(np.array([-20.0]), np.array([0.0]), np.array([246]), np.array([[200.0]]))[0, 0]
    photo, rso, cloud, noon = (float(x) for x in ch)
    assert photo == pytest.approx(11.7, abs=0.05)
    ra_mj = rso / K.RSO_A * 86400 / 1e6                                     # Rso = 0.75 Ra at z = 0
    assert ra_mj == pytest.approx(32.2, abs=0.05)
    assert cloud == pytest.approx(200.0 / rso, rel=1e-5)
    assert noon == pytest.approx(90.0 - abs(-20.0 - math.degrees(0.120)), abs=0.05)


def test_clear_sky_radiation_elevation_term_and_local_solar_hour():
    lo = S.solar_channels(np.array([40.0, 40.0]), np.array([0.0, 2000.0]), np.array([172]), np.full((2, 1), 250.0))
    assert lo[1, 0, 1] / lo[0, 0, 1] == pytest.approx((0.75 + 2e-5 * 2000.0) / 0.75, rel=1e-5)     # FAO-56 eq. 37
    assert S.local_solar_hour(12.0, -90.0) == pytest.approx(6.0)             # 15 degrees of longitude per hour
    assert S.local_solar_hour(23.0, 30.0) == pytest.approx(1.0)


def test_humidity_and_wind_function_against_the_sources():
    from sentinel.hydromet_sota.thermox import physics as TP

    assert TP.sat_vp_kpa(np.array([24.5]))[0] == pytest.approx(3.075, abs=0.001)          # FAO-56 Example 3
    assert TP.psychrometric(81.8) == pytest.approx(0.054, abs=0.0005)                     # FAO-56 Example 2
    assert TP.vp_from_sph(0.010, 100.0) == pytest.approx(0.010 * 100.0 / (0.622 + 0.378 * 0.010), rel=1e-12)
    # Shuttleworth 2007 eq. 1: 6.43 (1 + 0.536 U) D, MJ m-2 d-1 for D in kPa; here in W m-2 kPa-1
    for u in (0.0, 1.0, 3.5):
        assert TP.wind_function(u) * 86400 / 1e6 == pytest.approx(6.43 * (1 + 0.536 * u), rel=1e-12)
    # Crawford & Duchon 1999 eq. 16 with e in millibars and T in kelvin; eq. 3 / 11 for the cloud fraction
    e_kpa, t_c = 1.5, 14.85
    eps_c = 1.24 * (15.0 / 288.0) ** (1 / 7)
    assert TP.emissivity_clear(e_kpa, t_c) == pytest.approx(eps_c, rel=1e-9)
    for s_ratio, clf in ((1.0, 0.0), (0.6, 0.4), (0.0, 1.0), (1.2, 0.0)):                  # clf = 1 - s, not below 0
        assert TP.emissivity_all_sky(e_kpa, t_c, s_ratio) == pytest.approx(clf + (1 - clf) * eps_c, rel=1e-9)
    assert (TP.SIGMA, TP.EMIS_WATER, TP.ALBEDO_WATER) == (5.670374419e-8, 0.97, 0.08)


def _numbers(path):
    out = set()
    for tok in tokenize.generate_tokens(io.StringIO(Path(path).read_text()).readline):
        if tok.type == tokenize.NUMBER:
            try:
                out.add(float(tok.string))
            except ValueError:
                pass
    return out


def test_no_private_copy_of_a_constant_in_the_arch_v2_code():
    """X7: the arms call the shared module. No arch-v2 file other than ``constants.py`` holds one of the distinctive
    numbers as a literal in code (comments and docstrings are not code)."""
    distinctive = {c.value for k, c in K.C.items()
                   if k not in ("DAYS_PER_YEAR", "RSO_A", "WIND_HEIGHT_M", "HPA_PER_KPA", "DEG_PER_HOUR",
                                "SECONDS_PER_DAY", "ALBEDO_WATER", "DR_AMPLITUDE")}
    distinctive |= {0.378, 1366.7, 0.0820e6, 6.43e6}
    files = [f for f in sorted((ROOT / "sentinel" / "hydromet_sota").glob("*/*.py")) if f.name != "constants.py"]
    files += sorted((ROOT / "scripts").glob("*/sota_hydromet_v2_*.py"))
    assert len(files) > 8
    bad = {f.name: sorted(_numbers(f) & distinctive) for f in files if _numbers(f) & distinctive}
    assert not bad, bad
    for f in ("gso_mt/solar.py", "thermox/physics.py"):
        t = (ROOT / "sentinel" / "hydromet_sota" / f).read_text()
        assert "constants as K" in t, f


def test_do_saturation_and_pressure_are_the_frozen_shared_routines():
    from sentinel.models.hydromet import physics as HP

    assert K.do_sat_mgl is HP.do_sat_mgl and K.pressure_atm_from_elevation is HP.pressure_atm_from_elevation
    np.testing.assert_allclose(K.do_sat_mgl(np.array([0.0, 10.0, 20.0, 30.0])), [14.62, 11.29, 9.09, 7.56], atol=0.01)
