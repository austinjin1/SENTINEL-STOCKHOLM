# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Physical constants of the M6 architecture-v2 round — **the one place they are written** (lead rulings §2.3,
"Physical constants"; lead note X7).

Every value used by the US input set v2 (solar channels), by the G2 reference's physics features (equilibrium
temperature, DO saturation at it) and by the arms D1 / D2 is defined here with its source, the place in the source
where it is printed, and a status:

* ``verified``   — read in the source itself on 2026-11-10 (text or equation image);
* ``secondary``  — read in a paper that reprints the formula of the original (original not accessible);
* ``definition`` — an exact definition or a unit identity;
* ``nominal``    — a rounded textbook value, no page checked (stated as such; used by nothing released).

``sentinel/hydromet_sota/gso_mt/solar.py`` and ``sentinel/hydromet_sota/thermox/physics.py`` import their numbers
from this module and hold no literal of their own (``tests/test_hydromet_sota_v2_constants.py`` checks it, checks
each value against the printed one and re-computes the worked examples of the sources).

Shared routines that stay where they are (frozen M6 code, not copied): DO saturation
``sentinel.models.hydromet.physics.do_sat_mgl`` (Benson & Krause 1984; table test in ``tests/test_hydromet.py``) and
the barometric pressure ``pressure_atm_from_elevation``; both are re-exported below so that an arm has one import.

Gas exchange (K600, Schmidt number) is **not** in this module: no released fit uses it. Raymond et al. 2012 eq. 1
could not be read from the paper on 2026-11-10 (publisher HTTP 403); it needs its own stamped deviations entry
before any DO-stage fit (rulings L4)."""

from __future__ import annotations

import dataclasses
import math

from ...models.hydromet.physics import do_sat_mgl, pressure_atm_from_elevation  # noqa: F401  (shared, frozen)

FAO56 = ("Allen, Pereira, Raes & Smith 1998, FAO Irrigation and Drainage Paper 56, chapter 3 "
         "(https://www.fao.org/4/x0490e/x0490e07.htm)")
SHUT07 = ("Shuttleworth 2007, Hydrol. Earth Syst. Sci. 11:210-244 "
          "(https://hess.copernicus.org/articles/11/210/2007/hess-11-210-2007.pdf)")
CD99 = ("Crawford & Duchon 1999, J. Appl. Meteorol. 38(4):474-480 "
        "(https://journals.ametsoc.org/view/journals/apme/38/4/1520-0450_1999_038_0474_aipfee_2.0.co_2.pdf)")
MCMAHON13 = ("McMahon, Peel, Lowe, Srikanthan & McVicar 2013, Hydrol. Earth Syst. Sci. 17:1331-1363 and supplement "
             "(https://hess.copernicus.org/articles/17/1331/2013/)")
JOBSON78 = "Jobson 1978, USGS Open-File Report 78-1026 (https://pubs.usgs.gov/of/1978/1026/report.pdf)"
NIST = "NIST CODATA 2018 (https://physics.nist.gov/cgi-bin/cuu/Value?sigma)"


@dataclasses.dataclass(frozen=True)
class Const:
    value: float
    unit: str
    source: str
    where: str            # equation / table / page as printed in the source
    printed: str          # the text as printed (what the unit test compares with)
    status: str           # verified | secondary | definition | nominal
    used_by: str


C = {
    # ---------------------------------------------------------------- solar geometry (US input set v2: 4 channels)
    "GSC_MJ_M2_MIN": Const(0.0820, "MJ m-2 min-1", FAO56, "eq. 21", "Gsc solar constant = 0.0820 MJ m-2 min-1",
                           "verified", "solar channels"),
    "DR_AMPLITUDE": Const(0.033, "-", FAO56, "eq. 23", "dr = 1 + 0.033 cos(2 pi J / 365)", "verified",
                          "solar channels"),
    "DECLINATION_AMPLITUDE": Const(0.409, "rad", FAO56, "eq. 24", "delta = 0.409 sin(2 pi J / 365 - 1.39)",
                                   "verified", "solar channels"),
    "DECLINATION_PHASE": Const(1.39, "rad", FAO56, "eq. 24", "delta = 0.409 sin(2 pi J / 365 - 1.39)", "verified",
                               "solar channels"),
    "DAYS_PER_YEAR": Const(365.0, "d", FAO56, "eq. 23, 24", "2 pi J / 365", "verified", "solar channels"),
    "RSO_A": Const(0.75, "-", FAO56, "eq. 37", "Rso = (0.75 + 2 10-5 z) Ra", "verified", "solar channels"),
    "RSO_B": Const(2e-5, "m-1", FAO56, "eq. 37", "Rso = (0.75 + 2 10-5 z) Ra", "verified", "solar channels"),
    # ------------------------------------------------- humidity, pressure, wind (G2 physics features; D2 if released)
    "SVP_A": Const(0.6108, "kPa", FAO56, "eq. 11", "e0(T) = 0.6108 exp[17.27 T / (T + 237.3)]", "verified",
                   "equilibrium temperature"),
    "SVP_B": Const(17.27, "-", FAO56, "eq. 11", "e0(T) = 0.6108 exp[17.27 T / (T + 237.3)]", "verified",
                   "equilibrium temperature"),
    "SVP_C": Const(237.3, "degC", FAO56, "eq. 11", "e0(T) = 0.6108 exp[17.27 T / (T + 237.3)]", "verified",
                   "equilibrium temperature"),
    "SVP_SLOPE": Const(4098.0, "degC", FAO56, "eq. 13", "Delta = 4098 [0.6108 exp(17.27 T/(T + 237.3))] / (T + 237.3)^2",
                       "verified", "equilibrium temperature"),
    "PSYCHROMETRIC": Const(0.665e-3, "degC-1", FAO56, "eq. 8", "gamma = cp P / (epsilon lambda) = 0.665 x 10-3 P",
                           "verified", "equilibrium temperature"),
    "EPSILON": Const(0.622, "-", FAO56, "eq. 8 (symbols)", "ratio molecular weight of water vapour/dry air = 0.622",
                     "verified", "vapour pressure from specific humidity"),
    "WIND_Z_A": Const(4.87, "-", FAO56, "eq. 47", "u2 = uz 4.87 / ln(67.8 z - 5.42)", "verified",
                      "equilibrium temperature"),
    "WIND_Z_B": Const(67.8, "m-1", FAO56, "eq. 47", "u2 = uz 4.87 / ln(67.8 z - 5.42)", "verified",
                      "equilibrium temperature"),
    "WIND_Z_C": Const(5.42, "-", FAO56, "eq. 47", "u2 = uz 4.87 / ln(67.8 z - 5.42)", "verified",
                      "equilibrium temperature"),
    "WIND_HEIGHT_M": Const(10.0, "m", "gridMET (Abatzoglou 2013): wind speed at 10 m", "variable definition (vs)",
                           "wind velocity at 10 m", "definition", "equilibrium temperature"),
    # ------------------------------------------------------------------------- surface energy balance of open water
    "WIND_FN_A_MJ": Const(6.43, "MJ m-2 d-1 kPa-1", SHUT07, "eq. 1, p. 211",
                          "E_Penman = {Delta A' + gamma [6.43 (1 + 0.536 U_m) D] / lambda} / (Delta + gamma)",
                          "verified", "equilibrium temperature"),
    "WIND_FN_B": Const(0.536, "s m-1", SHUT07, "eq. 1, p. 211",
                       "E_Penman = {Delta A' + gamma [6.43 (1 + 0.536 U_m) D] / lambda} / (Delta + gamma)",
                       "verified", "equilibrium temperature"),
    "BRUTSAERT_A": Const(1.24, "-", CD99 + ", reprinting Brutsaert 1975, Water Resour. Res. 11(5):742-744",
                         "eq. 16, p. 476", "epsilon = {clf + (1 - clf)[1.24 (e/T)^(1/7)]}; T in kelvin, e in millibars",
                         "secondary", "equilibrium temperature"),
    "BRUTSAERT_EXPONENT": Const(1.0 / 7.0, "-", CD99 + ", reprinting Brutsaert 1975", "eq. 16, p. 476",
                                "epsilon = {clf + (1 - clf)[1.24 (e/T)^(1/7)]}", "secondary",
                                "equilibrium temperature"),
    "EMIS_WATER": Const(0.97, "-", JOBSON78 + "; also " + MCMAHON13, "Jobson eq. 9, PDF p. 34; McMahon Table 2, p. 1340",
                        "emissivity for water (0.97)", "verified", "equilibrium temperature"),
    "ALBEDO_WATER": Const(0.08, "-", MCMAHON13, "supplement Table S3 (PDF p. 143) and Sect. S4 (PDF p. 27)",
                          "Water 0.08 Maidment (1992); for open-water alpha = 0.08", "verified",
                          "equilibrium temperature"),
    "SIGMA": Const(5.670374419e-8, "W m-2 K-4", NIST, "Stefan-Boltzmann constant",
                   "5.670 374 419... x 10-8 W m-2 K-4 (exact)", "verified", "equilibrium temperature"),
    # ------------------------------------------------------------------------------------- unit identities, nominal
    "KPA_PER_ATM": Const(101.325, "kPa atm-1", "definition of the standard atmosphere (101 325 Pa)", "-",
                         "1 atm = 101 325 Pa", "definition", "pressure of the frozen barometric routine in kPa"),
    "KELVIN_OFFSET": Const(273.15, "K", "definition of the Celsius scale", "-", "T/K = t/degC + 273.15", "definition",
                           "long-wave terms"),
    "HPA_PER_KPA": Const(10.0, "hPa kPa-1", "unit identity (1 mbar = 1 hPa)", "-", "1 kPa = 10 hPa", "definition",
                         "Brutsaert emissivity (e in millibars)"),
    "SECONDS_PER_DAY": Const(86400.0, "s d-1", "unit identity", "-", "1 d = 86 400 s", "definition",
                             "wind function in W m-2 kPa-1"),
    "DEG_PER_HOUR": Const(15.0, "deg h-1", "geometry (360 deg / 24 h)", "-", "360 / 24 = 15", "definition",
                          "local solar hour of a sample"),
    "RHO_CP_WATER": Const(4.184e6, "J m-3 K-1", "nominal: 1000 kg m-3 x 4184 J kg-1 K-1 (15 degC calorie)", "-",
                          "no page checked", "nominal", "D2 relaxation term only (not released)"),
}

globals().update({k: v.value for k, v in C.items()})          # GSC_MJ_M2_MIN, SVP_A, ... as plain module numbers

# ---- derived values (no new number)
GSC_WM2 = C["GSC_MJ_M2_MIN"].value * 1e6 / 60.0                                  # 1366.7 W m-2
WIND_FN_A_W = C["WIND_FN_A_MJ"].value * 1e6 / C["SECONDS_PER_DAY"].value         # 74.42 W m-2 kPa-1
WIND_10M_TO_2M = C["WIND_Z_A"].value / math.log(C["WIND_Z_B"].value * C["WIND_HEIGHT_M"].value - C["WIND_Z_C"].value)
HOURS_PER_RADIAN = 24.0 / math.pi                                                # FAO-56 eq. 34: N = (24 / pi) omega_s

FORMULAS = {
    "sunset hour angle": (FAO56, "eq. 25", "omega_s = arccos[-tan(phi) tan(delta)]", "verified"),
    "extraterrestrial radiation": (FAO56, "eq. 21",
                                   "Ra = (24 (60) / pi) Gsc dr [omega_s sin(phi) sin(delta) + cos(phi) cos(delta) "
                                   "sin(omega_s)]", "verified"),
    "daylight hours": (FAO56, "eq. 34", "N = (24 / pi) omega_s", "verified"),
    "cloud fraction": (CD99, "eq. 3, p. 475", "clf = 1 - s, s = measured / clear-sky solar irradiance; clf < 0 set "
                       "to 0", "verified"),
    "all-sky emissivity": (CD99, "eq. 11, p. 476", "LWd = [clf + (1 - clf) eps_c] sigma T^4", "verified"),
    "vapour pressure from specific humidity": ("definition of specific humidity with " + FAO56 + " epsilon",
                                               "-", "e = q P / (epsilon + (1 - epsilon) q)", "definition"),
    "equilibrium temperature and exchange coefficient": (
        "Edinger, Duttweiler & Geyer 1968, Water Resour. Res. 4(5):1137-1143 (doi:10.1029/WR004i005p01137)",
        "abstract (metadata record; the paper itself was not read)",
        "net rate in terms of a thermal exchange coefficient and an equilibrium temperature", "secondary"),
    "DO saturation": ("Benson & Krause 1984, Limnol. Oceanogr. 29:620-632, as implemented in the frozen "
                      "sentinel.models.hydromet.physics.do_sat_mgl", "table test in tests/test_hydromet.py",
                      "frozen shared routine; not re-read on 2026-11-10", "secondary"),
    "barometric pressure": ("frozen sentinel.models.hydromet.physics.pressure_atm_from_elevation (standard atmosphere "
                            "at 15 degC)", "FAO-56 eq. 7 uses 20 degC: 81.8 against 81.5 kPa at 1800 m (0.4 %)",
                            "one formula kept (the frozen one)", "definition"),
}


def table() -> list[dict]:
    """Rows for the deviations entry and the run records: name, value, unit, status, source, where."""
    return [{"name": k, "value": v.value, "unit": v.unit, "status": v.status, "source": v.source, "where": v.where,
             "used_by": v.used_by} for k, v in C.items()]
