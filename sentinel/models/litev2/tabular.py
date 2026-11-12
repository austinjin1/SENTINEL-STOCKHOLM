# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tabular models of architecture round v2: the anchor (= comparator S6, GBM-parity) and the comparators S1–S11,
P1, P2a, P2b (amendment §4.2 and §5; verifier rulings F2 = S10 TabPFN, DS-06 = S11 GBM-stacked, F4 = no teacher
feature on an M2 Test-S pairing). Everything here is fitted on the v2 training rows only (``rows.train_rows``).

Feature sets
    * 45 frozen chip features (``lite.features.chip_features``), the 50 DOQQ scene statistics, lat, lon, DOY = 98
      (:func:`parity_columns`).
    * turbidity: one pooled model over NTU, FNU and NTRU rows with the two indicators :data:`UNIT_COLS`; it predicts
      a unit by setting the indicator. TSS and SSC are separate models. Censored rows are not used by a GBM.
    * anchors of training rows are out-of-fold over five water-body folds (:func:`anchor_fold`); evaluation rows get
      the model refitted on all v2 training rows.

Readings of the text that are decided here (each is listed for the lead in the implementation notes):
    * S8, "the other targets": a per-target model gets the out-of-fold S6 predictions of every other quantity
      (the other targets and NTRU) and of the four teacher means; the pooled turbidity model gets them for Secchi,
      TSS, SSC and chl-a and the four teacher means (the three turbidity units are its own targets).
    * S8, "chip-mean teacher μ": the trust-weighted mean of the teacher cells that the KD loss of the student uses
      (``lite.distill.load_kd_tensors``); missing where the chip has no pairing, no trusted cell, or is masked (F4).
    * S9: the listed XGBoost settings and package defaults for everything else; the three seeds are kept although
      the default settings make them identical.
    * P2a: water pixels = the frozen ``features.water_mask``; with fewer than five water pixels the statistics are
      taken over the whole chip (as the frozen chip features do); "percent water" is the mask fraction.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..lite import baselines as B
from ..lite import data as D
from ..lite import features as F
from . import labels2b as L

SCENE_BANDS = ("red", "green", "blue", "nir")
SCENE_STATS = ("mean", "std", "p01", "p02", "p05", "p25", "p50", "p75", "p95", "p98", "p99", "darknir_median")
SCENE_COLS = tuple(f"{b}_{s}" for b in SCENE_BANDS for s in SCENE_STATS) + ("valid_frac", "darknir_frac")   # 50
CTX_COLS = ("lat", "lon", "doy")
UNIT_COLS = ("unit_fnu", "unit_ntru")
TURB_UNITS = {"turbidity": (0.0, 0.0), "turb_fnu": (1.0, 0.0), "turb_ntru": (0.0, 1.0)}
SINGLE = ("secchi", "tss", "ssc", "chla")                       # one model per target
QUANTITIES = (*L.TARGETS, "turb_ntru")                          # the seven anchor outputs
TEACHER_TARGETS = tuple(D.TARGETS)                              # secchi, turbidity, tss, chla
TEACHER_COLS = tuple(f"tmu_{t}" for t in TEACHER_TARGETS)
N_FOLDS = 5
FOLD_SALT = "SENTINEL-Lite-anchor|"
SEEDS = tuple(B.GBM_SEEDS)

AFT_PARAMS = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
              "aft_loss_distribution_scale": 1.0, "learning_rate": 0.03, "max_depth": 4, "nthread": 6,
              "verbosity": 0}
AFT_ROUNDS = 400
AFT_MIN_CENSORED = 20                                           # amendment l.276
TABPFN_CAP = 10_000                                             # lead order item 6 (memory test may lower it)
TABPFN_SALT = "SENTINEL-Lite-tabpfn|"
TABPFN_CHUNK = 2000
STACK_PCS = 16                                                  # S11
CYFI_STATS = ("mean", "min", "max", "p05", "p95")
P1_FORMS = {"secchi": "secchi", "turbidity": "turbidity", "turb_fnu": "turbidity", "tss": "tss", "ssc": "tss",
            "chla": "chla"}                                     # frozen form family used for each target (§5 P1)


class TabularError(RuntimeError):
    """A tabular model was asked for something the amendment does not allow."""


# ------------------------------------------------------------------------------------------------ features
_CHIP_COLS: list[str] | None = None


def chip_columns() -> list[str]:
    """Names of the 45 frozen chip features (from the frozen function, on one blank chip)."""
    global _CHIP_COLS
    if _CHIP_COLS is None:
        z = np.zeros((1, 4, D.CHIP_SIZE, D.CHIP_SIZE), np.uint8)
        _CHIP_COLS = list(F.chip_features(z, np.array([1.0]), np.ones((1, 4))).columns)
    return list(_CHIP_COLS)


def parity_columns() -> list[str]:
    """The 98 features of S6 / the anchor: 45 chip features, 50 scene statistics, lat, lon, DOY."""
    cols = [*chip_columns(), *SCENE_COLS, *CTX_COLS]
    if len(cols) != 98 or len(set(cols)) != 98:
        raise TabularError(f"the parity feature list has {len(cols)} columns ({len(set(cols))} distinct), not 98")
    return cols


def feature_table(rows: pd.DataFrame, chips: np.ndarray, scene: pd.DataFrame) -> pd.DataFrame:
    """98 parity features aligned with ``rows``. ``scene``: scene-statistics table aligned with ``rows``
    (``rows.scene_table``). The scene median used by the chip features is the frozen one (band p50)."""
    if not (len(rows) == len(chips) == len(scene)):
        raise TabularError("rows, chips and scene statistics differ in length")
    p50 = np.column_stack([scene[f"{b}_p50"].to_numpy(float) for b in SCENE_BANDS])
    if not np.isfinite(p50).all():
        raise TabularError("a row lacks the DOQQ band medians")
    chip = F.chip_features(np.asarray(chips), rows["gsd_m"].to_numpy(float), p50)
    out = pd.concat([chip.reset_index(drop=True), scene[list(SCENE_COLS)].reset_index(drop=True).astype(float),
                     rows[list(CTX_COLS)].reset_index(drop=True).astype(float)], axis=1)
    return out[parity_columns()]


def cyfi_features(rows: pd.DataFrame, chips: np.ndarray) -> pd.DataFrame:
    """CyFi-style features of P2a (30): per band mean, min, max, p05, p95 over water pixels; B/R, G/R, B/G, NIR/R,
    NDVI, NDWI from the band means; green p95/p05; percent water; month; days between image and sample (a
    label-side field, see ``rows``)."""
    chips = np.asarray(chips)
    wm = F.water_mask(chips)
    month = pd.to_datetime(rows["naip_date"]).dt.month.to_numpy(float)
    recs = []
    for i in range(len(chips)):
        x = chips[i].astype(np.float32).reshape(4, -1)
        m = wm[i].reshape(-1)
        px = x[:, m] if m.sum() >= 5 else x
        f: dict = {}
        for k, b in enumerate(D.NAIP_BANDS):
            p05, p95 = np.percentile(px[k], [5, 95])
            f[f"cy_{b}_mean"], f[f"cy_{b}_min"], f[f"cy_{b}_max"] = float(px[k].mean()), float(px[k].min()), float(px[k].max())
            f[f"cy_{b}_p05"], f[f"cy_{b}_p95"] = float(p05), float(p95)
        r, g, b_, n = (f[f"cy_{b}_mean"] + 1.0 for b in D.NAIP_BANDS)
        f["cy_B_R"], f["cy_G_R"], f["cy_B_G"], f["cy_N_R"] = b_ / r, g / r, b_ / g, n / r
        f["cy_ndvi"], f["cy_ndwi"] = (n - r) / (n + r), (g - n) / (g + n)
        f["cy_G_p95_p05"] = (f["cy_G_p95"] + 1.0) / (f["cy_G_p05"] + 1.0)
        f["cy_pct_water"] = 100.0 * float(m.mean())
        recs.append(f)
    out = pd.DataFrame(recs)
    out["cy_month"] = month
    out["cy_dt_label_days"] = rows["dt_label_days"].to_numpy(float)
    if out.shape[1] != 30:
        raise TabularError(f"the CyFi-style table has {out.shape[1]} columns, not 30")
    return out


def context_block(rows: pd.DataFrame, scene: pd.DataFrame) -> pd.DataFrame:
    """s (51) and c (5) of §4.1, not standardised: 50 scene statistics + gsd; unit-sphere position; DOY sin/cos."""
    lat, lon = np.deg2rad(rows["lat"].to_numpy(float)), np.deg2rad(rows["lon"].to_numpy(float))
    ang = 2.0 * np.pi * rows["doy"].to_numpy(float) / 365.25
    out = scene[list(SCENE_COLS)].reset_index(drop=True).astype(float)
    out["gsd_m"] = rows["gsd_m"].to_numpy(float)
    out["c_x"], out["c_y"], out["c_z"] = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    out["c_doy_sin"], out["c_doy_cos"] = np.sin(ang), np.cos(ang)
    if out.shape[1] != 56:
        raise TabularError(f"[s, c] has {out.shape[1]} columns, not 56")
    return out


# ------------------------------------------------------------------------------------------------ folds
def anchor_fold(water_body) -> int:
    """Fold of a water body: ``int(sha256("SENTINEL-Lite-anchor|" + water_body).hexdigest()[:8], 16) % 5``."""
    return int(hashlib.sha256((FOLD_SALT + str(water_body)).encode()).hexdigest()[:8], 16) % N_FOLDS


def folds(water_body) -> np.ndarray:
    wb = pd.Series(np.asarray(water_body).astype(str))
    return wb.map({w: anchor_fold(w) for w in wb.unique()}).to_numpy(int)


# ------------------------------------------------------------------------------------------------ GBM
def fit_gbm(X: np.ndarray, y: np.ndarray, seeds=SEEDS) -> list:
    """LightGBM with the frozen ``GBM_PARAMS``, one model per seed, on rows with a finite label (as the frozen
    ``baselines.gbm``). Returns [] when no row has a label."""
    import lightgbm as lgb

    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if not ok.any():
        return []
    X = np.asarray(X, float)
    return [lgb.LGBMRegressor(random_state=sd, **B.GBM_PARAMS).fit(X[ok], y[ok]) for sd in seeds]


def predict_gbm(models: list, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    if not models:
        return np.full(len(X), np.nan)
    return np.mean([m.predict(X) for m in models], axis=0)


def turbidity_stack(X: np.ndarray, Y: pd.DataFrame, units=tuple(TURB_UNITS)) -> tuple[np.ndarray, np.ndarray]:
    """Pooled turbidity training set: one copy of a row per unit it has a label in, with the unit indicators."""
    X = np.asarray(X, float)
    xs, ys = [], []
    for u in units:
        y = Y[f"y_{u}"].to_numpy(float)
        ok = np.isfinite(y)
        ind = np.tile(np.asarray(TURB_UNITS[u], float), (int(ok.sum()), 1))
        xs.append(np.column_stack([X[ok], ind]))
        ys.append(y[ok])
    return np.vstack(xs), np.concatenate(ys)


def with_unit(X: np.ndarray, unit: str) -> np.ndarray:
    X = np.asarray(X, float)
    return np.column_stack([X, np.tile(np.asarray(TURB_UNITS[unit], float), (len(X), 1))])


def fit_parity(X: pd.DataFrame, Y: pd.DataFrame, cols=None, *, pooled: bool = True, seeds=SEEDS,
               turb_cols=None) -> dict:
    """The S6 model set (``pooled=True``) or the S7 one (``pooled=False``: NTU-only and FNU-only turbidity models,
    NTRU rows unused). ``cols``: features of the per-target models; ``turb_cols``: features of the turbidity
    model(s) before the unit indicators (default ``cols``)."""
    cols = list(cols or X.columns)
    tcols = list(turb_cols or cols)
    out = {"cols": cols, "turb_cols": tcols, "pooled": bool(pooled), "seeds": list(seeds), "models": {}, "n_fit": {}}
    Xa, Xt = X[cols].to_numpy(float), X[tcols].to_numpy(float)
    for t in SINGLE:
        y = Y[f"y_{t}"].to_numpy(float)
        out["models"][t] = fit_gbm(Xa, y, seeds)
        out["n_fit"][t] = int(np.isfinite(y).sum())
    if pooled:
        Xp, yp = turbidity_stack(Xt, Y)
        out["models"]["turb_pooled"] = fit_gbm(Xp, yp, seeds)
        out["n_fit"]["turb_pooled"] = int(len(yp))
        out["n_fit"].update({u: int(Y[f"y_{u}"].notna().sum()) for u in TURB_UNITS})
    else:
        for u in ("turbidity", "turb_fnu"):
            y = Y[f"y_{u}"].to_numpy(float)
            out["models"][u] = fit_gbm(Xt, y, seeds)
            out["n_fit"][u] = int(np.isfinite(y).sum())
    return out


def predict_parity(model: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Predictions (log10) for :data:`QUANTITIES`; ``turb_ntru`` is missing for the per-unit set."""
    Xa, Xt = X[model["cols"]].to_numpy(float), X[model["turb_cols"]].to_numpy(float)
    out = {t: predict_gbm(model["models"][t], Xa) for t in SINGLE}
    for u in TURB_UNITS:
        if model["pooled"]:
            out[u] = predict_gbm(model["models"]["turb_pooled"], with_unit(Xt, u))
        else:
            out[u] = predict_gbm(model["models"].get(u, []), Xt)
    return pd.DataFrame(out, index=X.index)[list(QUANTITIES)]


def oof_parity(X: pd.DataFrame, Y: pd.DataFrame, water_body, cols=None, *, pooled: bool = True, seeds=SEEDS,
               turb_cols=None) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold predictions of every quantity for every training row (five water-body folds)."""
    fold = folds(water_body)
    out = pd.DataFrame(np.nan, index=X.index, columns=list(QUANTITIES))
    info = {"n_folds": N_FOLDS, "rows_per_fold": np.bincount(fold, minlength=N_FOLDS).tolist()}
    for k in range(N_FOLDS):
        te = fold == k
        if not te.any():
            continue
        m = fit_parity(X.loc[~te], Y.loc[~te], cols, pooled=pooled, seeds=seeds, turb_cols=turb_cols)
        out.loc[te] = predict_parity(m, X.loc[te]).to_numpy()
    return out, info


def oof_single(X: np.ndarray, y: np.ndarray, water_body, seeds=SEEDS) -> np.ndarray:
    """Out-of-fold predictions of one GBM target for every training row."""
    fold, X, y = folds(water_body), np.asarray(X, float), np.asarray(y, float)
    out = np.full(len(X), np.nan)
    for k in range(N_FOLDS):
        te = fold == k
        if te.any():
            out[te] = predict_gbm(fit_gbm(X[~te], y[~te], seeds), X[te])
    return out


def anchors(Xtr: pd.DataFrame, Ytr: pd.DataFrame, wb_tr, Xev: dict[str, pd.DataFrame] | None = None,
            seeds=SEEDS) -> dict:
    """Anchor table of §4.2: ``{'train': out-of-fold anchors, 'model': the model refitted on all training rows,
    '<block>': anchors of each evaluation frame of ``Xev``}``. The refitted model is also comparator S6."""
    cols = parity_columns()
    oof, info = oof_parity(Xtr, Ytr, wb_tr, cols, seeds=seeds)
    model = fit_parity(Xtr, Ytr, cols, seeds=seeds)
    out = {"train": oof, "model": model, "info": {**info, "n_fit": model["n_fit"]}}
    for name, Xe in (Xev or {}).items():
        out[name] = predict_parity(model, Xe)
    return out


# ------------------------------------------------------------------------------------------------ S1–S5, P1
def frozen_list(rows_tr: pd.DataFrame, Xtr: pd.DataFrame, rows_ev: pd.DataFrame, Xev: pd.DataFrame) -> dict:
    """S1–S5 and every P1 form with the frozen functions, one model per target (NTRU rows are not used; FNU and SSC
    are their own targets). Returns ``{name: {target: prediction}}``; P1 forms are ``P1_<form>``."""
    chip_cols = chip_columns()
    ctx = [*chip_cols, *CTX_COLS]
    out: dict = {k: {} for k in ("S1_train_median", "S2_loc_knn", "S3_loc_season_gbm", "S4_gbm_rgbn", "S5_gbm_ctx")}
    for t in L.TARGETS:
        y = rows_tr[f"y_{t}"].to_numpy(float)
        if np.isfinite(y).sum() == 0:
            continue
        out["S1_train_median"][t] = np.full(len(rows_ev), np.nanmedian(y))
        out["S2_loc_knn"][t] = B.location_knn(rows_tr["lat"], rows_tr["lon"], rows_tr["site_id"], y, rows_ev["lat"],
                                              rows_ev["lon"])[0]
        out["S3_loc_season_gbm"][t] = B.location_season_gbm(Xtr, y, Xev)[0]
        out["S4_gbm_rgbn"][t] = B.gbm(Xtr, y, Xev, cols=chip_cols)[0]
        out["S5_gbm_ctx"][t] = B.gbm(Xtr, y, Xev, cols=ctx)[0]
        for form in B.BAND_RATIO_FORMS[P1_FORMS[t]]:
            out.setdefault(f"P1_{form}", {})[t] = B.band_ratio(Xtr, y, Xev, P1_FORMS[t], form)[0]
    return out


def parity_predictions(model: dict, Xev: pd.DataFrame) -> dict:
    """{target: prediction} of an S6 / S7 model set for the six targets."""
    p = predict_parity(model, Xev)
    return {t: p[t].to_numpy(float) for t in L.TARGETS}


# ------------------------------------------------------------------------------------------------ S8
def s8_extra_columns(target: str) -> list[str]:
    """First-stage columns given to the S8 model of ``target`` (``'turb_pooled'`` for the turbidity model)."""
    own = set(TURB_UNITS) if target == "turb_pooled" else {target}
    return [f"a_{q}" for q in QUANTITIES if q not in own] + list(TEACHER_COLS)


def fit_predict_s8(Xtr: pd.DataFrame, Ytr: pd.DataFrame, wb_tr, teacher_tr: pd.DataFrame, Xev: pd.DataFrame,
                   *, first_stage: dict | None = None, seeds=SEEDS) -> tuple[dict, dict]:
    """S8 GBM-parity-MT. ``teacher_tr``: chip-mean teacher μ of the four teacher targets for the training rows
    (:data:`TEACHER_COLS`; missing where there is none or where the pairing is masked). ``first_stage``: the output
    of :func:`anchors` (re-used when given). No teacher value of an evaluation row is used: the teacher columns of
    an evaluation row are predictions of a GBM on its 98 features."""
    cols = parity_columns()
    fs = first_stage or anchors(Xtr, Ytr, wb_tr, seeds=seeds)
    a_tr = fs["train"].add_prefix("a_")
    a_ev = predict_parity(fs["model"], Xev).add_prefix("a_")
    t_tr, t_ev, n_teacher = {}, {}, {}
    Xa, Xe = Xtr[cols].to_numpy(float), Xev[cols].to_numpy(float)
    for c in TEACHER_COLS:
        y = teacher_tr[c].to_numpy(float)
        n_teacher[c] = int(np.isfinite(y).sum())
        t_tr[c] = oof_single(Xa, y, wb_tr, seeds)
        t_ev[c] = predict_gbm(fit_gbm(Xa, y, seeds), Xe)
    Ztr = pd.concat([Xtr[cols].reset_index(drop=True), a_tr.reset_index(drop=True), pd.DataFrame(t_tr)], axis=1)
    Zev = pd.concat([Xev[cols].reset_index(drop=True), a_ev.reset_index(drop=True), pd.DataFrame(t_ev)], axis=1)
    Yr = Ytr.reset_index(drop=True)
    preds: dict = {}
    for t in SINGLE:
        c2 = cols + s8_extra_columns(t)
        preds[t] = predict_gbm(fit_gbm(Ztr[c2].to_numpy(float), Yr[f"y_{t}"].to_numpy(float), seeds),
                               Zev[c2].to_numpy(float))
    c2 = cols + s8_extra_columns("turb_pooled")
    Xp, yp = turbidity_stack(Ztr[c2].to_numpy(float), Yr)
    mt = fit_gbm(Xp, yp, seeds)
    for u in ("turbidity", "turb_fnu"):
        preds[u] = predict_gbm(mt, with_unit(Zev[c2].to_numpy(float), u))
    return preds, {"n_teacher_rows": n_teacher, "extra_columns": {t: s8_extra_columns(t) for t in (*SINGLE, "turb_pooled")}}


def teacher_chip_means(kd: dict) -> pd.DataFrame:
    """Trust-weighted chip mean of the teacher μ per teacher target from ``lite.distill.load_kd_tensors`` output."""
    w = np.where(np.isfinite(kd["t_mu"]), kd["t_w"], 0.0).astype(float)
    mu = np.where(np.isfinite(kd["t_mu"]), kd["t_mu"], 0.0).astype(float)
    sw = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(sw > 0, (w * mu).sum(axis=1) / sw, np.nan)
    return pd.DataFrame(m, columns=list(TEACHER_COLS))


# ------------------------------------------------------------------------------------------------ S9
def aft_targets(rows_tr: pd.DataFrame, minimum: int = AFT_MIN_CENSORED) -> dict:
    """{target: number of censored training rows} for the targets that S9 runs on (count only)."""
    n = {t: int((rows_tr[f"cens_{t}"] != 0).sum()) for t in L.TARGETS}
    return {t: k for t, k in n.items() if k >= minimum}


def aft_bounds(y_log: np.ndarray, cens: np.ndarray, lim_log: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row mask, lower, upper) in linear units for XGBoost ``survival:aft``: uncensored [y, y]; left-censored
    [0, L]; right-censored [L, +inf)."""
    y, c, lim = np.asarray(y_log, float), np.asarray(cens, int), np.asarray(lim_log, float)
    unc = np.isfinite(y)
    use = unc | ((c != 0) & np.isfinite(lim))
    lo = np.where(unc, 10.0 ** np.where(unc, y, 0.0), np.where(c > 0, 10.0 ** np.where(np.isfinite(lim), lim, 0.0), 0.0))
    hi = np.where(unc, 10.0 ** np.where(unc, y, 0.0), np.where(c < 0, 10.0 ** np.where(np.isfinite(lim), lim, 0.0), np.inf))
    return use, lo[use], hi[use]


def fit_predict_aft(Xtr: np.ndarray, y_log, cens, lim_log, Xev: np.ndarray, seeds=SEEDS) -> tuple[np.ndarray, dict]:
    """S9 GBM-parity-AFT for one target: log10 prediction for the evaluation rows (mean of the seeds)."""
    import xgboost as xgb

    use, lo, hi = aft_bounds(y_log, cens, lim_log)
    Xtr, Xev = np.asarray(Xtr, float), np.asarray(Xev, float)
    dtr = xgb.DMatrix(Xtr[use])
    dtr.set_float_info("label_lower_bound", lo)
    dtr.set_float_info("label_upper_bound", hi)
    dev = xgb.DMatrix(Xev)
    preds = []
    for sd in seeds:
        bst = xgb.train({**AFT_PARAMS, "seed": int(sd)}, dtr, num_boost_round=AFT_ROUNDS)
        preds.append(bst.predict(dev, output_margin=True) / np.log(10.0))
    c = np.asarray(cens, int)[use]
    return np.mean(preds, axis=0), {"n_fit": int(use.sum()), "n_left": int(((c < 0) & (lo == 0)).sum()),
                                    "n_right": int(np.isinf(hi).sum()), "rounds": AFT_ROUNDS}


# ------------------------------------------------------------------------------------------------ S10
def tabpfn_keep(water_body, cap: int = TABPFN_CAP) -> np.ndarray:
    """Row mask of the F2 rule: all rows when there are at most ``cap``; otherwise whole water bodies in ascending
    order of ``sha256("SENTINEL-Lite-tabpfn|" + water_body)`` until the next one would exceed ``cap``."""
    wb = np.asarray(water_body).astype(str)
    if len(wb) <= cap:
        return np.ones(len(wb), bool)
    size = pd.Series(wb).value_counts()
    order = sorted(size.index, key=lambda w: hashlib.sha256((TABPFN_SALT + w).encode()).hexdigest())
    kept, total = set(), 0
    for w in order:
        if total + int(size[w]) > cap:
            break
        kept.add(w)
        total += int(size[w])
    return np.isin(wb, list(kept))


def fit_predict_tabpfn(Xtr: np.ndarray, y_log, water_body, Xev: np.ndarray, *, cap: int = TABPFN_CAP,
                       device: str = "cuda", seeds=SEEDS, chunk: int = TABPFN_CHUNK) -> tuple[np.ndarray, dict]:
    """S10 for one target: TabPFN v2 regressor with package defaults on the uncensored rows, 3 seeds averaged.
    The package is loaded only through the project loader; nothing is installed or downloaded."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from ..chemtox.tabpfn_model import load_tabpfn

    TabPFNRegressor, version = load_tabpfn()
    y = np.asarray(y_log, float)
    ok = np.isfinite(y)
    Xa, ya, wb = np.asarray(Xtr, np.float32)[ok], y[ok], np.asarray(water_body).astype(str)[ok]
    keep = tabpfn_keep(wb, cap)
    Xa, ya = Xa[keep], ya[keep]
    Xe = np.asarray(Xev, np.float32)
    preds = []
    for sd in seeds:
        m = TabPFNRegressor(device=device, random_state=int(sd))
        m.fit(Xa, ya)
        preds.append(np.concatenate([np.asarray(m.predict(Xe[i:i + chunk]), float) for i in range(0, len(Xe), chunk)])
                     if len(Xe) else np.empty(0))
    return np.mean(preds, axis=0), {"n_uncensored": int(ok.sum()), "n_fit": int(len(ya)), "cap": int(cap),
                                    "n_water_bodies_fit": int(len(set(wb[keep]))), "tabpfn_version": version,
                                    "seeds": list(seeds)}


# ------------------------------------------------------------------------------------------------ S11
def pca_fit(E: np.ndarray, k: int = STACK_PCS) -> dict:
    """PCA of the pooled embedding on v2 training rows only (mean, first ``k`` right singular vectors)."""
    E = np.asarray(E, float)
    mean = E.mean(axis=0)
    _, _, vt = np.linalg.svd(E - mean, full_matrices=False)
    return {"mean": mean, "components": vt[:k]}


def pca_apply(p: dict, E: np.ndarray) -> pd.DataFrame:
    z = (np.asarray(E, float) - p["mean"]) @ p["components"].T
    return pd.DataFrame(z, columns=[f"pc{i:02d}" for i in range(z.shape[1])])


def fit_predict_stacked(Xtr: pd.DataFrame, Ytr: pd.DataFrame, Etr: np.ndarray, Xev: pd.DataFrame, Eev: np.ndarray,
                        seeds=SEEDS) -> tuple[dict, dict]:
    """S11 GBM-stacked: the S6 model set on the 98 features + 16 principal components of the pooled embedding."""
    cols = parity_columns()
    p = pca_fit(Etr)
    Ztr = pd.concat([Xtr[cols].reset_index(drop=True), pca_apply(p, Etr)], axis=1)
    Zev = pd.concat([Xev[cols].reset_index(drop=True), pca_apply(p, Eev)], axis=1)
    m = fit_parity(Ztr, Ytr.reset_index(drop=True), list(Ztr.columns), seeds=seeds)
    return parity_predictions(m, Zev), {"n_pcs": int(p["components"].shape[0]), "n_fit": m["n_fit"]}


__all__ = ["SCENE_COLS", "CTX_COLS", "UNIT_COLS", "TURB_UNITS", "SINGLE", "QUANTITIES", "TEACHER_COLS", "N_FOLDS",
           "SEEDS", "AFT_PARAMS", "AFT_ROUNDS", "TABPFN_CAP", "TabularError", "chip_columns", "parity_columns",
           "feature_table", "cyfi_features", "context_block", "anchor_fold", "folds", "fit_gbm", "predict_gbm",
           "turbidity_stack", "with_unit", "fit_parity", "predict_parity", "oof_parity", "oof_single", "anchors",
           "frozen_list", "parity_predictions", "s8_extra_columns", "fit_predict_s8", "teacher_chip_means",
           "aft_targets", "aft_bounds", "fit_predict_aft", "tabpfn_keep", "fit_predict_tabpfn", "pca_fit",
           "pca_apply", "fit_predict_stacked"]
