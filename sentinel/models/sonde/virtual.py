# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Task N-b: virtual sensing at UNSEEN stations (RESEARCH_PLAN §5.1, §15.2 M1 primary part 2).

Estimate variable v on day t at a val/test-role site where v's sensor is absent, from the site's
OTHER co-located sensors (days <= t; same-day values allowed — this is estimation, not forecasting),
lagged observed weather, calendar and static attributes. The whole SENSOR GROUP of v is withheld
(e.g. DO mean AND DO min; pH mean AND median; turbidity mean AND median), and nothing derived from v
at that site is used (no v climatology, no v site statistics).

Splits: fit on train-role sites × TRAIN block; early stopping on train-role sites × VAL block (the
val-role evaluation sites are never used for fitting or stopping); score on val-role sites × VAL
block (the unseen-station cell). Metric: per-site KGE of the point estimate (original units),
median across sites; comparison = per-site KGE gain over the GBM baseline with a basin-group
bootstrap CI.

Models: ``VirtualGBM`` (LightGBM, L2) on trailing features; ``VirtualLSTM`` (sequence model over the
last ``context`` days of co-located standardised anomalies/levels + weather, MSE on the globally
standardised target).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...data.sonde_splits import TIME_BLOCKS
from .common import SondeData
from .data import from_model_space, to_model_space
from .features import FeatureBuilder, rolling_mean

SENSOR_GROUPS = [{"temp_c_mean"}, {"do_mgl_mean", "do_mgl_min"}, {"spcond_uscm_mean"}, {"ph_mean", "ph_median"},
                 {"turb_fnu_mean", "turb_fnu_median"}, {"no3no2_mgl_mean"}, {"chl_ugl_mean"}, {"fdom_qse_mean"},
                 {"discharge_cfs_mean"}]


def withheld(v: str) -> set:
    for g in SENSOR_GROUPS:
        if v in g:
            return g
    return {v}


@dataclass
class VRows:
    var: str
    si: np.ndarray
    ti: np.ndarray
    y: np.ndarray        # model space


def virtual_rows(data: SondeData, v: str, block: str, roles) -> VRows:
    """(site, day) rows of ``block`` (embargoed block days) where v is observed, for sites in roles."""
    j = data.cube.var_index(v)
    z = to_model_space(v, data.cube.y[:, :, j])
    t_idx = data.issue_idx(block, 0) if block != "train" else np.flatnonzero(
        data.split.time_mask(data.cube.dates, "train"))
    s_idx = data.site_idx(roles)
    approved = ~data.cube.prov[:, :, j]
    m = np.isfinite(z[np.ix_(s_idx, t_idx)]) & approved[np.ix_(s_idx, t_idx)]
    a, k = np.nonzero(m)
    si, ti = s_idx[a].astype(np.int64), t_idx[k].astype(np.int64)
    return VRows(v, si, ti, z[si, ti])


def virtual_features(fb: FeatureBuilder, data: SondeData, v: str, si: np.ndarray, ti: np.ndarray) -> pd.DataFrame:
    """Same-day and trailing features of the OTHER sensor groups (+ weather, calendar, statics)."""
    excl = withheld(v)
    X = {}
    for u in data.cube.variables:
        if u in excl:
            continue
        z = fb.z(u)
        j = data.cube.var_index(u)
        cm = data.clim_mean[:, fb.doy, j]
        a = z - cm
        X[f"{u}_val"] = z[si, ti]
        X[f"{u}_anom"] = a[si, ti]
        X[f"{u}_anom7"] = rolling_mean(a, 7, 2)[si, ti]
        X[f"{u}_mean30"] = rolling_mean(z, 30, 5)[si, ti]
        X[f"{u}_clim"] = cm[si, ti]                     # u's own training climatology (u is observed)
    for k, arr in fb.weather_block(data.weather).items():
        X[k] = arr[si, ti]
    d = fb.doy[ti]
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * d / 365.25), np.cos(2 * np.pi * d / 365.25)
    for c in ("lat", "lon", "elev_m", "log_drain_area_km2") + tuple(
            c for c in data.static.columns if str(c).startswith("sc_")):
        X[c] = data.static[c].to_numpy(dtype=float)[si]
    df = pd.DataFrame({k: np.asarray(x, dtype="float32") for k, x in X.items()})
    df["huc2"] = pd.Categorical(fb.huc2[si].astype(int), categories=list(range(-1, 23)))
    return df


class VirtualGBM:
    def __init__(self, num_threads: int = 16, max_rounds: int = 2000, seed: int = 0, max_rows: int = 3_000_000):
        self.p = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                  "num_threads": num_threads, "seed": seed, "deterministic": True, "force_row_wise": True,
                  "verbose": -1}
        self.max_rounds, self.seed, self.max_rows = max_rounds, seed, max_rows
        self.models = {}

    def fit(self, fb, data, v, tr: VRows, es: VRows):
        import lightgbm as lgb

        rs = np.random.default_rng(self.seed)
        k = np.arange(len(tr.y)) if len(tr.y) <= self.max_rows else np.sort(rs.choice(len(tr.y), self.max_rows, False))
        X = virtual_features(fb, data, v, tr.si[k], tr.ti[k])
        ds = lgb.Dataset(X, tr.y[k])
        ke = np.arange(len(es.y)) if len(es.y) <= 400_000 else np.sort(rs.choice(len(es.y), 400_000, False))
        Xe = virtual_features(fb, data, v, es.si[ke], es.ti[ke])
        b = lgb.train(self.p, ds, self.max_rounds, valid_sets=[lgb.Dataset(Xe, es.y[ke], reference=ds)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        self.models[v] = b
        return {"n_fit": int(len(k)), "n_es": int(len(ke)), "best_iteration": int(b.best_iteration)}

    def predict(self, fb, data, v, rows: VRows) -> np.ndarray:
        b = self.models[v]
        return b.predict(virtual_features(fb, data, v, rows.si, rows.ti), num_iteration=b.best_iteration or None)


@dataclass
class VLSTMConfig:
    context: int = 90
    hidden: int = 128
    batch_size: int = 512
    lr: float = 1e-3
    max_epochs: int = 30
    steps_per_epoch: int = 800
    patience: int = 4
    amp: bool = True
    seed: int = 0


class VirtualLSTM:
    """Sequence estimator: LSTM over the last ``context`` days of the co-located (non-withheld)
    standardised anomalies, site-normalised levels, masks, lagged weather and DOY + static MLP."""

    def __init__(self, cfg: VLSTMConfig | None = None):
        self.cfg = cfg or VLSTMConfig()

    def _tensors(self, data: SondeData, v: str):
        excl = withheld(v)
        U = [u for u in data.cube.variables if u not in excl]
        tr_s = data.site_idx(("train",))
        te = train_end_index(data)          # normalisation statistics: train-role sites, TRAINING YEARS only
        chans = []
        doy = data.cube.dates.dayofyear.to_numpy() - 1
        for u in U:
            j = data.cube.var_index(u)
            z = to_model_space(u, data.cube.y[:, :, j])
            cm, cs = data.clim_mean[:, doy, j], np.maximum(data.clim_std[:, doy, j], 1e-3)
            a = np.clip((z - cm) / cs, -8, 8)
            with np.errstate(invalid="ignore"):
                lv = z
                ref = lv[tr_s, :te]
                mu = np.nanmean(ref) if np.isfinite(ref).any() else 0.0
                sd = np.nanstd(ref) + 1e-6 if np.isfinite(ref).any() else 1.0
            chans += [np.nan_to_num(a), np.nan_to_num(np.clip((lv - mu) / sd, -8, 8)), np.isfinite(z).astype(float)]
        ang = 2 * np.pi * doy / 365.25
        S, T = data.cube.y.shape[:2]
        chans += [np.broadcast_to(np.sin(ang), (S, T)), np.broadcast_to(np.cos(ang), (S, T))]
        if data.weather is not None:
            w = data.weather.astype(float).copy()
            w[..., 0] = np.log1p(np.maximum(w[..., 0], 0))
            wt = w[tr_s, :te].reshape(-1, w.shape[-1])
            wz = (w - np.nanmean(wt, 0)) / (np.nanstd(wt, 0) + 1e-6)
            chans += [np.nan_to_num(np.clip(wz[..., k], -8, 8)) for k in range(w.shape[-1])]
        dyn = np.stack(chans, axis=-1).astype(np.float16)
        st = []
        for c in ("lat", "lon", "elev_m", "log_drain_area_km2") + tuple(
                c for c in data.static.columns if str(c).startswith("sc_")):
            x = data.static[c].to_numpy(float)
            xt = x[tr_s][np.isfinite(x[tr_s])]
            st += [np.nan_to_num((x - xt.mean()) / (xt.std() + 1e-6)) if xt.size >= 3 else np.zeros(S),
                   np.isfinite(x).astype(float)]
        static = np.clip(np.stack(st, 1), -10, 10).astype(np.float32)
        return dyn, static

    def fit_predict(self, data: SondeData, v: str, tr: VRows, es: VRows, ev: VRows, device="cuda", log=print):
        import torch
        from torch import nn

        from ...core.seed import seed_everything

        cfg = self.cfg
        seed_everything(cfg.seed)
        dev = torch.device(device)
        dyn, static = self._tensors(data, v)
        C = cfg.context
        pad = np.zeros((dyn.shape[0], C - 1, dyn.shape[2]), np.float16)
        D = torch.from_numpy(np.concatenate([pad, dyn], 1)).to(dev)
        ST = torch.from_numpy(static).to(dev)
        mu, sd = float(tr.y.mean()), float(tr.y.std() + 1e-6)          # global TRAIN-site target scaling
        ar = torch.arange(C, device=dev)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.st = nn.Sequential(nn.Linear(static.shape[1], 32), nn.ReLU())
                self.inp = nn.Linear(dyn.shape[2] + 32, cfg.hidden)
                self.lstm = nn.LSTM(cfg.hidden, cfg.hidden, batch_first=True)
                self.head = nn.Sequential(nn.Linear(cfg.hidden + 32, 128), nn.ReLU(), nn.Linear(128, 1))

            def forward(self, x, s):
                e = self.st(s)
                h, _ = self.lstm(self.inp(torch.cat([x, e[:, None].expand(-1, x.shape[1], -1).to(x.dtype)], -1)))
                return self.head(torch.cat([h[:, -1], e.to(h.dtype)], -1)).float().squeeze(-1)

        net = Net().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and dev.type == "cuda")

        def batch(si, ti):
            s = torch.as_tensor(si, device=dev)
            t = torch.as_tensor(ti, device=dev)
            return D[s[:, None], t[:, None] + ar[None, :]].float(), ST[s]

        def predict(rows: VRows) -> np.ndarray:
            net.eval()
            out = []
            with torch.no_grad():
                for a in range(0, len(rows.y), 4096):
                    x, s = batch(rows.si[a:a + 4096], rows.ti[a:a + 4096])
                    with torch.autocast(dev.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                        out.append(net(x, s).float().cpu().numpy())
            return np.concatenate(out) * sd + mu if out else np.zeros(0)

        rs = np.random.default_rng(cfg.seed)
        es_k = np.sort(rs.choice(len(es.y), min(len(es.y), 60000), replace=False))
        es_sub = VRows(v, es.si[es_k], es.ti[es_k], es.y[es_k])
        ytr = torch.as_tensor((tr.y - mu) / sd, device=dev, dtype=torch.float32)
        best, best_state, bad, t0 = np.inf, None, 0, time.time()
        for ep in range(cfg.max_epochs):
            net.train()
            for _ in range(cfg.steps_per_epoch):
                k = rs.integers(0, len(tr.y), cfg.batch_size)
                x, s = batch(tr.si[k], tr.ti[k])
                with torch.autocast(dev.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                    pred = net(x, s)
                loss = ((pred.float() - ytr[torch.as_tensor(k, device=dev)]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            vl = float(np.mean((predict(es_sub) - es_sub.y) ** 2))
            log(f"[vlstm] {v} epoch {ep}: val mse {vl:.4f} ({(time.time() - t0) / 60:.1f} min)")
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k_: t_.detach().clone() for k_, t_ in net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        net.load_state_dict(best_state)
        return predict(ev), {"best_es_mse": best, "epochs": ep + 1,
                             "n_params": int(sum(p.numel() for p in net.parameters()))}


def site_kge(data: SondeData, v: str, rows: VRows, pred_model: np.ndarray, min_n: int = 60) -> pd.DataFrame:
    from ...eval.metrics import kge, nse

    y = from_model_space(v, rows.y)
    p = from_model_space(v, pred_model)
    df = pd.DataFrame({"s": rows.si, "y": y, "p": p})
    out = []
    for s, g in df.groupby("s"):
        if len(g) >= min_n:
            out.append({"site_no": data.sites[s], "n": len(g), "kge": kge(g["y"], g["p"]), "nse": nse(g["y"], g["p"])})
    return pd.DataFrame(out)


def train_end_index(data: SondeData) -> int:
    return int(np.searchsorted(data.cube.dates, pd.Timestamp(TIME_BLOCKS["train"][1]), side="right"))
