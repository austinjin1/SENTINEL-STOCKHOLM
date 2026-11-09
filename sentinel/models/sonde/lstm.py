# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Multi-site quantile LSTM for SENTINEL-Sonde (NeuralHydrology-style; RESEARCH_PLAN §5.1 (B)).

Inputs per day t (all variables in :data:`.data.TARGETS`, model space):
  * standardised anomaly  (z − clim_mean(doy)) / clim_std(doy)          [clipped ±8, NaN→0]
  * site-normalised level (z − m_site) / s_site                          [clipped ±8, NaN→0]
  * observation mask (1 = observed)
  * DOY sin/cos
where clim_mean/clim_std are the training-years climatology (``data.climatology``) and m_site /
s_site are the site's training-years mean/std derived from it — so every normalisation statistic
comes from WY2008–2021 only (val/test values cannot change them; tested). Static inputs: lat,
lon, elevation, log drainage area (standardised with train-role-site statistics), HUC2 embedding,
per-variable m_site / log s_site and availability flags.

Encoder: LSTM over a 365-day context ending at the issue date (inclusive). Head: MLP → for each
variable × horizon (1, 2, 3, 5, 7, 10, 14) × 19 quantiles of the standardised anomaly at the
target day; monotone by construction (first quantile + cumulative softplus increments), plus an
optional learnable damped-anomaly anchor w[v,h]·a_last[v] (a_last = latest observed standardised
anomaly, <= 3 days old; a common shift, so quantiles stay monotone). Loss:
pinball, masked on missing targets, averaged per variable (balanced). Forecast quantiles in model
space: clim_mean(d+h) + clim_std(d+h)·q.

Training samples: random (site, issue date) from train-role sites × train block (targets inside
the block). Early stopping: validation loss on val-role sites AND train-role sites in the val
block. fp16 autocast + GradScaler (no bf16 on Turing).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ...data.sonde_splits import TIME_BLOCKS
from ..baselines.sonde_baselines import QUANTILES
from .common import HORIZONS, Rows, SondeData
from .data import TARGETS, to_model_space


@dataclass
class LSTMConfig:
    context: int = 365
    hidden: int = 256
    layers: int = 1
    dropout: float = 0.2
    static_hidden: int = 32
    huc_emb: int = 4
    head_hidden: int = 256
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 40
    steps_per_epoch: int = 1000
    patience: int = 5
    amp: bool = True
    val_stride_days: int = 3
    val_max_samples: int = 60_000
    grad_clip: float = 1.0
    anom_clip: float = 8.0
    scale_floor_frac: float = 0.05
    balance_variables: bool = True
    min_recent_obs_days: int = 3
    anchor_last: bool = True   # add learnable w[v,h]·(latest observed standardised anomaly, <= 3 d old)
    use_weather: bool = False  # append lagged observed weather channels (common.load_weather)
    seed: int = 0
    max_minutes: float | None = None


# ----------------------------------------------------------------------------- tensors

class SondeTensors:
    """Normalised dynamic/static inputs and standardised-anomaly targets for a SondeData."""

    def __init__(self, data: SondeData, *, variables=TARGETS, horizons=HORIZONS, context: int = 365,
                 scale_floor_frac: float = 0.05, anom_clip: float = 8.0, train_roles=("train",),
                 frozen: dict | None = None, use_weather: bool = False):
        """``frozen`` (from :meth:`frozen_stats` of the training-time tensors) fixes the cross-site
        constants (variable scale floors, static standardisation) so inference never recomputes
        them from a different site set."""
        cube = data.cube
        self.data = data
        self.variables = [v for v in variables if v in cube.variables]
        self.horizons = tuple(horizons)
        self.context = int(context)
        S, T = len(cube.sites), len(cube.dates)
        V = len(self.variables)
        jj = [cube.var_index(v) for v in self.variables]
        self.doy = cube.dates.dayofyear.to_numpy() - 1
        cm = data.clim_mean[:, :, jj]                      # (S, 366, V) training years only
        cs = data.clim_std[:, :, jj]
        train_s = data.site_idx(train_roles)
        warnings.filterwarnings("ignore", message="(Mean of empty slice|All-NaN slice|Degrees of freedom)")
        if frozen is not None:
            floor = np.asarray(frozen["floor"], dtype=float)
        else:
            with np.errstate(invalid="ignore"):
                floor = scale_floor_frac * np.nanmedian(cs[train_s], axis=(0, 1))
            floor = np.where(np.isfinite(floor), floor, 1e-3)
        self.floor = floor
        self.cm = cm
        self.cs_eff = np.maximum(cs, floor[None, None, :])
        with np.errstate(invalid="ignore", divide="ignore"):
            self.m_site = np.nanmean(cm, axis=1)                                                    # (S, V)
            self.s_site = np.sqrt(np.nanmean(cs ** 2 + (cm - self.m_site[:, None, :]) ** 2, axis=1))
        self.s_site = np.maximum(self.s_site, floor[None, :])
        z = np.stack([to_model_space(v, cube.y[:, :, j]) for v, j in zip(self.variables, jj)], axis=-1)
        anom = (z - cm[:, self.doy]) / self.cs_eff[:, self.doy]
        lvl = (z - self.m_site[:, None, :]) / self.s_site[:, None, :]
        mask = np.isfinite(z) & np.isfinite(anom)
        anom = np.where(mask, np.clip(anom, -anom_clip, anom_clip), np.nan)
        lvl = np.where(mask, np.clip(lvl, -anom_clip, anom_clip), np.nan)
        ang = 2 * np.pi * self.doy / 365.25
        dyn = np.concatenate([np.nan_to_num(anom), np.nan_to_num(lvl), mask.astype(float),
                              np.broadcast_to(np.sin(ang)[None, :, None], (S, T, 1)),
                              np.broadcast_to(np.cos(ang)[None, :, None], (S, T, 1))], axis=-1)
        self.weather_stats = None
        if use_weather:
            if data.weather is None:
                raise ValueError("use_weather=True but data.weather is None (set use_weather in the config)")
            w = data.weather.astype(float).copy()
            w[..., 0] = np.log1p(np.maximum(w[..., 0], 0))            # precipitation: log1p
            if frozen is not None and frozen.get("weather_stats"):
                mu, sd = (np.asarray(x) for x in frozen["weather_stats"])
            else:
                te = int(np.searchsorted(data.cube.dates, pd.Timestamp(TIME_BLOCKS["train"][1]), side="right"))
                wt = w[train_s, :te].reshape(-1, w.shape[-1])     # training years only
                mu, sd = np.nanmean(wt, axis=0), np.nanstd(wt, axis=0) + 1e-6
            self.weather_stats = (mu.tolist(), sd.tolist())
            wz = np.clip((w - mu) / sd, -8, 8)
            wm = np.isfinite(wz).all(axis=-1, keepdims=True)
            dyn = np.concatenate([dyn, np.nan_to_num(wz), wm.astype(float)], axis=-1)
        self.dyn = dyn.astype(np.float16)                     # (S, T, 3V + 2 [+ W + 1])
        self.targets = anom.astype(np.float32)                                  # (S, T, V), NaN = missing
        self.obs_mask = mask
        # static
        st = data.static
        cols = []
        self.static_stats = {}
        fz = (frozen or {}).get("static_stats")
        for c in ("lat", "lon", "elev_m", "log_drain_area_km2") + tuple(
                c for c in st.columns if str(c).startswith("sc_")):
            x = st[c].to_numpy(dtype=float)
            if fz is not None:
                mu, sd = fz[c]
            else:
                xt = x[train_s][np.isfinite(x[train_s])]
                mu, sd = (float(xt.mean()), float(max(xt.std(), 1e-3))) if xt.size >= 3 else (0.0, 1.0)
            self.static_stats[c] = (float(mu), float(sd))
            cols += [np.nan_to_num((x - mu) / sd), np.isfinite(x).astype(float)]
        avail = np.isfinite(self.m_site)
        for k in range(V):
            for name, x in (("m", self.m_site[:, k]), ("logs", np.log(self.s_site[:, k]))):
                key = f"{name}_{self.variables[k]}"
                if fz is not None:
                    mu, sd = fz[key]
                else:
                    xs = x[train_s][np.isfinite(x[train_s])]
                    # < 3 train sites (or no spread): unit scale, else a 1-site variable would be
                    # standardised by ~1e-6 and overflow fp16 at other sites
                    mu, sd = (float(xs.mean()), float(max(xs.std(), 0.05))) if xs.size >= 3 else \
                        ((float(xs.mean()), 1.0) if xs.size else (0.0, 1.0))
                self.static_stats[key] = (float(mu), float(sd))
                cols.append(np.where(avail[:, k], np.nan_to_num((x - mu) / sd), 0.0))
            cols.append(avail[:, k].astype(float))
        self.static = np.clip(np.stack(cols, axis=1), -10, 10).astype(np.float32)   # (S, n_static)
        huc = st["huc2"].astype(str)
        self.huc = np.array([int(h) + 1 if h.isdigit() else 0 for h in huc], dtype=np.int64)  # 0 = unknown
        self.n_huc = 24
        # sample eligibility: at least one target and a recent observation (any variable)
        Hmax = max(self.horizons)
        tpad = np.full((S, T + Hmax + 1, V), np.nan, dtype=np.float32)
        tpad[:, :T] = self.targets
        self.targets_padded = tpad
        has_t = np.zeros((S, T), dtype=bool)
        for h in self.horizons:
            has_t |= np.isfinite(tpad[:, h:h + T]).any(axis=-1)
        anyobs = mask.any(axis=-1).astype(float)
        c = np.concatenate([np.zeros((S, 1)), np.cumsum(anyobs, axis=1)], axis=1)
        w = 3
        lo = np.maximum(np.arange(1, T + 1) - w, 0)
        recent = (c[:, 1:] - c[:, lo]) > 0
        self.eligible = has_t & recent

    def frozen_stats(self) -> dict:
        return {"floor": [float(f) for f in self.floor], "static_stats": dict(self.static_stats),
                "weather_stats": self.weather_stats}

    @property
    def n_dyn(self) -> int:
        return self.dyn.shape[-1]

    def samples(self, block: str, roles, *, stride: int = 1, max_n: int | None = None, seed: int = 0):
        """(si, ti) eligible samples with issue dates in ``block`` (targets within the block)."""
        d = self.data
        t_idx = d.issue_idx(block, max(self.horizons))
        if stride > 1:
            t_idx = t_idx[::stride]
        s_idx = d.site_idx(roles)
        m = self.eligible[np.ix_(s_idx, t_idx)]
        a, b = np.nonzero(m)
        si, ti = s_idx[a], t_idx[b]
        if max_n is not None and si.size > max_n:
            pick = np.sort(np.random.default_rng(seed).choice(si.size, max_n, replace=False))
            si, ti = si[pick], ti[pick]
        return si.astype(np.int64), ti.astype(np.int64)

    def target_doy(self, ti: np.ndarray, h: int) -> np.ndarray:
        dates = self.data.cube.dates[ti] + pd.to_timedelta(h, "D")
        return dates.dayofyear.to_numpy() - 1

    def to_model_space(self, q_anom: np.ndarray, si: np.ndarray, ti: np.ndarray, v: str) -> np.ndarray:
        """(n, H, Q) standardised-anomaly quantiles -> model-space quantiles for variable v."""
        k = self.variables.index(v)
        out = np.empty_like(q_anom, dtype=np.float32)
        for hk, h in enumerate(self.horizons):
            d = self.target_doy(ti, h)
            out[:, hk] = self.cm[si, d, k][:, None] + self.cs_eff[si, d, k][:, None] * q_anom[:, hk]
        return out


# ----------------------------------------------------------------------------- model

def _build_model(n_dyn: int, n_static: int, n_huc: int, V: int, H: int, Q: int, cfg: LSTMConfig,
                 horizons=HORIZONS):
    import torch
    from torch import nn

    class QuantileLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.V, self.H, self.Q = V, H, Q
            self.huc = nn.Embedding(n_huc, cfg.huc_emb)
            self.static_net = nn.Sequential(nn.Linear(n_static + cfg.huc_emb, cfg.static_hidden), nn.ReLU())
            self.inp = nn.Linear(n_dyn + cfg.static_hidden, cfg.hidden)
            self.lstm = nn.LSTM(cfg.hidden, cfg.hidden, cfg.layers, batch_first=True,
                                dropout=cfg.dropout if cfg.layers > 1 else 0.0)
            self.drop = nn.Dropout(cfg.dropout)
            self.head = nn.Sequential(nn.Linear(cfg.hidden + cfg.static_hidden, cfg.head_hidden), nn.ReLU(),
                                      nn.Dropout(cfg.dropout), nn.Linear(cfg.head_hidden, V * H * Q))
            self.anchor = bool(cfg.anchor_last)
            if self.anchor:  # damped-anomaly prior, learnable per (variable, horizon); init 0.9^h
                hs = torch.tensor(horizons, dtype=torch.float32)
                self.anchor_w = nn.Parameter(0.9 ** hs[None, :].repeat(V, 1))
            with torch.no_grad():  # start near N(0,1) quantiles: q05 = -1.645, spacing ~0.18
                last = self.head[-1]
                last.weight.mul_(0.1)
                b = torch.full((V, H, Q), float(np.log(np.expm1(3.29 / (Q - 1)))))
                b[..., 0] = -1.645
                last.bias.copy_(b.reshape(-1))

        def encode(self, x_dyn, x_static, huc):
            st = self.static_net(torch.cat([x_static, self.huc(huc)], dim=-1))
            x = torch.cat([x_dyn, st[:, None, :].expand(-1, x_dyn.shape[1], -1).to(x_dyn.dtype)], dim=-1)
            out, _ = self.lstm(self.inp(x))
            return torch.cat([self.drop(out[:, -1]), st.to(out.dtype)], dim=-1)

        def last_anomaly(self, x_dyn):
            """Latest observed standardised anomaly in the final 4 days of the window (issue day
            inclusive, i.e. <= 3 days stale), 0 if none — inputs only, so causal."""
            V = self.V
            a = x_dyn[:, -4:, :V].float()
            m = x_dyn[:, -4:, 2 * V:3 * V].float() > 0.5
            idx = torch.arange(4, device=x_dyn.device)[None, :, None].expand_as(m)
            pos = torch.where(m, idx, torch.full_like(idx, -1)).max(dim=1).values      # (B, V)
            val = torch.gather(a, 1, pos.clamp(min=0)[:, None, :]).squeeze(1)
            return torch.where(pos >= 0, val, torch.zeros_like(val))

        def quantiles(self, emb, a_last=None):
            raw = self.head(emb).float().view(-1, self.V, self.H, self.Q)
            inc = torch.nn.functional.softplus(raw[..., 1:])
            q = torch.cat([raw[..., :1], raw[..., :1] + torch.cumsum(inc, dim=-1)], dim=-1)
            if self.anchor and a_last is not None:  # same shift for all quantiles: stays monotone
                q = q + (self.anchor_w[None] * a_last[:, :, None])[..., None]
            return q

        def forward(self, x_dyn, x_static, huc):
            return self.quantiles(self.encode(x_dyn, x_static, huc), self.last_anomaly(x_dyn))

    return QuantileLSTM()


def pinball_loss(q, y, taus, balance: bool = True):
    """q (B, V, H, Q), y (B, V, H) with NaN = missing; masked mean pinball loss."""
    import torch

    m = torch.isfinite(y)
    yy = torch.where(m, y, torch.zeros_like(y))
    diff = yy[..., None] - q
    loss = torch.maximum(taus * diff, (taus - 1) * diff).mean(dim=-1)  # (B, V, H)
    loss = loss * m
    if not balance:
        return loss.sum() / m.sum().clamp(min=1)
    per_var = loss.sum(dim=(0, 2)) / m.sum(dim=(0, 2)).clamp(min=1)
    present = (m.sum(dim=(0, 2)) > 0).float()
    return (per_var * present).sum() / present.sum().clamp(min=1)


class SondeLSTM:
    def __init__(self, cfg: LSTMConfig | None = None, horizons=HORIZONS, variables=TARGETS):
        self.cfg = cfg or LSTMConfig()
        self.horizons = tuple(horizons)
        self.variables = list(variables)
        self.model = None
        self.tensors: SondeTensors | None = None
        self.history: list[dict] = []
        self.device = None

    # --- tensors on device
    def _to_device(self, ten: SondeTensors):
        import torch

        C = self.cfg.context
        S, _, F = ten.dyn.shape
        pad = np.zeros((S, C - 1, F), dtype=np.float16)
        self._dyn = torch.from_numpy(np.concatenate([pad, ten.dyn], axis=1)).to(self.device)
        self._tgt = torch.from_numpy(ten.targets_padded).to(self.device)
        self._static = torch.from_numpy(ten.static).to(self.device)
        self._huc = torch.from_numpy(ten.huc).to(self.device)
        self._ar = torch.arange(C, device=self.device)
        self._h = torch.tensor(self.horizons, device=self.device)
        self._taus = torch.tensor(QUANTILES, dtype=torch.float32, device=self.device)

    def _batch(self, s, t):
        """Window of the C days ending at issue index t (inclusive) for sites s."""
        x = self._dyn[s[:, None], t[:, None] + self._ar[None, :]].float()   # padded offset C-1
        y = self._tgt[s[:, None], t[:, None] + self._h[None, :]]              # (B, H, V)
        return x, self._static[s], self._huc[s], y.permute(0, 2, 1)

    def fit(self, data: SondeData, *, device: str = "cuda", log=print, checkpoint_path=None) -> dict:
        import torch

        from ...core.seed import seed_everything

        cfg = self.cfg
        seed_everything(cfg.seed)
        self.device = torch.device(device)
        ten = SondeTensors(data, variables=self.variables, horizons=self.horizons, context=cfg.context,
                           scale_floor_frac=cfg.scale_floor_frac, anom_clip=cfg.anom_clip,
                           use_weather=cfg.use_weather)
        self.tensors = ten
        self.variables = ten.variables
        self._to_device(ten)
        tr_si, tr_ti = ten.samples("train", ("train",))
        va_si, va_ti = ten.samples("val", ("train", "val"), stride=cfg.val_stride_days,
                                   max_n=cfg.val_max_samples, seed=cfg.seed)
        if tr_si.size == 0 or va_si.size == 0:
            raise RuntimeError(f"no training ({tr_si.size}) or validation ({va_si.size}) samples")
        log(f"[lstm] train samples {tr_si.size} ({np.unique(tr_si).size} sites), "
            f"val samples {va_si.size} ({np.unique(va_si).size} sites)")
        tr_s, tr_t = torch.from_numpy(tr_si).to(self.device), torch.from_numpy(tr_ti).to(self.device)
        self.model = _build_model(ten.n_dyn, ten.static.shape[1], ten.n_huc, len(self.variables),
                                  len(self.horizons), len(QUANTILES), cfg, self.horizons).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, cfg.patience // 2))
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and self.device.type == "cuda")
        best, best_state, bad = np.inf, None, 0
        t0 = time.time()
        gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        for ep in range(cfg.max_epochs):
            self.model.train()
            tot, n = 0.0, 0
            for _ in range(cfg.steps_per_epoch):
                k = torch.randint(0, tr_s.numel(), (cfg.batch_size,), device=self.device, generator=gen)
                x, xs, hc, y = self._batch(tr_s[k], tr_t[k])
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                    q = self.model(x, xs, hc)
                loss = pinball_loss(q.float(), y, self._taus, cfg.balance_variables)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                tot += float(loss.detach())
                n += 1
            vl = self.val_loss(va_si, va_ti)
            sched.step(vl)
            rec = {"epoch": ep, "train_loss": tot / max(n, 1), "val_loss": vl,
                   "lr": opt.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60}
            self.history.append(rec)
            log(f"[lstm] epoch {ep}: train {rec['train_loss']:.4f} val {vl:.4f} lr {rec['lr']:.2e}")
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if checkpoint_path:
                    self.save(checkpoint_path, state=best_state)
            else:
                bad += 1
                if bad >= cfg.patience:
                    log(f"[lstm] early stop at epoch {ep} (best val {best:.4f})")
                    break
            if cfg.max_minutes and (time.time() - t0) / 60 > cfg.max_minutes:
                log(f"[lstm] time budget reached at epoch {ep}")
                break
        if best_state is None:
            raise RuntimeError("validation loss never finite; no checkpoint selected")
        self.model.load_state_dict(best_state)
        return {"best_val_loss": best, "epochs": len(self.history), "n_train_samples": int(tr_si.size),
                "n_val_samples": int(va_si.size), "n_params": int(sum(p.numel() for p in self.model.parameters()))}

    def val_loss(self, si: np.ndarray, ti: np.ndarray, bs: int = 2048) -> float:
        import torch

        self.model.eval()
        tot, w = 0.0, 0
        with torch.no_grad():
            for a in range(0, si.size, bs):
                s = torch.from_numpy(si[a:a + bs]).to(self.device)
                t = torch.from_numpy(ti[a:a + bs]).to(self.device)
                x, xs, hc, y = self._batch(s, t)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                    enabled=self.cfg.amp and self.device.type == "cuda"):
                    q = self.model(x, xs, hc)
                tot += float(pinball_loss(q.float(), y, self._taus, self.cfg.balance_variables)) * s.numel()
                w += s.numel()
        return tot / max(w, 1)

    def _forward_np(self, si: np.ndarray, ti: np.ndarray, *, embed: bool = False, bs: int = 2048) -> np.ndarray:
        import torch

        self.model.eval()
        outs = []
        with torch.no_grad():
            for a in range(0, si.size, bs):
                s = torch.from_numpy(np.asarray(si[a:a + bs], dtype=np.int64)).to(self.device)
                t = torch.from_numpy(np.asarray(ti[a:a + bs], dtype=np.int64)).to(self.device)
                x, xs, hc, _ = self._batch(s, t)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                    enabled=self.cfg.amp and self.device.type == "cuda"):
                    e = self.model.encode(x, xs, hc)
                    out = e if embed else self.model.quantiles(e, self.model.last_anomaly(x))
                outs.append(out.float().cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0,))

    def predict_rows(self, rows_by_var: dict[str, Rows]) -> dict[str, np.ndarray]:
        """Model-space quantiles (n, H, 19) for every variable's common rows (one forward pass per
        unique site/issue pair)."""
        keys = np.concatenate([r.si.astype(np.int64) * 100_000 + r.ti for r in rows_by_var.values()])
        uniq, inv = np.unique(keys, return_inverse=True)
        si, ti = uniq // 100_000, uniq % 100_000
        qa = self._forward_np(si, ti)  # (n_uniq, V, H, Q)
        out, a = {}, 0
        for v, r in rows_by_var.items():
            idx = inv[a:a + len(r)]
            a += len(r)
            if v not in self.variables:
                out[v] = np.full((len(r), len(self.horizons), len(QUANTILES)), np.nan, np.float32)
                continue
            k = self.variables.index(v)
            out[v] = self.tensors.to_model_space(qa[idx, k], si[idx], ti[idx], v)
        return out

    def trained_variables(self, min_sites: int = 3) -> list[str]:
        """Variables with >= ``min_sites`` train-role sites contributing training targets."""
        ten = self.tensors
        tr = self.tensors.data.site_idx(("train",))
        t_idx = self.tensors.data.issue_idx("train", max(self.horizons))
        out = []
        for k, v in enumerate(self.variables):
            has = np.isfinite(ten.targets[np.ix_(tr, t_idx)][..., k]).any(axis=1)
            if int(has.sum()) >= min_sites:
                out.append(v)
        return out

    def embed(self, si: np.ndarray, ti: np.ndarray) -> np.ndarray:
        return self._forward_np(si, ti, embed=True)

    def save(self, path, state=None) -> None:
        import torch

        torch.save({"state_dict": state or self.model.state_dict(), "cfg": asdict(self.cfg),
                    "variables": self.variables, "horizons": list(self.horizons), "history": self.history,
                    "frozen": self.tensors.frozen_stats() if self.tensors else None}, path)

    def load_state(self, path, data: SondeData, device: str = "cuda") -> None:
        """Rebuild tensors for ``data`` (per-site statistics from each site's own training-years
        climatology; cross-site constants frozen from training) and load weights."""
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = LSTMConfig(**ck["cfg"])
        self.horizons, self.variables = tuple(ck["horizons"]), list(ck["variables"])
        self.device = torch.device(device)
        ten = SondeTensors(data, variables=self.variables, horizons=self.horizons, context=self.cfg.context,
                           scale_floor_frac=self.cfg.scale_floor_frac, anom_clip=self.cfg.anom_clip,
                           frozen=ck.get("frozen"), use_weather=self.cfg.use_weather)
        self.tensors = ten
        self._to_device(ten)
        self.model = _build_model(ten.n_dyn, ten.static.shape[1], ten.n_huc, len(self.variables),
                                  len(self.horizons), len(QUANTILES), self.cfg, self.horizons).to(self.device)
        self.model.load_state_dict(ck["state_dict"])
        self.history = ck.get("history", [])
