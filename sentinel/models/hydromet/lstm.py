# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Entity-aware LSTM (EA-LSTM; Kratzert et al. 2019 HESS) for SENTINEL-HydroMet.

* **Inputs.** Dynamic: the 11 daily forcings (:data:`.data.FORCING_VARS`, lag-shifted by
  ``forcing_lag_days``; standardised with train-role × train-block statistics) + 4 DOY harmonics.
  Static: site/StreamCat attributes (:class:`.data.Normaliser`). No sensor input of any kind; the
  column groups are checked with :func:`.data.assert_allowed_inputs`.
* **EA-LSTM cell.** The input gate depends on the static attributes only, i = σ(W_s s + b_s); the
  forget/output gates and cell candidate depend on the dynamic input and h_{t−1}. Optionally a
  static embedding is concatenated to h_t before the head (``static_to_head``).
* **Causality.** The output at day t is a function of inputs on days ≤ t only (recurrent, no
  look-ahead); with lag L those inputs are forcings of days ≤ t − L (tested).
* **Head.** Multi-target (:data:`.data.TARGETS`), standardised model space. ``gaussian``: μ and
  σ = softplus(·) + 1e-3, Gaussian NLL; ``quantile``: monotone quantiles (first + cumulative
  softplus increments), pinball loss. Masked loss on missing targets, averaged per target (balanced).
* **Physics-guided DO head (ablation, ``physics_do``).** DO_mean and DO_min are produced as
  DO_sat(T̂_w, P(elev)) × r_t, with T̂_w the model's own temperature output (°C, de-standardised),
  DO_sat Benson–Krause (:func:`.physics.do_sat_mgl`) and r_t = r_lo + (r_hi − r_lo)·σ(z_t) a learned
  ratio; σ stays a free output. Gaussian head only.
* **Training.** Windows of ``seq_len`` days ending in the train block at fit (train-role) sites; loss
  on the last ``predict_last`` days (targets in the train block only). Early stopping on inner
  train-role basin groups (never val-/test-role sites). Inference: consecutive windows whose last
  ``predict_last`` outputs tile the requested dates (≥ seq_len − predict_last days of warm-up).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .data import (CALENDAR_COLS, FORCING_VARS, TARGETS, HydroMetData, Normaliser, assert_allowed_inputs, doy_harmonics,
                   forcing_names)
from .features import shift_right
from .physics import pressure_atm_from_elevation

DO_TARGETS = ("do_mgl_mean", "do_mgl_min")


@dataclass
class EALSTMConfig:
    seq_len: int = 365
    predict_last: int = 90
    hidden: int = 128
    static_emb: int = 32
    dropout: float = 0.2
    static_to_head: bool = True
    head: str = "gaussian"                 # gaussian | quantile
    quantiles: tuple = (0.05, 0.25, 0.5, 0.75, 0.95)
    physics_do: bool = False
    detach_temp_for_do: bool = False
    ratio_bounds: tuple = (0.2, 1.6)
    initial_forget_bias: float = 3.0
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 30
    steps_per_epoch: int = 500
    patience: int = 5
    grad_clip: float = 1.0
    forcing_lag_days: int = 0
    val_max_windows: int = 4000
    early_stop_metric: str = "mse"         # "mse" (point error of mu/median, standardised, target-balanced) | "nll"
    static_noise: float = 0.0              # std of Gaussian noise added to standardised static inputs (training only)
    site_sampling: str = "target_share"    # "target_share" (rare targets up-weighted) | "sqrt_share" | "uniform"
    max_minutes: float | None = None
    targets: tuple = tuple(TARGETS)
    seed: int = 0


def _torch():
    import torch

    return torch


def build_inputs(data: HydroMetData, norm: Normaliser, lag: int) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Dynamic (S, T, D) and static (S, n_static) model inputs + their column names."""
    forc = np.stack([shift_right(data.forc[:, :, j].astype("float64"), lag) for j in range(data.forc.shape[-1])],
                    axis=-1)
    dyn = norm.forcing(forc)
    cal = doy_harmonics(data.dates).to_numpy(dtype="float32")
    dyn = np.concatenate([dyn, np.broadcast_to(cal[None], (dyn.shape[0],) + cal.shape)], axis=-1)
    fnames = [f"f_{v}_d" for v in forcing_names(data.forc.shape[-1])]
    dyn_names = fnames + list(CALENDAR_COLS)
    st = norm.static(data.static)
    groups = assert_allowed_inputs(dyn_names + list(norm.static_cols), static_cols=list(norm.static_cols),
                                   forcing_feature_cols=fnames)
    assert not groups.get("unknown")
    return dyn.astype("float32"), st, dyn_names, list(norm.static_cols)


def make_model(n_dyn: int, n_static: int, cfg: EALSTMConfig):
    torch = _torch()
    nn = torch.nn

    class EALSTM(nn.Module):
        def __init__(self):
            super().__init__()
            H = cfg.hidden
            self.H = H
            self.input_gate = nn.Linear(n_static, H)
            self.W = nn.Linear(n_dyn, 3 * H)
            self.U = nn.Linear(H, 3 * H, bias=False)
            with torch.no_grad():
                self.W.bias.zero_()
                self.W.bias[:H] = cfg.initial_forget_bias
            self.semb = nn.Sequential(nn.Linear(n_static, cfg.static_emb), nn.Tanh()) if cfg.static_to_head else None
            self.drop = nn.Dropout(cfg.dropout)
            nv = len(cfg.targets)
            self.nq = 2 if cfg.head == "gaussian" else len(cfg.quantiles)
            self.head = nn.Linear(H + (cfg.static_emb if cfg.static_to_head else 0), nv * self.nq)

        def forward(self, x, s, out_last: int):
            B, T, _ = x.shape
            i_gate = torch.sigmoid(self.input_gate(s))
            xw = self.W(x)
            h = x.new_zeros(B, self.H)
            c = x.new_zeros(B, self.H)
            hs = []
            for t in range(T):
                g = xw[:, t] + self.U(h)
                f, o, gg = g.chunk(3, dim=-1)
                c = torch.sigmoid(f) * c + i_gate * torch.tanh(gg)
                h = torch.sigmoid(o) * torch.tanh(c)
                if t >= T - out_last:
                    hs.append(h)
            Hs = self.drop(torch.stack(hs, dim=1))
            if self.semb is not None:
                e = self.semb(s)[:, None, :].expand(-1, Hs.shape[1], -1)
                Hs = torch.cat([Hs, e], dim=-1)
            return self.head(Hs).view(B, Hs.shape[1], len(cfg.targets), self.nq)

    return EALSTM()


@dataclass
class HydroMetEALSTM:
    cfg: EALSTMConfig
    norm: Normaliser | None = None
    state: dict | None = None
    history: list = field(default_factory=list)
    n_dyn: int = 0
    n_static: int = 0
    info: dict = field(default_factory=dict)

    name = "ealstm"

    # ------------------------------------------------------------------ outputs -> model space
    def _decode(self, raw, p_atm):
        """raw (B, L, V, nq) -> (mu, sigma) or quantiles in STANDARDISED model space."""
        torch = _torch()
        cfg = self.cfg
        if cfg.head == "quantile":
            q0 = raw[..., :1]
            inc = torch.nn.functional.softplus(raw[..., 1:])
            return torch.cat([q0, q0 + torch.cumsum(inc, dim=-1)], dim=-1), None
        mu = raw[..., 0]
        sigma = torch.nn.functional.softplus(raw[..., 1]) + 1e-3
        if cfg.physics_do:
            from .physics import do_sat_mgl

            jt = cfg.targets.index("temp_c_mean")
            tm, ts = (float(self.norm.target_mean[TARGETS.index("temp_c_mean")]),
                      float(self.norm.target_std[TARGETS.index("temp_c_mean")]))
            t_c = mu[..., jt] * ts + tm
            if cfg.detach_temp_for_do:
                t_c = t_c.detach()
            sat = do_sat_mgl(t_c.clamp(0.0, 40.0), p_atm[:, None])
            lo, hi = cfg.ratio_bounds
            cols = []
            for j, v in enumerate(cfg.targets):
                if v in DO_TARGETS:
                    k = TARGETS.index(v)
                    do = sat * (lo + (hi - lo) * torch.sigmoid(mu[..., j]))
                    cols.append((do - float(self.norm.target_mean[k])) / float(self.norm.target_std[k]))
                else:
                    cols.append(mu[..., j])
            mu = torch.stack(cols, dim=-1)
        return mu, sigma

    def _loss(self, out, y, m, p_atm):
        torch = _torch()
        mu, sigma = self._decode(out, p_atm)
        per_t = []
        for j in range(y.shape[-1]):
            mj = m[..., j]
            n = mj.sum()
            if n < 1:
                continue
            yj = torch.where(mj > 0, y[..., j], torch.zeros_like(y[..., j]))
            if self.cfg.head == "quantile":
                qs = torch.tensor(self.cfg.quantiles, device=y.device)
                diff = yj[..., None] - mu[..., j, :]
                l = torch.maximum(qs * diff, (qs - 1) * diff).mean(-1)
            else:
                z = (yj - mu[..., j]) / sigma[..., j]
                l = torch.log(sigma[..., j]) + 0.5 * z ** 2 + 0.5 * math.log(2 * math.pi)
            per_t.append((l * mj).sum() / n)
        return torch.stack(per_t).mean() if per_t else None

    # ------------------------------------------------------------------ windows
    @staticmethod
    def _valid_ends(obs_any: np.ndarray, seq_len: int, last: int, tmask: np.ndarray) -> list:
        """Per site: window end indices e (≥ seq_len−1, e in block) whose last ``last`` days hold ≥ 1
        observed target inside the block."""
        S, T = obs_any.shape
        ok = obs_any & tmask[None, :]
        cs = np.concatenate([np.zeros((S, 1), np.int64), np.cumsum(ok, axis=1)], axis=1)
        e = np.arange(T)
        cnt = cs[:, e + 1] - cs[:, np.maximum(e + 1 - last, 0)]
        valid = (cnt > 0) & (e >= seq_len - 1)[None, :] & tmask[None, :]
        return [np.flatnonzero(valid[i]) for i in range(S)]

    def fit(self, data: HydroMetData, fit_idx, inner_idx, *, device="cuda", log=print, checkpoint_path=None):
        torch = _torch()
        cfg = self.cfg
        if cfg.physics_do and cfg.head != "gaussian":
            raise ValueError("physics_do requires the gaussian head")
        fit_idx, inner_idx = np.asarray(fit_idx, int), np.asarray(inner_idx, int)
        if not np.isin(data.roles[np.r_[fit_idx, inner_idx]], ["train"]).all():
            raise ValueError("EA-LSTM may only be fitted/early-stopped on train-role sites")
        rs = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.norm = Normaliser.fit(data, fit_idx)
        dyn, st, dyn_names, st_names = build_inputs(data, self.norm, cfg.forcing_lag_days)
        self.n_dyn, self.n_static = dyn.shape[-1], st.shape[-1]
        self.info["inputs"] = {"dynamic": dyn_names, "static": st_names}
        tmask = data.block_mask("train")
        Y = np.stack([self.norm.target(v, data.y[:, :, data.vi(v)]) for v in cfg.targets], axis=-1).astype("float32")
        Y[:, ~tmask] = np.nan                       # targets only from the train block
        M = np.isfinite(Y)
        p_atm = pressure_atm_from_elevation(data.static["elev_m"].to_numpy()).astype("float32")
        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        Xd = torch.tensor(dyn, device=dev)
        Xs = torch.tensor(st, device=dev)
        Yt = torch.tensor(np.nan_to_num(Y), device=dev)
        Mt = torch.tensor(M.astype("float32"), device=dev)
        Pt = torch.tensor(p_atm, device=dev)
        L, Tseq = cfg.predict_last, cfg.seq_len
        ends_fit = self._valid_ends(M[fit_idx].any(-1), Tseq, L, tmask)
        # site sampling weight: per-target share of observations (rare targets are not drowned out)
        tot = M[fit_idx].sum(axis=(0, 1)).astype(float)
        share = (M[fit_idx].sum(axis=1) / np.maximum(tot, 1)[None, :]).sum(axis=1)
        w_site = {"target_share": share, "sqrt_share": np.sqrt(share),
                  "uniform": np.ones(len(fit_idx))}[cfg.site_sampling]
        w_site = np.where([len(e) > 0 for e in ends_fit], w_site, 0.0)
        if w_site.sum() <= 0:
            raise RuntimeError("no training windows")
        w_site = w_site / w_site.sum()
        # fixed inner-validation windows (tiling the train block at inner sites)
        val_w = []
        ends_in = self._valid_ends(M[inner_idx].any(-1), Tseq, L, tmask)
        for k, es in enumerate(ends_in):
            if len(es):
                last = es[-1]
                grid = np.arange(last, Tseq - 2, -L)
                grid = grid[np.isin(grid, es)]
                val_w += [(inner_idx[k], e) for e in grid]
        if len(val_w) > cfg.val_max_windows:
            val_w = [val_w[i] for i in np.sort(rs.choice(len(val_w), cfg.val_max_windows, replace=False))]
        model = make_model(self.n_dyn, self.n_static, cfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        n_params = int(sum(p.numel() for p in model.parameters()))
        log(f"[ealstm] params {n_params}, fit sites {len(fit_idx)}, inner sites {len(inner_idx)}, "
            f"val windows {len(val_w)}, device {dev}")
        offs = torch.arange(-Tseq + 1, 1, device=dev)

        def gather(si, ei):
            si_t = torch.as_tensor(si, device=dev)
            ti = torch.as_tensor(ei, device=dev)[:, None] + offs[None, :]
            x = Xd[si_t[:, None], ti]
            y = Yt[si_t[:, None], ti[:, -L:]]
            m = Mt[si_t[:, None], ti[:, -L:]]
            return x, Xs[si_t], y, m, Pt[si_t]

        def val_loss():
            """Inner-group (NLL-or-pinball loss, point MSE): MSE of mu (gaussian) / median (quantile) in
            standardised model space, averaged per target over observed cells, then over targets."""
            if not val_w:
                return float("nan"), float("nan")
            model.eval()
            tot_l, tot_n = 0.0, 0
            V = len(cfg.targets)
            se, cnt = np.zeros(V), np.zeros(V)
            with torch.no_grad():
                for b in range(0, len(val_w), 512):
                    chunk = val_w[b:b + 512]
                    x, s, y, m, p = gather([c[0] for c in chunk], [c[1] for c in chunk])
                    out = model(x, s, L)
                    l = self._loss(out, y, m, p)
                    if l is not None:
                        tot_l += float(l) * len(chunk)
                        tot_n += len(chunk)
                    mu, _ = self._decode(out, p)
                    pt = mu if cfg.head == "gaussian" else mu[..., len(cfg.quantiles) // 2]
                    se += (((pt - y) ** 2) * m).sum(dim=(0, 1)).cpu().numpy()
                    cnt += m.sum(dim=(0, 1)).cpu().numpy()
            model.train()
            ok = cnt > 0
            self._last_val_mse_by_target = {v: (float(se[j] / cnt[j]) if cnt[j] else None)
                                            for j, v in enumerate(cfg.targets)}
            return tot_l / max(tot_n, 1), float(np.mean(se[ok] / cnt[ok])) if ok.any() else float("nan")

        best, best_state, bad, t0 = np.inf, None, 0, time.time()
        for epoch in range(cfg.max_epochs):
            model.train()
            tr = []
            for _ in range(cfg.steps_per_epoch):
                sk = rs.choice(len(fit_idx), cfg.batch_size, p=w_site)
                ei = np.array([ends_fit[k][rs.integers(len(ends_fit[k]))] for k in sk])
                x, s, y, m, p = gather(fit_idx[sk], ei)
                if cfg.static_noise > 0:
                    s = s + cfg.static_noise * torch.randn_like(s)
                loss = self._loss(model(x, s, L), y, m, p)
                if loss is None or not torch.isfinite(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                tr.append(float(loss))
            v_nll, v_mse = val_loss()
            vl = v_mse if cfg.early_stop_metric == "mse" else v_nll
            self.history.append({"epoch": epoch, "train_loss": float(np.mean(tr)) if tr else float("nan"),
                                 "inner_val_nll": v_nll, "inner_val_mse": v_mse, "inner_val_loss": vl,
                                 "inner_val_mse_by_target": dict(getattr(self, "_last_val_mse_by_target", {})),
                                 "minutes": (time.time() - t0) / 60})
            log(f"[ealstm] epoch {epoch}: train {self.history[-1]['train_loss']:.4f} inner nll {v_nll:.4f} "
                f"mse {v_mse:.4f} [{cfg.early_stop_metric}] ({self.history[-1]['minutes']:.1f} min) by target "
                f"{ {k[:8]: round(x, 3) for k, x in self.history[-1]['inner_val_mse_by_target'].items() if x is not None} }")
            if vl < best - 1e-4 or not np.isfinite(best):
                best, bad = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if checkpoint_path:
                    self.state = best_state
                    self.save(checkpoint_path)
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
            if cfg.max_minutes and (time.time() - t0) / 60 > cfg.max_minutes:
                log("[ealstm] time budget reached")
                break
        self.state = best_state or {k: v.detach().cpu() for k, v in model.state_dict().items()}
        self.info.update({"n_params": n_params, "best_inner_val_loss": float(best), "epochs": len(self.history),
                          "early_stop_metric": cfg.early_stop_metric,
                          "n_fit_sites": int(len(fit_idx)), "n_inner_sites": int(len(inner_idx))})
        return self.info

    # ------------------------------------------------------------------ inference
    def predict(self, data: HydroMetData, site_idx, t_idx, *, device="cuda", return_dist: bool = False):
        """Point estimates (original units; median of the predictive distribution) for
        ``cfg.targets`` at (site_idx × t_idx); optionally model-space (mu, sigma) / quantiles."""
        torch = _torch()
        cfg = self.cfg
        site_idx, t_idx = np.asarray(site_idx, int), np.asarray(t_idx, int)
        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        model = make_model(self.n_dyn, self.n_static, cfg).to(dev)
        model.load_state_dict(self.state)
        model.eval()
        sub = data.subset(site_idx)
        dyn, st, _, _ = build_inputs(sub, self.norm, cfg.forcing_lag_days)
        p_atm = pressure_atm_from_elevation(sub.static["elev_m"].to_numpy()).astype("float32")
        L, Tseq = cfg.predict_last, cfg.seq_len
        tmax = int(t_idx.max())
        ends = np.arange(tmax, int(t_idx.min()) - 1, -L)
        ends = ends[ends >= Tseq - 1]
        S, V = len(site_idx), len(cfg.targets)
        nq = 2 if cfg.head == "gaussian" else len(cfg.quantiles)
        full = np.full((S, dyn.shape[1], V, nq), np.nan, dtype="float32")
        Xd = torch.tensor(dyn, device=dev)
        Xs = torch.tensor(st, device=dev)
        Pt = torch.tensor(p_atm, device=dev)
        offs = torch.arange(-Tseq + 1, 1, device=dev)
        pairs = [(i, e) for e in ends for i in range(S)]
        with torch.no_grad():
            for b in range(0, len(pairs), 1024):
                ch = pairs[b:b + 1024]
                si = torch.as_tensor([c[0] for c in ch], device=dev)
                ti = torch.as_tensor([c[1] for c in ch], device=dev)[:, None] + offs[None, :]
                out = model(Xd[si[:, None], ti], Xs[si], L)
                a, sd = self._decode(out, Pt[si])
                res = torch.stack([a, sd], dim=-1) if sd is not None else a
                res = res.cpu().numpy()
                for k, (i, e) in enumerate(ch):
                    full[i, e - L + 1:e + 1] = res[k]
        full = full[:, t_idx]
        point, dist = {}, {"mu": {}, "sigma": {}} if cfg.head == "gaussian" else {"q": {}}
        for j, v in enumerate(cfg.targets):
            k = TARGETS.index(v)
            if cfg.head == "gaussian":
                mu = full[:, :, j, 0] * self.norm.target_std[k] + self.norm.target_mean[k]
                sig = full[:, :, j, 1] * self.norm.target_std[k]
                dist["mu"][v], dist["sigma"][v] = mu, sig
                from .data import from_model

                point[v] = from_model(v, mu).astype("float32")
            else:
                q = full[:, :, j, :] * self.norm.target_std[k] + self.norm.target_mean[k]
                dist["q"][v] = q
                from .data import from_model

                point[v] = from_model(v, q[..., len(cfg.quantiles) // 2]).astype("float32")
        return (point, dist) if return_dist else point

    # ------------------------------------------------------------------ io
    def save(self, path):
        torch = _torch()
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "state": self.state, "norm": asdict(self.norm), "n_dyn": self.n_dyn,
                    "n_static": self.n_static, "history": self.history, "info": self.info}, path)

    @classmethod
    def load(cls, path) -> "HydroMetEALSTM":
        torch = _torch()
        d = torch.load(path, map_location="cpu", weights_only=False)
        cfg = d["cfg"]
        cfg["quantiles"] = tuple(cfg["quantiles"])
        cfg["ratio_bounds"] = tuple(cfg["ratio_bounds"])
        cfg["targets"] = tuple(cfg["targets"])
        return cls(EALSTMConfig(**cfg), Normaliser(**d["norm"]), d["state"], d["history"], d["n_dyn"], d["n_static"],
                   d["info"])
