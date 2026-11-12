# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Lever trainer for the M6 SOTA campaign (``docs/preregistration/hydromet_sota_protocol.md`` §2, §4).

:class:`SotaEALSTM` subclasses the frozen :class:`sentinel.models.hydromet.lstm.HydroMetEALSTM` (same network,
decoding, loss and prediction). With every lever option off, :meth:`SotaEALSTM.fit` consumes the random streams
exactly like the frozen ``fit`` and therefore reproduces it (``tests/test_hydromet_sota_lstm.py``). Additions:

* ``store="cpu"`` — the input/target arrays stay in host memory and each batch is gathered there (identical values;
  needed when tens of thousands of weak-label stations do not fit on a shared GPU);
* per-epoch **resume** checkpoints and a per-job time budget (``job_minutes``): the job saves its state after an epoch
  and raises :class:`ResumeNeeded`; the next job continues bit-for-bit (RNG states restored). Jobs stay ≤ 2 h;
* **weak-label stream** (lever 1): ``weak_share`` s of every batch comes from weak-label stations (drawn ∝ √n_obs);
  per target, loss = (1 − s)·dense-cell mean + s·weak-cell mean; then the frozen mean over targets;
* **auxiliary targets** (lever 2): extra Gaussian outputs (e.g. log-discharge), standardised on dense fit sites ×
  train block, loss += λ·NLL. Never inputs, never scored.

Inner windows and the inner (early-stopping) loss use the 7 M6 targets only, so the held-out-group criterion is the
same as the reference runs. ``inner_primary_mse`` (mean over the 4 primary endpoints) is logged per epoch.
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path

import numpy as np

from ..models.hydromet.data import TARGETS, HydroMetData, Normaliser
from ..models.hydromet.lstm import EALSTMConfig, HydroMetEALSTM, _torch, build_inputs, make_model
from ..models.hydromet.physics import pressure_atm_from_elevation

PRIMARY = ("temp_c_mean", "temp_c_7dadm", "do_mgl_mean", "do_mgl_min")


class ResumeNeeded(Exception):
    """Raised after a resume checkpoint was written because the job's time budget is used up."""


class SotaEALSTM(HydroMetEALSTM):
    name = "sota_ealstm"

    def __init__(self, cfg: EALSTMConfig, *, weak_share: float = 0.0, aux: dict | None = None, store: str = "gpu",
                 **kw):
        super().__init__(cfg, **kw)
        self.weak_share = float(weak_share)
        self.aux = dict(aux or {})            # {aux name: loss weight λ}
        self.aux_norm: dict = {}              # {aux name: (mean, std)} of the standardised aux target
        self.store = store
        if (self.weak_share or self.aux) and cfg.head != "gaussian":
            raise ValueError("lever options need the gaussian head")
        if cfg.physics_do and (self.weak_share or self.aux):
            raise ValueError("lever options are not combined with physics_do")

    # ------------------------------------------------------------------ helpers
    def _model_cfg(self) -> EALSTMConfig:
        return dataclasses.replace(self.cfg, targets=tuple(self.cfg.targets) + tuple(self.aux))

    def _per_target(self, out, y, m, p_atm):
        """Per-output Gaussian NLL sums and cell counts over the batch (gaussian head)."""
        torch = _torch()
        mu, sigma = self._decode(out, p_atm)
        yy = torch.where(m > 0, y, torch.zeros_like(y))
        z = (yy - mu) / sigma
        l = torch.log(sigma) + 0.5 * z ** 2 + 0.5 * math.log(2 * math.pi)
        return (l * m).sum(dim=(0, 1)), m.sum(dim=(0, 1))

    def _combined_loss(self, out_d, yd, md, pd_, out_w, yw, mw, pw):
        """Lever loss: per WQ target (1 − s)·dense mean + s·weak mean (missing part drops out), mean over the WQ
        targets present, + Σ λ_a · dense aux NLL."""
        torch = _torch()
        V = len(self.cfg.targets)
        sd, nd = self._per_target(out_d, yd, md, pd_)
        terms = []
        if out_w is not None:
            sw, nw = self._per_target(out_w[..., :V, :], yw, mw, pw)
        for j in range(V):
            parts, wts = [], []
            if nd[j] >= 1:
                parts.append(sd[j] / nd[j])
                wts.append(1.0 - self.weak_share if out_w is not None else 1.0)
            if out_w is not None and nw[j] >= 1:
                parts.append(sw[j] / nw[j])
                wts.append(self.weak_share)
            if parts:
                tw = sum(wts)
                terms.append(sum(w * p for w, p in zip(wts, parts)) / tw)
        if not terms:
            return None
        loss = torch.stack(terms).mean()
        for k, (a, lam) in enumerate(self.aux.items()):
            j = V + k
            if nd[j] >= 1:
                loss = loss + float(lam) * sd[j] / nd[j]
        return loss

    # ------------------------------------------------------------------ training
    def fit(self, data: HydroMetData, fit_idx, inner_idx, *, weak_idx=None, aux_y: np.ndarray | None = None,
            device="cuda", log=print, checkpoint_path=None, resume_path=None, job_minutes: float | None = None):
        """As the frozen ``fit`` (train-role sites × train block; early stopping on inner groups), plus the lever
        options. ``weak_idx``: weak-label stations (train role, never inner). ``aux_y``: (S, T, n_aux) raw aux targets
        in model space (e.g. log-discharge), NaN where missing."""
        torch = _torch()
        cfg = self.cfg
        if cfg.physics_do and cfg.head != "gaussian":
            raise ValueError("physics_do requires the gaussian head")
        fit_idx, inner_idx = np.asarray(fit_idx, int), np.asarray(inner_idx, int)
        weak_idx = np.asarray([] if weak_idx is None else weak_idx, int)
        if not np.isin(data.roles[np.r_[fit_idx, inner_idx, weak_idx]], ["train"]).all():
            raise ValueError("EA-LSTM may only be fitted/early-stopped on train-role sites")
        if set(weak_idx) & (set(fit_idx) | set(inner_idx)):
            raise ValueError("weak-label stations must be disjoint from dense fit and inner sites")
        use_weak = self.weak_share > 0 and len(weak_idx) > 0
        if self.weak_share > 0 and not len(weak_idx):
            raise ValueError("weak_share > 0 but no weak-label stations")
        if self.aux and (aux_y is None or aux_y.shape[-1] != len(self.aux)):
            raise ValueError("aux targets requested but aux_y missing / wrong shape")
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
        V = Y.shape[-1]
        if self.aux:
            A = np.asarray(aux_y, dtype="float64").copy()
            A[:, ~tmask] = np.nan
            for k, a in enumerate(self.aux):
                x = A[fit_idx][:, :, k]
                x = x[np.isfinite(x)]
                mu_, sd_ = (float(x.mean()), float(max(x.std(), 1e-6))) if x.size > 1 else (0.0, 1.0)
                self.aux_norm[a] = (mu_, sd_)
                A[:, :, k] = (A[:, :, k] - mu_) / sd_
            if len(weak_idx):
                A[weak_idx] = np.nan                # aux labels are used at dense sites only
            Yfull = np.concatenate([Y, A.astype("float32")], axis=-1)
        else:
            Yfull = Y
        Mfull = np.isfinite(Yfull)
        p_atm = pressure_atm_from_elevation(data.static["elev_m"].to_numpy()).astype("float32")
        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        L, Tseq = cfg.predict_last, cfg.seq_len
        if self.store == "gpu":
            Xd = torch.tensor(dyn, device=dev)
            Xs = torch.tensor(st, device=dev)
            Yt = torch.tensor(np.nan_to_num(Yfull), device=dev)
            Mt = torch.tensor(Mfull.astype("float32"), device=dev)
            Pt = torch.tensor(p_atm, device=dev)
            offs = torch.arange(-Tseq + 1, 1, device=dev)
        else:   # host-memory store; batches gathered with numpy (identical values)
            Yh = np.nan_to_num(Yfull).astype("float32")
            Mh = Mfull.astype("float32")
            offs_np = np.arange(-Tseq + 1, 1)
        ends_fit = self._valid_ends(Mfull[fit_idx].any(-1), Tseq, L, tmask)
        tot = Mfull[fit_idx].sum(axis=(0, 1)).astype(float)
        share = (Mfull[fit_idx].sum(axis=1) / np.maximum(tot, 1)[None, :]).sum(axis=1)
        w_site = {"target_share": share, "sqrt_share": np.sqrt(share),
                  "uniform": np.ones(len(fit_idx))}[cfg.site_sampling]
        w_site = np.where([len(e) > 0 for e in ends_fit], w_site, 0.0)
        if w_site.sum() <= 0:
            raise RuntimeError("no training windows")
        w_site = w_site / w_site.sum()
        val_w = []
        ends_in = self._valid_ends(M[inner_idx].any(-1), Tseq, L, tmask)   # 7 M6 targets only
        for k, es in enumerate(ends_in):
            if len(es):
                last = es[-1]
                grid = np.arange(last, Tseq - 2, -L)
                grid = grid[np.isin(grid, es)]
                val_w += [(inner_idx[k], e) for e in grid]
        if len(val_w) > cfg.val_max_windows:
            val_w = [val_w[i] for i in np.sort(rs.choice(len(val_w), cfg.val_max_windows, replace=False))]
        if use_weak:
            ends_weak = self._valid_ends(M[weak_idx].any(-1), Tseq, L, tmask)
            nobs = M[weak_idx].sum(axis=(1, 2)).astype(float)
            w_weak = np.where([len(e) > 0 for e in ends_weak], np.sqrt(nobs), 0.0)
            if w_weak.sum() <= 0:
                raise RuntimeError("no weak-label windows")
            w_weak = w_weak / w_weak.sum()
            n_w = int(round(self.weak_share * cfg.batch_size))
            n_d = cfg.batch_size - n_w
        else:
            n_d, n_w = cfg.batch_size, 0
        mcfg = self._model_cfg() if self.aux else cfg
        model = make_model(self.n_dyn, self.n_static, mcfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        n_params = int(sum(p.numel() for p in model.parameters()))
        log(f"[sota_ealstm] params {n_params}, fit sites {len(fit_idx)}, inner sites {len(inner_idx)}, weak stations "
            f"{len(weak_idx)} (share {self.weak_share}), aux {self.aux}, val windows {len(val_w)}, device {dev}, "
            f"store {self.store}")

        def gather(si, ei, Vout):
            if self.store == "gpu":
                si_t = torch.as_tensor(si, device=dev)
                ti = torch.as_tensor(ei, device=dev)[:, None] + offs[None, :]
                return (Xd[si_t[:, None], ti], Xs[si_t], Yt[si_t[:, None], ti[:, -L:]][..., :Vout],
                        Mt[si_t[:, None], ti[:, -L:]][..., :Vout], Pt[si_t])
            si = np.asarray(si)
            ti = np.asarray(ei)[:, None] + offs_np[None, :]
            to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)   # noqa: E731
            return (to(dyn[si[:, None], ti]), to(st[si]), to(Yh[si[:, None], ti[:, -L:]][..., :Vout]),
                    to(Mh[si[:, None], ti[:, -L:]][..., :Vout]), to(p_atm[si]))

        Vtot = Yfull.shape[-1]

        def val_loss():
            if not val_w:
                return float("nan"), float("nan")
            model.eval()
            tot_l, tot_n = 0.0, 0
            se, cnt = np.zeros(V), np.zeros(V)
            with torch.no_grad():
                for b in range(0, len(val_w), 512):
                    chunk = val_w[b:b + 512]
                    x, s, y, m, p = gather([c[0] for c in chunk], [c[1] for c in chunk], V)
                    out = model(x, s, L)[..., :V, :]
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

        best, best_state, bad, t0, epoch0 = np.inf, None, 0, time.time(), 0
        used_min = 0.0
        if resume_path and Path(resume_path).is_file():
            ck = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            rs.bit_generator.state = ck["rs"]
            torch.set_rng_state(ck["torch_cpu"])
            if torch.cuda.is_available() and ck.get("torch_cuda") is not None and dev.type == "cuda":
                torch.cuda.set_rng_state(ck["torch_cuda"])
            best, best_state, bad, epoch0 = ck["best"], ck["best_state"], ck["bad"], ck["epoch"] + 1
            self.history, used_min = ck["history"], ck["minutes_used"]
            self.aux_norm = ck.get("aux_norm", self.aux_norm)
            log(f"[sota_ealstm] resumed after epoch {ck['epoch']} (best {best:.4f}, bad {bad}, {used_min:.1f} min used)")
        stopped = bool(self.history) and self.history[-1].get("stop", False)
        for epoch in range(epoch0, cfg.max_epochs):
            if stopped:
                break
            model.train()
            tr = []
            for _ in range(cfg.steps_per_epoch):
                sk = rs.choice(len(fit_idx), n_d, p=w_site)
                ei = np.array([ends_fit[k][rs.integers(len(ends_fit[k]))] for k in sk])
                if not use_weak and not self.aux:      # frozen path (bitwise identical to HydroMetEALSTM.fit)
                    x, s, y, m, p = gather(fit_idx[sk], ei, V)
                    if cfg.static_noise > 0:
                        s = s + cfg.static_noise * torch.randn_like(s)
                    loss = self._loss(model(x, s, L), y, m, p)
                else:
                    x, s, y, m, p = gather(fit_idx[sk], ei, Vtot)
                    if cfg.static_noise > 0:
                        s = s + cfg.static_noise * torch.randn_like(s)
                    out_d = model(x, s, L)
                    out_w = yw = mw = pw = None
                    if use_weak:
                        kw_ = rs.choice(len(weak_idx), n_w, p=w_weak)
                        ew = np.array([ends_weak[k][rs.integers(len(ends_weak[k]))] for k in kw_])
                        xw, sw, yw, mw, pw = gather(weak_idx[kw_], ew, V)
                        if cfg.static_noise > 0:
                            sw = sw + cfg.static_noise * torch.randn_like(sw)
                        out_w = model(xw, sw, L)
                    loss = self._combined_loss(out_d, y, m, p, out_w, yw, mw, pw)
                if loss is None or not torch.isfinite(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                tr.append(float(loss))
            v_nll, v_mse = val_loss()
            vl = v_mse if cfg.early_stop_metric == "mse" else v_nll
            byt = dict(getattr(self, "_last_val_mse_by_target", {}))
            prim = [byt.get(v) for v in PRIMARY]
            minutes = used_min + (time.time() - t0) / 60
            self.history.append({"epoch": epoch, "train_loss": float(np.mean(tr)) if tr else float("nan"),
                                 "inner_val_nll": v_nll, "inner_val_mse": v_mse, "inner_val_loss": vl,
                                 "inner_val_mse_by_target": byt,
                                 "inner_primary_mse": float(np.mean(prim)) if all(x is not None for x in prim) else None,
                                 "minutes": minutes})
            log(f"[sota_ealstm] epoch {epoch}: train {self.history[-1]['train_loss']:.4f} inner nll {v_nll:.4f} "
                f"mse {v_mse:.4f} primary {self.history[-1]['inner_primary_mse']} ({minutes:.1f} min)")
            if vl < best - 1e-4 or not np.isfinite(best):
                best, bad = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if checkpoint_path:
                    self.state = best_state
                    self.save(checkpoint_path)
            else:
                bad += 1
                if bad >= cfg.patience:
                    self.history[-1]["stop"] = True
                    stopped = True
            if cfg.max_minutes and minutes > cfg.max_minutes:
                log("[sota_ealstm] time budget reached")
                self.history[-1]["stop"] = True
                stopped = True
            done = stopped or epoch == cfg.max_epochs - 1
            if resume_path:
                torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                            "rs": rs.bit_generator.state, "torch_cpu": torch.get_rng_state(),
                            "torch_cuda": torch.cuda.get_rng_state() if dev.type == "cuda" else None,
                            "best": best, "best_state": best_state, "bad": bad, "history": self.history,
                            "minutes_used": minutes, "aux_norm": self.aux_norm}, str(resume_path) + ".tmp")
                Path(str(resume_path) + ".tmp").replace(resume_path)
            if not done and job_minutes and (time.time() - t0) / 60 > job_minutes:
                log(f"[sota_ealstm] job budget {job_minutes} min used after epoch {epoch}; resume checkpoint written")
                raise ResumeNeeded(str(resume_path))
        self.state = best_state or {k: v.detach().cpu() for k, v in model.state_dict().items()}
        hist = [h["inner_val_loss"] for h in self.history]
        kbest = int(np.nanargmin(hist)) if hist else 0
        self.info.update({"n_params": n_params, "best_inner_val_loss": float(best), "epochs": len(self.history),
                          "best_epoch": kbest, "early_stop_metric": cfg.early_stop_metric,
                          "best_inner_primary_mse": self.history[kbest].get("inner_primary_mse") if hist else None,
                          "best_inner_mse_by_target": self.history[kbest].get("inner_val_mse_by_target") if hist else None,
                          "n_fit_sites": int(len(fit_idx)), "n_inner_sites": int(len(inner_idx)),
                          "n_weak_stations": int(len(weak_idx)), "weak_share": self.weak_share, "aux": self.aux,
                          "aux_norm": self.aux_norm})
        return self.info

    # ------------------------------------------------------------------ inference / io
    def predict(self, data: HydroMetData, site_idx, t_idx, *, device="cuda", return_dist: bool = False):
        """Frozen ``predict`` (WQ targets only); the aux head is built but its outputs are discarded."""
        if not self.aux:
            return super().predict(data, site_idx, t_idx, device=device, return_dist=return_dist)
        torch = _torch()
        mcfg = self._model_cfg()
        V = len(self.cfg.targets)
        full_model = make_model(self.n_dyn, self.n_static, mcfg)
        full_model.load_state_dict(self.state)

        class _Trim(torch.nn.Module):   # same forward, WQ outputs only
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x, s, out_last):
                return self.m(x, s, out_last)[..., :V, :]

        trimmed = _Trim(full_model)
        state = {f"m.{k}": v for k, v in self.state.items()}
        from ..models.hydromet import lstm as _L

        orig, saved = _L.make_model, self.state
        try:   # the frozen predict builds its network through make_model(): hand it the trimmed one
            _L.make_model = lambda n_dyn, n_static, cfg: trimmed
            self.state = state
            return super().predict(data, site_idx, t_idx, device=device, return_dist=return_dist)
        finally:
            _L.make_model = orig
            self.state = saved

    def save(self, path):
        torch = _torch()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": dataclasses.asdict(self.cfg), "state": self.state, "norm": dataclasses.asdict(self.norm),
                    "n_dyn": self.n_dyn, "n_static": self.n_static, "history": self.history, "info": self.info,
                    "sota": {"weak_share": self.weak_share, "aux": self.aux, "aux_norm": self.aux_norm,
                             "store": self.store}}, path)

    @classmethod
    def load(cls, path) -> "SotaEALSTM":
        torch = _torch()
        d = torch.load(path, map_location="cpu", weights_only=False)
        cfg = d["cfg"]
        cfg["quantiles"] = tuple(cfg["quantiles"])
        cfg["ratio_bounds"] = tuple(cfg["ratio_bounds"])
        cfg["targets"] = tuple(cfg["targets"])
        so = d.get("sota", {})
        m = cls(EALSTMConfig(**cfg), weak_share=so.get("weak_share", 0.0), aux=so.get("aux"), store=so.get("store", "gpu"),
                norm=Normaliser(**d["norm"]), state=d["state"], history=d["history"], n_dyn=d["n_dyn"],
                n_static=d["n_static"], info=d["info"])
        m.aux_norm = so.get("aux_norm", {})
        return m


assert set(PRIMARY) <= set(TARGETS)
