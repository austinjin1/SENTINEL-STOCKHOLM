# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Common trainer of the architecture-v2 neural arms (D1 GSO-MT, its ablations, R-TD, D2 ThermOx-T); amendment §7.

AdamW, 365-day windows, loss on the last 90 days; a share ``grab_share`` of every batch is grab-station windows
(station weight ∝ √rows, capped), the rest dense-site windows (``target_share`` sampling as the reference recipe).
Early stopping on J (:mod:`.criterion`) computed on the inner sonde sites and the inner grab set; per-epoch resume
checkpoints and a per-job time budget (:class:`ResumeNeeded`; jobs stay ≤ 2 h). ``refit_epochs=E`` trains on
fit + inner rows for exactly E epochs with no validation (the evaluated models). Arrays stay in host memory."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np

from . import criterion as C
from .data import DAILY, Scales, V2Train, fit_scales, make_inputs, standardise_labels
from .nets import N_DAILY, PAD, _torch, daily_stats, inst_point, loss_terms, make_net, n_params, total_loss


class ResumeNeeded(Exception):
    """Raised after a resume checkpoint was written because the job's time budget is used up."""


@dataclasses.dataclass
class TrainCfg:
    arm: str = "d1"
    net: dict = dataclasses.field(default_factory=lambda: {"hidden": 192, "heads": "diel", "static": "film"})
    seq_len: int = 365
    predict_last: int = 90
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 1e-4
    steps_per_epoch: int = 200
    max_epochs: int = 60
    patience: int = 6
    min_delta: float = 1e-4
    grad_clip: float = 1.0
    grab_share: float | None = None     # required: no default in code (lead note 2026-11-10); set by the config
    grab_weight_cap_obs: int = 400
    variables: tuple = ("T", "DO")
    aux_w: dict | None = None
    lag: int = 0
    seed: int = 0


def _valid_ends(obs_any: np.ndarray, seq_len: int, last: int) -> list:
    """Per site: window ends e ≥ seq_len − 1 whose last ``last`` days hold ≥ 1 observation."""
    S, T = obs_any.shape
    cs = np.concatenate([np.zeros((S, 1), np.int64), np.cumsum(obs_any, axis=1)], axis=1)
    e = np.arange(T)
    cnt = cs[:, e + 1] - cs[:, np.maximum(e + 1 - last, 0)]
    valid = (cnt > 0) & (e >= seq_len - 1)[None, :]
    return [np.flatnonzero(valid[i]) for i in range(S)]


def _tiles(need: np.ndarray, seq_len: int, L: int) -> list:
    """(site row, end) windows whose last L days tile every needed day ≥ seq_len − 1 of each site (ends stepping back
    by L from the last needed day; windows without a needed day are skipped)."""
    out = []
    for i in range(need.shape[0]):
        t = np.flatnonzero(need[i])
        t = t[t >= seq_len - 1]
        if not len(t):
            continue
        e, lo = int(t[-1]), int(t[0])
        while e >= lo:
            if need[i, max(e - L + 1, 0):e + 1].any():
                out.append((i, e))
            e -= L
    return out


class Trainer:
    def __init__(self, cfg: TrainCfg, *, make=make_net):
        if cfg.grab_share is None:
            raise ValueError("TrainCfg.grab_share is required (no default in code)")
        self.cfg, self.make = cfg, make
        self.state = None
        self.scales: Scales | None = None
        self.history: list = []
        self.info: dict = {}
        self.n_dyn = self.n_static = None
        self.expect_inputs: tuple | None = None     # (n_dynamic, n_static) asserted when set (non-smoke runs: 30 / 191)

    def _make(self):
        """The network. A maker with a ``context`` attribute (D2 ThermOx-T) gets, as fourth argument, the train-only
        constants that turn standardised inputs back into physical units."""
        if hasattr(self.make, "context"):
            names = self.info["inputs"]
            return self.make(self.n_dyn, self.n_static, self.cfg.net,
                             self.make.context(self.scales, names["dynamic"], names["static"]))
        return self.make(self.n_dyn, self.n_static, self.cfg.net)

    # ------------------------------------------------------------------ arrays
    def _arrays(self, v2: V2Train):
        if self.scales is None:
            self.scales = fit_scales(v2)
        dyn, st, dn, sn = make_inputs(v2.data, v2.solar, v2.vaa, self.scales, lag=self.cfg.lag)
        Yd, inst, hourly = standardise_labels(v2, self.scales)
        self.n_dyn, self.n_static = dyn.shape[-1], st.shape[-1]
        if self.expect_inputs is not None and (self.n_dyn, self.n_static) != tuple(self.expect_inputs):
            raise RuntimeError(f"inputs {self.n_dyn} dynamic / {self.n_static} static; expected {self.expect_inputs}")
        self.info["inputs"] = {"dynamic": dn, "static": sn}
        return dyn, st, Yd, inst, hourly

    def _label_mask(self) -> np.ndarray:
        keep = np.array([k.split("_")[0] in self.cfg.variables for k in DAILY[:N_DAILY]] + [True] * (len(DAILY) - N_DAILY))
        return keep

    # ------------------------------------------------------------------ training
    def fit(self, v2: V2Train, *, refit_epochs: int | None = None, device="cuda", log=print, checkpoint_path=None,
            resume_path=None, job_minutes: float | None = None, max_steps: int | None = None,
            assert_finite: bool = False):
        torch = _torch()
        cfg = self.cfg
        if not (v2.data.roles == "train").all():
            raise ValueError("arch-v2 arms are fitted on train-role locations only")
        rs = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        dyn, st, Yd, inst, hourly = self._arrays(v2)
        S, T = Yd.shape[:2]
        L, Tseq, B = cfg.predict_last, cfg.seq_len, cfg.batch_size
        vmask = self._label_mask()
        Yd = np.where(vmask[None, None, :], Yd, np.nan)
        vi = np.array([v in cfg.variables for v in ("T", "DO")])
        inst = np.where(vi[None, None, :], inst, np.nan)
        refit = refit_epochs is not None
        dense = np.r_[v2.fit_idx, v2.inner_idx] if refit else v2.fit_idx
        gidx = np.r_[v2.grab_fit_idx, v2.grab_inner_idx].astype(int) if refit else v2.grab_fit_idx
        use_grab = cfg.grab_share > 0 and len(gidx) > 0
        if cfg.grab_share > 0 and not len(gidx):
            raise ValueError("grab_share > 0 but the data hold no grab stations")
        Md = np.isfinite(Yd[dense])
        ends_d = _valid_ends(Md.any(-1), Tseq, L)
        tot = Md.sum(axis=(0, 1)).astype(float)
        share = (Md.sum(axis=1) / np.maximum(tot, 1)[None, :]).sum(axis=1)
        w_d = np.where([len(e) > 0 for e in ends_d], share, 0.0)
        if w_d.sum() <= 0:
            raise RuntimeError("no dense training windows")
        w_d = w_d / w_d.sum()
        n_g = int(round(cfg.grab_share * B)) if use_grab else 0
        n_d = B - n_g
        if use_grab:
            Mg = np.isfinite(inst[gidx])
            ends_g = _valid_ends(Mg.any(-1), Tseq, L)
            nobs = np.minimum(Mg.sum(axis=(1, 2)), cfg.grab_weight_cap_obs).astype(float)
            w_g = np.where([len(e) > 0 for e in ends_g], np.sqrt(nobs), 0.0)
            if w_g.sum() <= 0:
                raise RuntimeError("no grab windows")
            w_g = w_g / w_g.sum()
        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        net = self._make().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        offs = np.arange(-Tseq + 1, 1)
        Yz, Mz = np.nan_to_num(Yd), np.isfinite(Yd).astype("float32")
        Iz, Mi = np.nan_to_num(inst), np.isfinite(inst).astype("float32")
        hrow, hfrac_all = v2.hourly_row, v2.hour_frac
        use_hourly = len(hourly) > 0 and (hrow[dense] >= 0).any()
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)   # noqa: E731

        def batch(si, ei):
            si, ti = np.asarray(si), np.asarray(ei)[:, None] + offs[None, :]
            tl = ti[:, -L:]
            b = {"yd": to(Yz[si[:, None], tl]), "md": to(Mz[si[:, None], tl]), "yi": to(Iz[si[:, None], tl]),
                 "mi": to(Mi[si[:, None], tl]), "hi": to(v2.inst_hour[si[:, None], tl]),
                 "pc": to(v2.prov_class[si].astype("int64"))}
            hr = hrow[si]
            yh = np.full((len(si), L, 24, 2), np.nan, dtype="float32")
            fr = np.zeros(len(si), dtype="float32")
            has = hr >= 0
            if use_hourly and has.any():
                yh[has] = hourly[hr[has][:, None], tl[has]]
                fr[has] = hfrac_all[hr[has]]
                yh[..., ~vi] = np.nan
            b["yh"], b["mh"], b["hfrac"] = to(np.nan_to_num(yh)), to(np.isfinite(yh).astype("float32")), to(fr)
            return to(dyn[si[:, None], ti]), to(st[si]), b

        self._ctx = (dyn, st, dev)
        y_in = Yd[v2.inner_idx] if not refit and len(v2.inner_idx) else None
        gi = v2.grab_inner_idx if not refit and len(v2.grab_inner_idx) and vi.any() else np.array([], int)
        crit_f = self.scales.crit_factor()

        def crit():
            net.eval()
            pd_ = self._predict(net, v2.inner_idx, np.isfinite(y_in[:, :, :N_DAILY]).any(-1))
            js = C.j_sonde(pd_["daily"], y_in, crit_f, cfg.variables)
            jg = None
            if len(gi):
                pg = self._predict(net, gi, np.isfinite(inst[gi]).any(-1), hours=v2.inst_hour[gi])
                jg = C.j_grab(pg["inst"], inst[gi], cfg.variables)
            net.train()
            return C.combine(js, jg), js, jg

        best, best_state, bad, epoch0, used_min, t0 = np.inf, None, 0, 0, 0.0, time.time()
        self.history = []
        if resume_path and Path(resume_path).is_file():
            ck = torch.load(resume_path, map_location="cpu", weights_only=False)
            if ck["cfg"] != dataclasses.asdict(cfg) or ck["refit_epochs"] != refit_epochs:
                raise RuntimeError("resume checkpoint belongs to another configuration")
            net.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            rs.bit_generator.state = ck["rs"]
            torch.set_rng_state(ck["torch_cpu"])
            if dev.type == "cuda" and ck.get("torch_cuda") is not None:
                torch.cuda.set_rng_state(ck["torch_cuda"], dev)
            best, best_state, bad, epoch0 = ck["best"], ck["best_state"], ck["bad"], ck["epoch"] + 1
            self.history, used_min = ck["history"], ck["minutes_used"]
            log(f"[v2.{cfg.arm}] resumed after epoch {ck['epoch']} (best {best:.5f}, bad {bad}, {used_min:.1f} min used)")
        n_ep = refit_epochs if refit else cfg.max_epochs
        log(f"[v2.{cfg.arm}] params {n_params(net)}, dense {len(dense)}, grab {len(gidx) if use_grab else 0} "
            f"(share {cfg.grab_share if use_grab else 0}), refit {refit_epochs}, device {dev}")
        stopped = bool(self.history) and self.history[-1].get("stop", False)
        step_times = []
        for epoch in range(epoch0, n_ep):
            if stopped:
                break
            net.train()
            if hasattr(net, "set_epoch"):                       # D2: residual frozen during the warm-up epochs
                net.set_epoch(epoch)
            tr, tsum, n_skip = [], {}, 0
            for step in range(cfg.steps_per_epoch if max_steps is None else min(max_steps, cfg.steps_per_epoch)):
                ts = time.time()
                sk = rs.choice(len(dense), n_d, p=w_d)
                si = dense[sk]
                ei = np.array([ends_d[k][rs.integers(len(ends_d[k]))] for k in sk])
                if n_g:
                    gk = rs.choice(len(gidx), n_g, p=w_g)
                    si = np.r_[si, gidx[gk]]
                    ei = np.r_[ei, np.array([ends_g[k][rs.integers(len(ends_g[k]))] for k in gk])]
                x, s, b = batch(si, ei)
                out = net(x, s, L)
                terms = loss_terms(net, out, b, L)
                loss = total_loss(net, terms, cfg.grab_share if n_g else 0.0, variables=cfg.variables, aux_w=cfg.aux_w)
                if loss is None or not torch.isfinite(loss):
                    if assert_finite:
                        raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
                    n_skip += 1
                    continue
                opt.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                if not torch.isfinite(gn):
                    if assert_finite:
                        raise FloatingPointError(f"non-finite gradient at epoch {epoch} step {step}")
                    n_skip += 1
                    opt.zero_grad()
                    continue
                opt.step()
                tr.append(float(loss))
                for k, v in terms.items():
                    tsum.setdefault(k, []).append(float(v))
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                step_times.append(time.time() - ts)
            minutes = used_min + (time.time() - t0) / 60
            h = {"epoch": epoch, "train_loss": float(np.mean(tr)) if tr else float("nan"),
                 "train_terms": {k: float(np.mean(v)) for k, v in tsum.items()}, "minutes": minutes,
                 "s_per_step": float(np.median(step_times)) if step_times else None, "skipped_steps": n_skip}
            if n_skip > 0.1 * cfg.steps_per_epoch:
                raise FloatingPointError(f"{n_skip} non-finite steps in epoch {epoch}")
            if not refit:
                j, js, jg = crit()
                h.update({"J": j, "j_sonde": js["j_sonde"], "by_target": js["by_target"],
                          "j_grab": None if jg is None else jg["j_grab"],
                          "grab_by_variable": None if jg is None else jg["by_variable"]})
                jv = np.inf if j is None else j
                if jv < best - cfg.min_delta or not np.isfinite(best):
                    best, bad = jv, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        h["stop"] = stopped = True
                log(f"[v2.{cfg.arm}] epoch {epoch}: train {h['train_loss']:.4f} J {j} (sonde {js['j_sonde']}, grab "
                    f"{h['j_grab']}) best {best:.5f} bad {bad} ({minutes:.1f} min, {h['s_per_step']} s/step)")
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                log(f"[v2.{cfg.arm}] refit epoch {epoch}: train {h['train_loss']:.4f} ({minutes:.1f} min)")
            self.history.append(h)
            self.state = best_state
            if checkpoint_path and best_state is not None:
                self.save(checkpoint_path)
            done = stopped or epoch == n_ep - 1
            if resume_path:
                tmp = str(resume_path) + ".tmp"
                torch.save({"epoch": epoch, "model": net.state_dict(), "opt": opt.state_dict(),
                            "rs": rs.bit_generator.state, "torch_cpu": torch.get_rng_state(),
                            "torch_cuda": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None,
                            "best": best, "best_state": best_state, "bad": bad, "history": self.history,
                            "minutes_used": minutes, "cfg": dataclasses.asdict(cfg), "refit_epochs": refit_epochs}, tmp)
                Path(tmp).replace(resume_path)
            if not done and job_minutes and (time.time() - t0) / 60 > job_minutes:
                log(f"[v2.{cfg.arm}] job budget {job_minutes} min used after epoch {epoch}; resume checkpoint written")
                raise ResumeNeeded(str(resume_path))
        self.state = best_state or {k: v.detach().cpu() for k, v in net.state_dict().items()}
        js_hist = [h.get("J") if h.get("J") is not None else np.inf for h in self.history]
        kb = int(np.argmin(js_hist)) if js_hist and not refit else len(self.history) - 1
        self.info.update({"arm": cfg.arm, "n_params": n_params(net), "epochs": len(self.history), "refit_epochs": refit_epochs,
                          "best_epoch": None if refit else kb, "best_J": None if refit else self.history[kb].get("J"),
                          "best_j_sonde": None if refit else self.history[kb].get("j_sonde"),
                          "best_j_grab": None if refit else self.history[kb].get("j_grab"),
                          "best_by_target": None if refit else self.history[kb].get("by_target"),
                          "n_dense": int(len(dense)), "n_grab": int(len(gidx) if use_grab else 0),
                          "s_per_step_median": float(np.median(step_times)) if step_times else None})
        return self.info

    # ------------------------------------------------------------------ inference
    def _predict(self, net, site_idx, need: np.ndarray, *, hours: np.ndarray | None = None, batch: int = 512) -> dict:
        """Daily statistics (n, T, 7) and, with ``hours`` (n, T, 2), instantaneous predictions (n, T, 2), in model
        space, on the days of ``need`` (n, T); NaN elsewhere."""
        torch = _torch()
        dyn, st, dev = self._ctx
        cfg = self.cfg
        L, Tseq = cfg.predict_last, cfg.seq_len
        site_idx = np.asarray(site_idx, int)
        n, T = need.shape
        daily = np.full((n, T, N_DAILY), np.nan, dtype="float32")
        inst = np.full((n, T, 2), np.nan, dtype="float32") if hours is not None else None
        wins = _tiles(need, Tseq, L)
        offs = np.arange(-Tseq + 1, 1)
        with torch.no_grad():
            for c in range(0, len(wins), batch):
                ch = wins[c:c + batch]
                ri = np.array([w[0] for w in ch])
                ti = np.array([w[1] for w in ch])[:, None] + offs[None, :]
                x = torch.from_numpy(np.ascontiguousarray(dyn[site_idx[ri][:, None], ti])).to(dev)
                s = torch.from_numpy(np.ascontiguousarray(st[site_idx[ri]])).to(dev)
                out = net(x, s, L)
                daily[ri[:, None], ti[:, -L:]] = daily_stats(out, L).cpu().numpy()
                if inst is not None:
                    hh = torch.from_numpy(np.ascontiguousarray(hours[ri[:, None], ti[:, -L:]])).to(dev)
                    inst[ri[:, None], ti[:, -L:]] = inst_point(out, L, hh).cpu().numpy()
        return {"daily": daily, "inst": inst}

    def predict(self, dyn, st, site_idx, need, *, hours=None, device="cuda") -> dict:
        """Predictions of the stored state on prepared inputs (see :func:`.data.make_inputs`)."""
        torch = _torch()
        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if (dyn.shape[-1], st.shape[-1]) != (self.n_dyn, self.n_static):
            raise ValueError("inputs differ from those the model was fitted on")
        net = self._make().to(dev)
        net.load_state_dict(self.state)
        net.eval()
        self._ctx = (dyn, st, dev)
        return self._predict(net, site_idx, need, hours=hours)

    # ------------------------------------------------------------------ io
    def save(self, path) -> None:
        torch = _torch()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": dataclasses.asdict(self.cfg), "state": self.state, "scales": self.scales.state(),
                    "n_dyn": self.n_dyn, "n_static": self.n_static, "history": self.history, "info": self.info}, path)

    @classmethod
    def load(cls, path, *, make=make_net) -> "Trainer":
        torch = _torch()
        d = torch.load(path, map_location="cpu", weights_only=False)
        cfg = d["cfg"]
        cfg["variables"] = tuple(cfg["variables"])
        t = cls(TrainCfg(**cfg), make=make)
        t.state, t.scales, t.history, t.info = d["state"], Scales.from_state(d["scales"]), d["history"], d["info"]
        t.n_dyn, t.n_static = d["n_dyn"], d["n_static"]
        return t
