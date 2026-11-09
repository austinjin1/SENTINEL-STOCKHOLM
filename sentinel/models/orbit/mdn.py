# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Mixture Density Network baseline — our re-implementation of the Pahlevan et al. (2020, RSE 240:111604)
architecture, retrained on our folds (orbit_protocol.md §4.2.3):

* inputs: band reflectances -> RobustScaler (fit rows only), no log transform — exactly the preprocessing of the
  official MDN-STREAM MSI weights (their config.json: x_scalers = RobustScaler; y_scalers = LogTransformer +
  MinMaxScaler(-1, 1)); ``x_log=True`` is kept only as a documented variant;
* target: log, MinMax-scaled to (-1, 1) (fit rows only); one MDN per target (single output, the 2020 setting;
  the official 2022+ weights are a joint chl/TSS/CDOM model, loaded by :class:`OfficialMDN`);
* 5 hidden layers x 100 ReLU units, K = 5 Gaussian components, Adam lr 1e-3, L2 1e-3, batch 128,
  10,000 iterations; 10 independent initialisations trained jointly (vectorised);
* point estimate per init = mean of the component with the largest mixing weight; ensemble estimate =
  median over inits (as published). Intervals from the equal-weight pooled mixture of the inits.

This is NOT the official MDN-STREAM code or weights; it is labelled "MDN (re-impl.)" everywhere.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

X_FLOOR = 1e-5


class _EnsLinear(nn.Module):
    def __init__(self, e, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.empty(e, i, o))
        self.b = nn.Parameter(torch.zeros(e, 1, o))
        for k in range(e):
            nn.init.kaiming_uniform_(self.w.data[k].T, a=math.sqrt(5))

    def forward(self, x):          # x [E, B, i]
        return torch.baddbmm(self.b, x, self.w)


class EnsembleMDN(nn.Module):
    def __init__(self, n_in, n_ens=10, hidden=100, n_layers=5, k=5):
        super().__init__()
        layers, d = [], n_in
        for _ in range(n_layers):
            layers.append(_EnsLinear(n_ens, d, hidden))
            d = hidden
        self.hidden = nn.ModuleList(layers)
        self.out = _EnsLinear(n_ens, d, 3 * k)
        self.k = k

    def forward(self, x):
        h = x
        for l in self.hidden:
            h = torch.relu(l(h))
        o = self.out(h)
        logit, mu, s = o[..., :self.k], o[..., self.k:2 * self.k], o[..., 2 * self.k:]
        sigma = torch.nn.functional.softplus(s) + 1e-3
        return torch.log_softmax(logit, -1), mu, sigma


def mixture_nll(logw, mu, sigma, y):
    lp = logw - 0.5 * ((y[..., None] - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    return -torch.logsumexp(lp, -1)


class MDNRegressor:
    def __init__(self, n_ens=10, n_iter=10000, batch=128, lr=1e-3, l2=1e-3, seed=0, device="cpu", x_log=False):
        self.n_ens, self.n_iter, self.batch, self.lr, self.l2, self.seed, self.device = n_ens, n_iter, batch, lr, l2, seed, device
        self.x_log = x_log

    def _x(self, X):
        X = np.asarray(X, float)
        return np.log10(np.clip(X, X_FLOOR, None)) if self.x_log else X

    def fit(self, X, y_log10):
        from sklearn.preprocessing import RobustScaler

        m = np.isfinite(y_log10) & np.all(np.isfinite(X), 1)
        X, y = self._x(X[m]), y_log10[m]
        self.xs = RobustScaler().fit(X)
        self.ylo, self.yhi = float(y.min()), float(y.max())
        xs = torch.tensor(self.xs.transform(X), dtype=torch.float32, device=self.device)
        ys = torch.tensor(2 * (y - self.ylo) / (self.yhi - self.ylo + 1e-12) - 1, dtype=torch.float32, device=self.device)
        torch.manual_seed(self.seed)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        self.net = EnsembleMDN(X.shape[1], self.n_ens).to(self.device)
        # Keras L2(l2) adds l2*sum(w^2) to each model's loss (kernels AND biases in the official config) ->
        # coupled gradient 2*l2*w == torch Adam weight_decay=2*l2. Each init's loss is its own batch mean NLL;
        # summing over inits keeps the members independent (a mean would scale every member's NLL gradient by
        # 1/n_ens against an unscaled weight decay — self-review fix).
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=2 * self.l2)
        n = xs.shape[0]
        for it in range(self.n_iter):
            idx = torch.randint(0, n, (self.n_ens, self.batch), generator=g).to(self.device)
            logw, mu, sg = self.net(xs[idx])
            loss = mixture_nll(logw, mu, sg, ys[idx]).mean(1).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.n_fit = int(n)
        return self

    @torch.no_grad()
    def mixture(self, X):
        """Per-init mixtures in log10 units: (logw, mu, sigma) each [E, N, K] (NaN rows where X non-finite)."""
        X = np.asarray(X, float)
        ok = np.all(np.isfinite(X), 1)
        xs = np.zeros((len(X), X.shape[1]))
        xs[ok] = self.xs.transform(self._x(X[ok]))
        t = torch.tensor(xs, dtype=torch.float32, device=self.device)[None].expand(self.n_ens, -1, -1)
        logw, mu, sg = (a.cpu().numpy() for a in self.net(t.contiguous()))
        scale = (self.yhi - self.ylo) / 2
        mu = (mu + 1) * scale + self.ylo
        sg = sg * scale
        mu[:, ~ok] = np.nan
        return logw, mu, sg

    def predict(self, X):
        logw, mu, sg = self.mixture(X)
        k = np.argmax(logw, -1)
        est = np.take_along_axis(mu, k[..., None], -1)[..., 0]     # [E, N]
        return np.median(est, 0)

    def quantiles(self, X, qs=(0.025, 0.16, 0.5, 0.84, 0.975)):
        from .satwq import mixture_quantiles

        logw, mu, sg = self.mixture(X)
        E, N, K = mu.shape
        w = np.exp(logw).transpose(1, 0, 2).reshape(N, E * K) / E
        return mixture_quantiles(w, mu.transpose(1, 0, 2).reshape(N, E * K), sg.transpose(1, 0, 2).reshape(N, E * K), qs)


class OfficialMDN:
    """Official MDN-STREAM MSI weights (github.com/STREAM-RS/MDN-STREAM, Weights/MSI/1b79098d...zip; GPL-3.0),
    read as DATA with h5py/json (no pickle, no code from the archive is executed). Joint chl/TSS/CDOM model,
    10 rounds; inputs: Rrs (sr^-1) at MSI 443, 490, 560, 665, 705, 740, 783 nm (S2 B01-B07).

    Estimate per round = mean of the highest-prior component (MDN ``top`` estimate), inverse-scaled
    (MinMax^-1 then exp); ensemble = median over rounds (as published). Trained on in-situ data overlapping
    GLORIA and US lakes: never scored on GLORIA; on our matchups any overlap favours this baseline."""

    MSI_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07")
    PRODUCTS = ("chl", "tss", "cdom")

    def __init__(self, root):
        import json
        from pathlib import Path

        import h5py

        self.rounds = []
        for rd in sorted(Path(root).glob("Round_*")):
            cfg = json.loads((rd / "config.json").read_text())
            sx = cfg["scalerx"]["scalers"]
            sy = cfg["scalery"]["scalers"]
            assert [s["class_path"].split(".")[-1] for s in sx] == ["RobustScaler"]
            assert [s["class_path"].split(".")[-1] for s in sy] == ["LogTransformer", "MinMaxScaler"]
            with h5py.File(rd / "trained_model.h5", "r") as f:
                g = f["model_weights"]
                mc = json.loads(f.attrs["model_config"])
                names = [l["config"]["name"] for l in mc["config"]["layers"] if l["class_name"] == "Dense"]
                dense = [(g[f"{n}/{n}/kernel:0"][()], g[f"{n}/{n}/bias:0"][()]) for n in names]
                mix = [k for k in g if k.startswith("mixture_layer")]
                assert len(mix) == 1 and len(dense) == 5
                leaves = {}
                g[mix[0]].visititems(lambda nm, ob: leaves.__setitem__(nm.split("/")[-1], ob[()])
                                     if isinstance(ob, h5py.Dataset) else None)
                out = (leaves["kernel:0"], leaves["bias:0"])
            self.rounds.append({"dense": dense, "out": out, "n_mix": cfg["n_mix"], "n_targets": cfg["n_targets"],
                                "xc": np.array(sx[0]["state"]["center_"]), "xs": np.array(sx[0]["state"]["scale_"]),
                                "ymin": np.array(sy[1]["state"]["min_"]), "yscale": np.array(sy[1]["state"]["scale_"])})
        if not self.rounds:
            raise FileNotFoundError(f"no Round_* folders under {root}")

    def predict_log10(self, rrs):
        """rrs [N, 7] (sr^-1) -> {product: log10 estimate [N]} (median over rounds)."""
        rrs = np.asarray(rrs, float)
        ok = np.all(np.isfinite(rrs), 1)
        est = []
        for r in self.rounds:
            h = (np.where(ok[:, None], rrs, 0.0) - r["xc"]) / r["xs"]
            for w, b in r["dense"]:
                h = np.maximum(h @ w + b, 0.0)
            o = h @ r["out"][0] + r["out"][1]
            k, t = r["n_mix"], r["n_targets"]
            prior, mu = o[:, :k], o[:, k:k + k * t].reshape(-1, k, t)   # tf.stack(tf.split(mu, k, 1), 1)
            top = mu[np.arange(len(o)), np.argmax(prior, 1)]            # [N, t] scaled
            ln = (top - r["ymin"]) / r["yscale"]                         # MinMax inverse -> ln(value)
            est.append(ln / np.log(10.0))
        e = np.median(np.stack(est), 0)
        e[~ok] = np.nan
        return {p: e[:, i] for i, p in enumerate(self.PRODUCTS)}
