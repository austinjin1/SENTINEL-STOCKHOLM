# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""tigramite wrappers with bounded parallelism (protocol §7).

Graph codes (uint8): 0 '' · 1 '-->' (i→j) · 2 '<--' · 3 'o-o' · 4 'x-x' (conflict) · 5 other LPCMCI marks
('o->', '<->', '<-o', …; adjacency without a definite i→j). ``graphs[b, i, j, τ]`` is the link from
X_i(t−τ) to X_j(t) in bootstrap replicate b (replicate 0 = the full-sample fit when requested).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

MISSING = 1.0e9
EXOG_DEFAULT = ("tair", "sw", "precip")
CODES = {"": 0, "-->": 1, "<--": 2, "o-o": 3, "x-x": 4}


def encode(graph: np.ndarray) -> np.ndarray:
    out = np.zeros(graph.shape, dtype=np.uint8)
    for s, c in CODES.items():
        out[graph == s] = c
    other = (out == 0) & (graph != "")
    # LPCMCI marks: 'o->' and '-->' style definite heads at j count as directed only for '-->'
    out[other] = 5
    out[graph == "o->"] = 6          # i (circle) *-> j: j is not an ancestor of i
    return out


def link_assumptions(names, tau_max: int, exogenous=("tair", "sw", "precip")) -> dict:
    """Weather variables exogenous: no links into them from river variables (any lag); contemporaneous
    weather–river edges oriented weather → river. Everything else left to discovery."""
    ex = {k for k, n in enumerate(names) if n in exogenous}
    la = {j: {} for j in range(len(names))}
    for j in range(len(names)):
        for i in range(len(names)):
            for tau in range(0, tau_max + 1):
                if i == j and tau == 0:
                    continue
                if j in ex and i not in ex:
                    if tau == 0:
                        la[j][(i, 0)] = "<?-"         # contemporaneous edge must point out of weather
                    continue                           # lagged river → weather forbidden
                if tau > 0:
                    la[j][(i, -tau)] = "-?>"
                elif i in ex and j not in ex:
                    la[j][(i, 0)] = "-?>"
                else:
                    la[j][(i, 0)] = "o?o"
    return la


def _dataframe(data: np.ndarray, names):
    from tigramite import data_processing as pp

    x = np.where(np.isfinite(data), data, MISSING)
    return pp.DataFrame(x, missing_flag=MISSING, var_names=list(names))


def _pcmci(data, names, verbosity=0):
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    return PCMCI(dataframe=_dataframe(data, names), cond_ind_test=ParCorr(significance="analytic"),
                 verbosity=verbosity)


def tested_mask(N: int, tau_max: int, link_assumptions: dict | None) -> np.ndarray:
    """(N, N, τ+1) bool: link tests entering the FDR family. Lagged links (incl. auto-lags) allowed by the
    link assumptions, and each contemporaneous unordered pair once (upper triangle, i < j)."""
    m = np.zeros((N, N, tau_max + 1), bool)
    if link_assumptions is None:
        m[:, :, 1:] = True
        m[:, :, 0] = np.triu(np.ones((N, N), bool), 1)
        return m
    for j, d in link_assumptions.items():
        for (i, lag), mark in d.items():
            tau = abs(lag)
            if tau == 0:
                if i != j:
                    m[min(i, j), max(i, j), 0] = True
            elif mark not in ("<--", "<?-"):
                m[i, j, tau] = True
    return m


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (q) of a 1-D array."""
    p = np.asarray(p, float)
    m = p.size
    if m == 0:
        return p
    o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1.0)
    return out


def fdr_prune(graph: np.ndarray, p_matrix: np.ndarray, alpha: float, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply FDR-BH to the GRAPH (review A1, 2026-11-08): tigramite's ``fdr_method`` only adjusts the
    returned p-matrix, so links were accepted at the uncorrected pc_alpha. Here BH runs over every tested
    link (``mask``), and every graph link with q > alpha is removed (both sides of a contemporaneous edge).
    Returns (pruned graph (str array), q-matrix (NaN outside the family))."""
    g = np.array(graph, dtype=object).copy()
    pm = np.where(np.isfinite(p_matrix), p_matrix, 1.0).copy()
    pm[:, :, 0] = np.maximum(pm[:, :, 0], pm[:, :, 0].T)   # contemporaneous: conservative symmetric p (LPCMCI)
    q = np.full(pm.shape, np.nan)
    q[mask] = bh_adjust(pm[mask])
    i0, j0 = np.nonzero(mask[:, :, 0])
    q[j0, i0, 0] = q[i0, j0, 0]                       # contemporaneous: symmetric
    drop = (g != "") & ~(q <= alpha)
    g[drop] = ""
    return g.astype("<U3"), q


def run_pcmciplus(data: np.ndarray, names, *, tau_max=7, pc_alpha=0.01, constrained=True, fdr="fdr_bh",
                  fdr_q: float | None = None, exogenous=EXOG_DEFAULT):
    """PCMCI+ full-sample fit; ``fdr_q`` (default = pc_alpha) is the BH level applied to the graph."""
    pc = _pcmci(data, names)
    la = link_assumptions(names, tau_max, exogenous) if constrained else None
    r = pc.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha, link_assumptions=la, fdr_method="none")
    g = r["graph"]
    if fdr not in (None, "none"):
        g, _ = fdr_prune(g, r["p_matrix"], pc_alpha if fdr_q is None else fdr_q, tested_mask(len(names), tau_max, la))
    return encode(g), r["val_matrix"].astype("float32"), r["p_matrix"].astype("float32")


def bagged_pcmciplus(data: np.ndarray, names, *, tau_max=7, pc_alpha=0.01, constrained=True, fdr="fdr_bh",
                     boot=50, block=10, seed=0, include_full=True, fdr_q: float | None = None,
                     exogenous=EXOG_DEFAULT):
    """Bagged-PCMCI+ (Debeire et al. 2024) with replicates run sequentially in-process. FDR-BH is applied
    to the graph of every replicate (:func:`fdr_prune`).

    Returns (graphs (B, N, N, τ+1) uint8, vals (B, N, N, τ+1) float32); replicate 0 is the full-sample
    fit when ``include_full``.
    """
    la = link_assumptions(names, tau_max, exogenous) if constrained else None
    args = dict(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha, link_assumptions=la, fdr_method="none")
    mask = tested_mask(len(names), tau_max, la)
    q_lev = pc_alpha if fdr_q is None else fdr_q
    gs, vs = [], []
    if include_full:
        g, v, _ = run_pcmciplus(data, names, tau_max=tau_max, pc_alpha=pc_alpha, constrained=constrained, fdr=fdr,
                                fdr_q=fdr_q, exogenous=exogenous)
        gs.append(g)
        vs.append(v)
    pc = _pcmci(data, names)
    pc.dataframe.bootstrap = {"boot_blocklength": block}
    for s in np.random.SeedSequence(seed).spawn(boot):
        r = pc.parallelized_bootstraps("run_pcmciplus", args, boot_seed=s)
        g = r["graph"]
        if fdr not in (None, "none"):
            g, _ = fdr_prune(g, r["p_matrix"], q_lev, mask)
        gs.append(encode(g))
        vs.append(r["val_matrix"].astype("float32"))
    return np.stack(gs), np.stack(vs)


def run_lpcmci(data, names, *, tau_max=7, pc_alpha=0.01, constrained=True, fdr_q: float | None = None):
    """LPCMCI single fit. With ``fdr_q``: BH over its max-p matrix applied to the graph (protocol §12 v1.4)."""
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.lpcmci import LPCMCI

    lp = LPCMCI(dataframe=_dataframe(data, names), cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    la = None
    if constrained:
        la = {}
        base = link_assumptions(names, tau_max)
        for j, d in base.items():
            la[j] = {k: ("-?>" if v == "-?>" else ("<?-" if v == "<?-" else "o?o")) for k, v in d.items()}
    r = lp.run_lpcmci(link_assumptions=la, tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
    g = r["graph"]
    if fdr_q is not None:
        g, _ = fdr_prune(g, r["p_matrix"], fdr_q, tested_mask(len(names), tau_max, None if la is None else la))
    return encode(g), r["val_matrix"].astype("float32")


def run_jpcmciplus(datasets: list[np.ndarray], names, *, tau_max=7, pc_alpha=0.01, constrained=True,
                   fdr_q: float | None = None):
    """J-PCMCI+ over several sites (datasets) with a space-dummy context node (Günther et al. 2023).

    The dummy is a one-hot dataset index; system variables are ``names``. Returns the system block of
    the graph.
    """
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr_mult import ParCorrMult
    from tigramite.jpcmciplus import JPCMCIplus

    M = len(datasets)
    N = len(names)
    data, masks = {}, {}
    for m, x in enumerate(datasets):
        T = x.shape[0]
        dummy = np.zeros((T, M - 1))
        if m > 0:
            dummy[:, m - 1] = 1.0
        data[m] = np.column_stack([np.where(np.isfinite(x), x, MISSING), dummy])
    node_cl = {j: "system" for j in range(N)}
    node_cl.update({N + k: "space_dummy" for k in range(M - 1)})
    df = pp.DataFrame(data, analysis_mode="multiple", missing_flag=MISSING,
                      var_names=list(names) + [f"dummy{k}" for k in range(M - 1)])
    jp = JPCMCIplus(dataframe=df, cond_ind_test=ParCorrMult(significance="analytic"), node_classification=node_cl,
                    verbosity=0)
    la = None
    if constrained:
        base = link_assumptions(names, tau_max)
        la = {j: dict(v) for j, v in base.items()}
        for k in range(M - 1):
            for j in range(N):
                la[j][(N + k, 0)] = "-?>"
            la[N + k] = {}
            for j in range(N):
                la[N + k][(j, 0)] = "<?-"
    r = jp.run_jpcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha, link_assumptions=la)
    g = r["graph"][:N, :N]
    if fdr_q is not None:   # FDR-BH over the system block (same family rule as PCMCI+, review W-a)
        base = link_assumptions(names, tau_max) if constrained else None
        g, _ = fdr_prune(g, r["p_matrix"][:N, :N], fdr_q, tested_mask(N, tau_max, base))
    return encode(g), r["val_matrix"][:N, :N].astype("float32")


def xcorr_baseline(data: np.ndarray, names, *, tau_max=7, alpha=0.01):
    """Lagged cross-correlation baseline: r(X_i(t−τ), X_j(t)), effective-n t-test (lag-1 autocorrelations),
    Benjamini–Hochberg over all (i, j, τ). Returns graph codes (1 for τ ≥ 1 significant, 3 for τ = 0) and r."""
    T, N = data.shape
    r1 = np.array([_lag1(data[:, k]) for k in range(N)])
    rs = np.zeros((N, N, tau_max + 1), "float32")
    ps = np.ones((N, N, tau_max + 1))
    for i in range(N):
        for j in range(N):
            for tau in range(tau_max + 1):
                if i == j and tau == 0:
                    continue
                a, b = data[: T - tau, i], data[tau:, j]
                ok = np.isfinite(a) & np.isfinite(b)
                n = ok.sum()
                if n < 30:
                    continue
                r = np.corrcoef(a[ok], b[ok])[0, 1]
                f = (1 - r1[i] * r1[j]) / (1 + r1[i] * r1[j])
                ne = max(5.0, n * f)
                t = r * np.sqrt((ne - 2) / max(1e-12, 1 - r * r))
                rs[i, j, tau] = r
                ps[i, j, tau] = 2 * stats.t.sf(abs(t), ne - 2)
    flat = ps.ravel()
    order = np.argsort(flat)
    m = flat.size
    thr = alpha * np.arange(1, m + 1) / m
    passed = np.zeros(m, bool)
    below = np.flatnonzero(flat[order] <= thr)
    if below.size:
        passed[order[: below.max() + 1]] = True
    sig = passed.reshape(ps.shape)
    g = np.zeros((N, N, tau_max + 1), np.uint8)
    g[..., 1:][sig[..., 1:]] = 1
    g[..., 0][sig[..., 0]] = 3
    return g, rs


def _lag1(x):
    a, b = x[:-1], x[1:]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else 0.0
