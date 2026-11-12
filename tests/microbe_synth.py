# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SYNTHETIC community simulator for the unit tests of M3 architecture round v2 (declared; RESEARCH_PLAN R9).

Nothing here touches real data. Units have Gaussian niches on two planted gradients; read counts are multinomial.
Used only by ``tests/test_microbe_niche.py`` and ``tests/test_microbe_archv2.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.data.microbe_geo import GEO_TARGETS
from sentinel.models.microbe import dna
from sentinel.models.microbe.design import TARGET_NAMES, Design

MASTER = tuple(TARGET_NAMES) + tuple(GEO_TARGETS) + ("field_temp_c",)
CORE = "GCAGCCGCGGTAA"                       # matches dna.V4_CORE_515F
PLANTED = {"log_sc": (2.0, 0.5, 0.0), "ph": (7.0, 0.0, 0.4), "log_tn": (-0.5, 0.3, -0.2),
           "log_ca": (1.0, 0.45, 0.1)}        # target -> (mean, loading on z1, loading on z2)


def _rand_seq(rng, n: int) -> str:
    return "".join(rng.choice(list("ACGT"), n))


def simulate(Z: np.ndarray, mu: np.ndarray, a: np.ndarray, tol: float, depth: int, rng) -> np.ndarray:
    """SYNTHETIC multinomial read counts [events x units] from Gaussian niches."""
    q = a[None, :] - 0.5 * (((Z[:, None, :] - mu[None, :, :]) / tol) ** 2).sum(2)
    p = np.exp(q - q.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return np.vstack([rng.multinomial(depth, pi) for pi in p]).astype(float)


def make_design(seed: int = 0, n_domains: int = 6, sites_per_domain: int = 2, per_site: int = 10, n_units: int = 150,
                depth: int = 4000, tol: float = 0.7):
    """SYNTHETIC design with the 17 master dimensions.

    Events of even sites are 'V4V5' (the read starts with the 230-bp window), events of odd sites are 'V3V4' (the
    window follows the 515F core at position 150), so the same window reaches both as different sequences. Unit 0
    additionally has a second V4V5 sequence with the same window, and every event carries one read set from a
    sequence too short for a window. Returns (design, truth) with truth = {'z', 'mu', 'counts', 'windows'}."""
    rng = np.random.default_rng(seed)
    n_sites = n_domains * sites_per_domain
    windows = [_rand_seq(rng, dna.V4_LEN) for _ in range(n_units)]
    seq_v45 = [w + _rand_seq(rng, 150) for w in windows]
    seq_v34 = [_rand_seq(rng, 150) + CORE + w + _rand_seq(rng, 20) for w in windows]
    twin0 = windows[0] + _rand_seq(rng, 140)                 # same window as unit 0, different sequence
    short = _rand_seq(rng, 100)                              # no V4 window
    mu = rng.uniform(-2, 2, (n_units, 2))
    a = rng.normal(0, 1, n_units)
    site_z = rng.normal(0, 1, (n_sites, 2))
    rows, toks, Zs = [], [], []
    genus = [f"g{k // 5:02d}" for k in range(n_units)]
    family = [f"f{k // 15:02d}" for k in range(n_units)]
    order = [f"o{k // 50:02d}" for k in range(n_units)]
    for si in range(n_sites):
        primer = "V4V5" if si % 2 == 0 else "V3V4"
        for j in range(per_site):
            z = 0.8 * site_z[si] + rng.normal(0, 0.6, 2)
            Zs.append(z)
            smp = f"E{si:02d}_{j:02d}"
            c = simulate(z[None, :], mu, a, tol, depth, rng)[0]
            row = {"sample_id": smp, "site_id": f"SITE{si:02d}", "domain_id": f"D{si // sites_per_domain:02d}",
                   "aquatic_type": ["stream", "lake", "river"][si % 3], "latitude": 30.0 + si,
                   "longitude": -100.0 + 2 * si, "elevation": np.nan if si == 0 else 100.0 * si,
                   "collect_date": pd.Timestamp("2019-01-15") + pd.Timedelta(days=33 * j + si),
                   "primer_set": primer, "source_product": "DP1.20141.002"}
            for t in MASTER:
                m, b1, b2 = PLANTED.get(t, (0.0, 0.0, 0.0))
                row[t] = m + b1 * z[0] + b2 * z[1] + rng.normal(0, 0.05 if t in PLANTED else 1.0)
            if j == 0:
                row["log_chla"] = np.nan                    # a partly missing dimension
            if si % 3 == 0:
                row["log_uv254"] = np.nan
            n_tok = 0
            for k in np.flatnonzero(c > 0):
                seq = seq_v45[k] if primer == "V4V5" else seq_v34[k]
                cnt = int(c[k])
                if k == 0 and primer == "V4V5" and cnt >= 2:
                    toks.append({"sample_id": smp, "seq_id": dna.seq_key(twin0), "count": cnt // 2,
                                 "genus": genus[0], "family": family[0], "order": order[0]})
                    cnt -= cnt // 2
                    n_tok += 1
                toks.append({"sample_id": smp, "seq_id": dna.seq_key(seq), "count": cnt,
                             "genus": None if k % 11 == 0 else genus[k], "family": family[k], "order": order[k]})
                n_tok += 1
            toks.append({"sample_id": smp, "seq_id": dna.seq_key(short), "count": 37, "genus": None,
                         "family": None, "order": "o_short"})
            row["n_reads"], row["n_seqs"] = int(c.sum()) + 37, n_tok + 1
            rows.append(row)
    S = pd.DataFrame(rows).set_index("sample_id", drop=False)
    T = pd.DataFrame(toks)
    T["count"] = T["count"].astype("Int64")
    allseq = seq_v45 + seq_v34 + [twin0, short]
    Q = pd.DataFrame({"seq_id": [dna.seq_key(s) for s in allseq], "sequence": allseq,
                      "length": [len(s) for s in allseq]})
    d = Design(S, T, Q, {"kind": "date_from", "date": "2030-01-01"}, False)
    return d, {"z": np.array(Zs), "mu": mu, "a": a, "windows": windows}


def lodo_folds(samples: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    dom = samples["domain_id"].to_numpy()
    ids = samples.index.to_numpy()
    return [(ids[dom != d], ids[dom == d]) for d in sorted(set(dom))]
