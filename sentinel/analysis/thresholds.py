# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Conductivity thresholds for macroinvertebrates (protocol §8; RESEARCH_PLAN §7.3).

* :func:`xc95` / :func:`hc05` — extirpation-concentration method as implemented in the EPA 2011
  field-based conductivity benchmark (Cormier & Suter 2013): per-taxon XC95 from the occurrence
  distribution over log10 SC weighted 1/n per bin (60 equal-width bins), HC05 at P = R/(N + 1) = 0.05.
* :func:`titan` — TITAN-style change points (Baker & King 2010; TITAN2 defaults minSplt 5, 250
  permutations, 500 bootstraps, purity/reliability 0.95): Dufrêne–Legendre IndVal at every candidate split
  of the gradient, z-scores against permutations, per-taxon change point = argmax z, community change
  points = argmax of the sum of z− (decreasers) / z+ (increasers) over pure and reliable taxa.
* :func:`richness_baseline` — SC where LOWESS-smoothed genus richness falls 5% below its low-SC level.

Data loaders: :func:`nrsa_samples` (EPA NRSA, genus or family) and :func:`uk_samples` (EA England, family).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import paths

NRSA = paths.PROCESSED / "nars_streams"
UKEA = paths.PROCESSED / "uk_ea_ecology"
TAXMAP = paths.PROCESSED / "taxonomy_harmonised" / "taxon_map.parquet"
WADEABLE = ("BERW", "BERWW")
BOATABLE = ("BETB", "BELGB")
APPALACHIA_L3 = ("69", "70")
SITE_RADIUS_KM = 1.0
# Pre-declared testability rule for a stratum's HC05 (protocol §8; review A5, logged as outcome-informed §12)
MIN_SAMPLES, MIN_UNITS, MIN_TAXA_HC05, MIN_REF_SAMPLES = 300, 150, 19, 10


def site_clusters(visits: pd.DataFrame, radius_km: float = SITE_RADIUS_KM) -> pd.Series:
    """Physical-site cluster per visit (review A4): NRSA ``unique_id`` exists only for 2018-19 and 2023-24,
    so visits are linked by coordinates ≤ ``radius_km`` (connected components, the M5 BioAssess rule) and
    by a shared ``unique_id``. Returns labels 'C00000'… indexed like ``visits``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from ..models.bioassess.data import location_clusters

    v = visits.reset_index(drop=True)
    lab = np.asarray(location_clusters(v["lat"].to_numpy(float), v["lon"].to_numpy(float), radius_km))
    rows, cols = list(range(len(v))), list(lab)
    uid = v["unique_id"].astype("string")
    n_loc = int(lab.max()) + 1 if len(lab) else 0
    codes = {u: n_loc + k for k, u in enumerate(sorted(uid.dropna().unique()))}
    for r, u in enumerate(uid):
        if u is not pd.NA and u in codes:
            rows.append(r)
            cols.append(codes[u])
    n = n_loc + len(codes)
    # bipartite graph visit ↔ (location label | unique_id) → components over visits
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(v), n)).tocsr()
    G = (A @ A.T).tocoo()
    _, comp = connected_components(G, directed=False)
    return pd.Series([f"C{c:05d}" for c in comp], index=visits.index)


# ----------------------------------------------------------------------------- data

def nrsa_samples(level: str = "genus", protocol: str = "wadeable", min_ph: float = 6.0):
    """Return (meta DataFrame per visit, abundance matrix DataFrame visit × taxon (counts))."""
    v = pd.read_parquet(NRSA / "visits.parquet")
    v = v.assign(site_cluster=site_clusters(v))       # clusters built on ALL visits (links across cycles)
    v = v[v["visit_no"].astype(str) == "1"]
    chem = pd.read_parquet(NRSA / "chemistry_wide.parquet", columns=["visit_id", "cond_us_cm", "ph_lab_su"])
    v = v.merge(chem, on="visit_id", how="left")
    v = v[(v["cond_us_cm"] > 0) & (v["ph_lab_su"] >= min_ph)]
    b = pd.read_parquet(NRSA / "benthic_counts.parquet", columns=["visit_id", "survey", "sample_type", "taxa_id", "count"])
    b = b[b["sample_type"].isin(WADEABLE if protocol == "wadeable" else BOATABLE) & (b["count"] > 0)]
    tm = pd.read_parquet(TAXMAP, columns=["dataset", "source_taxon_id", f"h_{level}"])
    tm = tm[tm["dataset"].str.startswith("nrsa")].drop_duplicates(["dataset", "source_taxon_id"])
    b = b.assign(taxa_id=b["taxa_id"].astype(str)).merge(
        tm.rename(columns={"dataset": "survey", "source_taxon_id": "taxa_id", f"h_{level}": "taxon"}),
        on=["survey", "taxa_id"], how="left")
    b = b[b["taxon"].notna() & ~b["taxon"].astype(str).str.contains(r"\(|UNDETERMINED", regex=True)]
    b = b[b["visit_id"].isin(v["visit_id"])]
    ab = b.pivot_table(index="visit_id", columns="taxon", values="count", aggfunc="sum", fill_value=0.0)
    meta = v.set_index("visit_id").loc[ab.index]
    meta = meta.assign(sc=meta["cond_us_cm"].astype(float), unit=meta["site_cluster"].astype(str))
    return meta, ab


def uk_samples(level: str = "family", max_pair_m: float = 250.0, min_year: int = 2000):
    """EA invertebrate samples with ±30-day conductivity@25 °C (median) at a paired WQ point ≤ 250 m;
    one sample per site (the latest). Abundance = log abundance class (cross-method comparable)."""
    ch = pd.read_parquet(UKEA / "chemistry_matched_wide_pm30d_invertebrate.parquet",
                         columns=["sample_id", "site_id", "pair_distance_m",
                                  "cond25__microsiemenspercentimetre__pm30d__median", "ph__phunits__pm30d__median"])
    ch = ch.rename(columns={"cond25__microsiemenspercentimetre__pm30d__median": "sc",
                            "ph__phunits__pm30d__median": "ph"})
    ch = ch[(ch["pair_distance_m"] <= max_pair_m) & (ch["sc"] > 0)]
    ch = ch[~(ch["ph"] < 6.0)]
    s = pd.read_parquet(UKEA / "samples_invertebrate.parquet", columns=["sample_id", "analysis_id", "sample_date",
                                                                         "analysis_type", "waterbody_type"])
    s = s[s["sample_id"].isin(ch["sample_id"])]
    s["sample_date"] = pd.to_datetime(s["sample_date"])
    s = s[s["sample_date"].dt.year >= min_year]
    m = s.merge(ch, on="sample_id").sort_values("sample_date").groupby("site_id").tail(1)
    c = pd.read_parquet(UKEA / "counts_invertebrate.parquet", columns=["analysis_id", "taxon_list_item_key",
                                                                        "log_abundance_class"])
    c = c[c["analysis_id"].isin(m["analysis_id"])]
    tm = pd.read_parquet(TAXMAP, columns=["dataset", "source_taxon_id", f"h_{level}", "group"])
    tm = tm[tm["dataset"] == "ea_invertebrate"].drop_duplicates("source_taxon_id")
    c = c.merge(tm.rename(columns={"source_taxon_id": "taxon_list_item_key", f"h_{level}": "taxon"}),
                on="taxon_list_item_key", how="left")
    c = c[c["taxon"].notna() & ~c["taxon"].astype(str).str.contains(r"\(|UNDETERMINED", regex=True)]
    c = c[c["group"].astype(str).str.lower().str.contains("invert") | c["group"].isna()]
    c = c.assign(log_abundance_class=pd.to_numeric(c["log_abundance_class"], errors="coerce").astype("float64").fillna(1.0))
    ab = c.pivot_table(index="analysis_id", columns="taxon", values="log_abundance_class", aggfunc="max",
                       fill_value=0.0)
    meta = m.set_index("analysis_id").loc[ab.index]
    meta = meta.assign(unit=meta["site_id"].astype(str))
    return meta, ab


# ----------------------------------------------------------------------------- XC95 / HC05

def xc95(sc: np.ndarray, present: np.ndarray, n_bins: int = 60, q: float = 0.95) -> float:
    """95th centile of the occurrence distribution over log10 SC, occurrences weighted 1/n per bin."""
    x = np.log10(sc)
    edges = np.linspace(x.min(), x.max() + 1e-9, n_bins + 1)
    b = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    n_bin = np.bincount(b, minlength=n_bins).astype(float)
    k_bin = np.bincount(b, weights=present.astype(float), minlength=n_bins)
    p = np.where(n_bin > 0, k_bin / np.maximum(n_bin, 1), 0.0)
    if p.sum() <= 0:
        return np.nan
    cum = np.concatenate([[0.0], np.cumsum(p) / p.sum()])
    return float(10 ** np.interp(q, cum, edges))


def hc05(xcs: np.ndarray, p: float = 0.05) -> float:
    """Interpolate the sorted taxon XC95s at P = R/(N + 1) = p."""
    x = np.sort(np.asarray(xcs, float)[np.isfinite(xcs)])
    if x.size < 2:
        return np.nan
    P = np.arange(1, x.size + 1) / (x.size + 1)
    return float(10 ** np.interp(p, P, np.log10(x)))


def extirpation(meta: pd.DataFrame, ab: pd.DataFrame, *, min_occ: int = 25, ref_mask: np.ndarray | None = None,
                n_boot: int = 1000, seed: int = 0) -> dict:
    """HC05 with a site-bootstrap CI. ``ref_mask`` (taxa) restricts to taxa seen at reference sites."""
    sc = meta["sc"].to_numpy(float)
    pres = (ab.to_numpy() > 0)
    occ = pres.sum(axis=0)
    taxa_ok = occ >= min_occ
    if ref_mask is not None:
        taxa_ok &= ref_mask
    xcs = np.array([xc95(sc, pres[:, k]) for k in np.flatnonzero(taxa_ok)])
    point = hc05(xcs)
    units = meta["unit"].to_numpy()
    u, inv = np.unique(units, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(u))]
    rs = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rs.integers(0, len(u), len(u))])
        pr = pres[idx]
        ok = pr.sum(axis=0) >= min_occ
        if ref_mask is not None:
            ok &= ref_mask
        bs.append(hc05(np.array([xc95(sc[idx], pr[:, k]) for k in np.flatnonzero(ok)])))
    bs = np.array(bs)
    return {"hc05": point, "ci_low": float(np.nanquantile(bs, 0.025)), "ci_high": float(np.nanquantile(bs, 0.975)),
            "n_taxa": int(taxa_ok.sum()), "n_samples": int(len(sc)), "n_units": int(len(u)),
            "hc05_clamped_at_lowest_rank": bool(taxa_ok.sum() < MIN_TAXA_HC05), "bootstrap_unit": "site cluster",
            "xc95": dict(zip(ab.columns[taxa_ok], xcs.tolist()))}


# ----------------------------------------------------------------------------- TITAN-style

def _indval_all_cuts(a: np.ndarray, cut_idx: np.ndarray):
    """a: (R, N) abundances sorted by the gradient (R replicates/permutations). Returns IndVal (R, C) of
    the best group and the group sign (−1 = low side, +1 = high side) for splits after positions cut_idx."""
    N = a.shape[1]
    pres = (a > 0).astype(float)
    ca, cp = np.cumsum(a, axis=1), np.cumsum(pres, axis=1)
    totA, totP = ca[:, -1:], cp[:, -1:]
    nL = cut_idx + 1
    nR = N - nL
    aL, pL = ca[:, cut_idx], cp[:, cut_idx]
    aR, pR = totA - aL, totP - pL
    mL, mR = aL / nL, aR / nR
    s = mL + mR
    s = np.where(s > 0, s, np.nan)
    ivL = 100 * (mL / s) * (pL / nL)
    ivR = 100 * (mR / s) * (pR / nR)
    iv = np.fmax(ivL, ivR)
    sign = np.where(ivL >= ivR, -1, 1)
    return np.nan_to_num(iv), sign


def _taxon_z(a_sorted: np.ndarray, cut_idx: np.ndarray, n_perm: int, rs) -> tuple:
    obs, sign = _indval_all_cuts(a_sorted[None], cut_idx)
    perm = np.stack([rs.permutation(a_sorted) for _ in range(n_perm)])
    piv, _ = _indval_all_cuts(perm, cut_idx)
    mu, sd = piv.mean(axis=0), piv.std(axis=0)
    z = (obs[0] - mu) / np.where(sd > 0, sd, np.nan)
    k = int(np.nanargmax(z)) if np.isfinite(z).any() else 0
    p = (np.sum(piv[:, k] >= obs[0, k]) + 1) / (n_perm + 1)
    return z, sign[0], k, p


def titan(meta: pd.DataFrame, ab: pd.DataFrame, *, min_occ: int = 5, min_split: int = 5, n_perm: int = 250,
          n_boot: int = 500, n_cuts: int = 100, purity: float = 0.95, reliability: float = 0.95,
          p_rel: float = 0.05, seed: int = 0) -> dict:
    """TITAN-style analysis on log10 SC. Candidate splits: ``n_cuts`` gradient quantiles leaving ≥
    ``min_split`` samples per side. Abundances log10(x + 1)."""
    rs = np.random.default_rng(seed)
    x = np.log10(meta["sc"].to_numpy(float))
    A = np.log10(ab.to_numpy(float) + 1.0)
    keep = (A > 0).sum(axis=0) >= min_occ
    A, taxa = A[:, keep], ab.columns[keep]
    order = np.argsort(x, kind="stable")
    xs, As = x[order], A[order]
    N = len(xs)
    cut_idx = np.unique(np.linspace(min_split - 1, N - min_split - 1, n_cuts).round().astype(int))
    cut_val = 0.5 * (xs[cut_idx] + xs[np.minimum(cut_idx + 1, N - 1)])

    def run(As_, xs_, rs_):
        Z, S, K, P = [], [], [], []
        for t in range(As_.shape[1]):
            z, sg, k, p = _taxon_z(As_[:, t], cut_idx, n_perm, rs_)
            Z.append(z)
            S.append(sg[k])
            K.append(k)
            P.append(p)
        return np.array(Z), np.array(S), np.array(K), np.array(P)

    def run_boot(As_, xs_, rs_):
        """Bootstrap replicate: its own candidate splits (sample size changes with cluster resampling);
        change points returned as indices into the ORIGINAL cut grid (nearest cut value)."""
        Nb = len(xs_)
        ci = np.unique(np.linspace(min_split - 1, Nb - min_split - 1, n_cuts).round().astype(int))
        cv = 0.5 * (xs_[ci] + xs_[np.minimum(ci + 1, Nb - 1)])
        Sb, Kb, Pb = [], [], []
        for t in range(As_.shape[1]):
            if (As_[:, t] > 0).sum() == 0:      # taxon absent from the replicate: no change point
                Sb.append(0)
                Kb.append(np.nan)
                Pb.append(1.0)
                continue
            z, sg, k, p = _taxon_z(As_[:, t], ci, n_perm, rs_)
            Sb.append(sg[k])
            Kb.append(int(np.argmin(np.abs(cut_val - cv[k]))))
            Pb.append(p)
        return None, np.array(Sb), np.array(Kb), np.array(Pb)

    Z, S, K, P = run(As, xs, rs)
    # bootstrap: resample SITE CLUSTERS with replacement (review A4; TITAN2 resamples samples — our samples
    # are revisits of the same sites, so the independent unit is the site), re-sort along the gradient
    units = meta["unit"].to_numpy()
    _, uinv = np.unique(units, return_inverse=True)
    members = [np.flatnonzero(uinv == k) for k in range(uinv.max() + 1)]
    bS, bK, bP = [], [], []
    for b in range(n_boot):
        idx = np.concatenate([members[k] for k in rs.integers(0, len(members), len(members))])
        o = np.argsort(x[idx], kind="stable")
        Zb, Sb, Kb, Pb = run_boot(A[idx][o], x[idx][o], rs)
        bS.append(Sb)
        bK.append(Kb)
        bP.append(Pb)
    bS, bK, bP = np.array(bS), np.array(bK), np.array(bP)
    pur = (bS == S[None, :]).mean(axis=0)
    rel = (bP <= p_rel).mean(axis=0)
    good = (pur >= purity) & (rel >= reliability)
    zpos = np.where(Z > 0, Z, 0)
    dec = good & (S == -1)
    inc = good & (S == 1)
    sum_minus = (zpos * dec[:, None]).sum(axis=0)
    sum_plus = (zpos * inc[:, None]).sum(axis=0)
    taxa_tab = pd.DataFrame({"taxon": taxa, "cp_sc": 10 ** cut_val[K], "z": Z[np.arange(len(K)), K], "direction":
                             np.where(S == -1, "z-", "z+"), "purity": pur, "reliability": rel, "p": P,
                             "pure_reliable": good,
                             "boot_cp_q05": _cp_q(bK, 0.05, cut_val), "boot_cp_q95": _cp_q(bK, 0.95, cut_val)})
    return {"taxa": taxa_tab, "cut_sc": 10 ** cut_val, "sum_z_minus": sum_minus, "sum_z_plus": sum_plus,
            "community_cp_minus": float(10 ** cut_val[int(np.argmax(sum_minus))]) if dec.any() else np.nan,
            "community_cp_plus": float(10 ** cut_val[int(np.argmax(sum_plus))]) if inc.any() else np.nan,
            "n_decreasers": int(dec.sum()), "n_increasers": int(inc.sum()), "n_taxa_tested": int(len(taxa)),
            "n_samples": int(N), "n_units": int(len(members)), "bootstrap_unit": "site cluster"}


def _cp_q(bK, q, cut_val):
    """Per-taxon bootstrap quantile of the change-point index (replicates without the taxon ignored)."""
    v = np.nanquantile(np.asarray(bK, float), q, axis=0)
    return np.where(np.isfinite(v), 10 ** cut_val[np.nan_to_num(v).astype(int)], np.nan)


def community_cp_ci(meta, ab, *, n_boot: int = 200, seed: int = 0, **kw) -> dict:
    """Bootstrap CI of the community sum(z−) change point: re-run the observed-data part of TITAN on
    site-resampled data (pure/reliable sets fixed from the full analysis are NOT reused — each replicate
    uses taxa with z > 0 and the observed direction, a documented simplification for compute)."""
    rs = np.random.default_rng(seed)
    x = np.log10(meta["sc"].to_numpy(float))
    A = np.log10(ab.to_numpy(float) + 1.0)
    keep = (A > 0).sum(axis=0) >= kw.get("min_occ", 5)
    A = A[:, keep]
    _, uinv = np.unique(meta["unit"].to_numpy(), return_inverse=True)
    members = [np.flatnonzero(uinv == k) for k in range(uinv.max() + 1)]
    cps = []
    n_cuts, min_split = kw.get("n_cuts", 100), kw.get("min_split", 5)
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rs.integers(0, len(members), len(members))])
        N = len(idx)
        o = np.argsort(x[idx], kind="stable")
        xs, As = x[idx][o], A[idx][o]
        cut_idx = np.unique(np.linspace(min_split - 1, N - min_split - 1, n_cuts).round().astype(int))
        cut_val = 0.5 * (xs[cut_idx] + xs[np.minimum(cut_idx + 1, N - 1)])
        tot = np.zeros(len(cut_idx))
        for t in range(As.shape[1]):
            if (As[:, t] > 0).sum() == 0:
                continue
            z, sg, k, p = _taxon_z(As[:, t], cut_idx, 50, rs)
            if sg[k] == -1 and p <= 0.05:
                tot += np.where(np.isfinite(z) & (z > 0), z, 0)
        cps.append(10 ** cut_val[int(np.argmax(tot))])
    return {"ci_low": float(np.quantile(cps, 0.025)), "ci_high": float(np.quantile(cps, 0.975)), "n_boot": n_boot,
            "bootstrap_unit": "site cluster"}


def richness_baseline(meta, ab, drop: float = 0.05, frac: float = 0.3) -> float:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    x = np.log10(meta["sc"].to_numpy(float))
    r = (ab.to_numpy() > 0).sum(axis=1).astype(float)
    fit = lowess(r, x, frac=frac, return_sorted=True)
    ref = np.mean(r[x <= np.quantile(x, 0.25)])
    below = np.flatnonzero((fit[:, 1] <= (1 - drop) * ref) & (fit[:, 0] > np.quantile(x, 0.25)))
    return float(10 ** fit[below[0], 0]) if below.size else np.nan
