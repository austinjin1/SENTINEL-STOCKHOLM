# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-X development machinery (protocol ``bioassess_x_protocol.md`` §6–§7 and amendment v4).

* :func:`load_context` — the frozen split table (sha-checked), the frozen-test keys of every workstream and the
  unified M5-X design (``xdata.load_x_design``; only allowed rows, protected labels never read), cached.
* :func:`dev_folds` — development folds (amendment v4.1): pseudo-test NLA 2017 (X-L, LOEO) / UK 2016–20
  ``pool_fit`` (X-U, leave-one-area-out); training = the fold's *test-design* pool minus cycle-/time-out rows minus
  the 1-km / cluster buffer. Every id list passes ``xtrack.assert_fit_ids``.
* :func:`test_folds` — the same structure for the test design (used only by the one-time scorer).
* Labels (:func:`labels`): same-continent or multi-source (harmonised shared targets, amendment v4.2).
* :class:`XFeaturizer` — the M5 ``CommunityFeaturizer`` semantics on a sparse sample × taxon matrix (fit on the
  fold's labelled training rows), plus source / ``analysis_method`` one-hots and geography / context columns.
* :func:`fit_predict` — every tabular comparator of §6 with M5's fixed hyperparameters and the §6 row budgets.

No function here reads a test-block label: the design holds only allowed rows (``xdata``).
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.models.bioassess import xtrack as X
from sentinel.models.bioassess.data import TARGET_NAMES

XDIR = paths.RUNS_ROOT / "bioassess" / "x_track"
DEV_DIR = XDIR / "dev"
SPLIT_SHA256 = "18a35ef3ba9ddd87a5be2e8c4ff1a18d29bea8869318f940b150b1306feee459"
BLOCKS = ("X-L", "X-U")
TARGETS = {"X-L": tuple(TARGET_NAMES), "X-U": tuple(X.UK_PRIMARY)}
CONTINENT = {"X-L": ("nla", "nrsa"), "X-U": ("uk",)}
# harmonised shared targets (amendment v4.2): X-L name <-> X-U name
SHARED = {"ph_lab": "uk_ph", "log10_cond_us_cm": "uk_log10_cond25_us_cm",
          "log10_chloride_mg_l": "uk_log10_chloride_mg_l"}
SHARED_INV = {v: k for k, v in SHARED.items()}
# union of both blocks' targets with the shared ones merged (BioSet multi-task output; amendment v4.9)
UNION_COLUMNS = ("log10_tp_ug_l", "log10_tn_mg_l", "log10_cond_us_cm", "log10_chloride_mg_l",
                 "log10_turbidity_ntu", "ph_lab", "uk_log10_orthop_mg_l", "uk_log10_nh4n_mg_l",
                 "uk_log10_nitrate_mg_l")
DEV_PSEUDO = {"X-L": "NLA 2017 first visits (LOEO x cycle-out)",
              "X-U": "UK pool_fit 2016-2020 (leave-one-area-out x time-out)"}
METHODS = ("ANLA", "ANLE", "ANAA")
GEO = ("lat", "lon", "elevation_m")
CONTEXT = {"X-L": ("lat", "lon", "elevation_m", "doy_sin", "doy_cos"),
           "X-U": ("lat", "lon", "elevation_m", "site_slope_m_per_km", "site_dist_from_source_km", "site_width_m",
                   "site_depth_cm", "site_discharge_category", "site_boulders_cobbles_pct", "site_pebbles_gravel_pct",
                   "site_sand_pct", "site_silt_clay_pct", "doy_sin", "doy_cos")}
INDICES = ("idx_whpt_aspt", "idx_whpt_n_taxa", "idx_psi_family_score", "idx_epsi_family_score",
           "idx_life_family_index", "idx_bmwp_aspt")
RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 3, "max_features": 0.3}
GBM_PARAMS = {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20,
              "colsample_bytree": 0.3, "subsample": 0.8, "subsample_freq": 1, "verbose": -1}
TABPFN_PARAMS = {"n_estimators": 8, "max_taxa": 300, "max_train_rows": 6000, "predict_chunk": 1000}
RF_MAX_ROWS, TABPFN_MAX_ROWS = 60_000, 6_000


# ------------------------------------------------------------------------------------------------- context
def _sha_text(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def load_split() -> pd.DataFrame:
    from sentinel.core.registry import file_sha256

    f = XDIR / "splits_x_v1.parquet"
    if file_sha256(f) != SPLIT_SHA256:
        raise RuntimeError("frozen M5-X split table changed")
    return pd.read_parquet(f)


def frozen_keys_cached(rebuild: bool = False) -> dict:
    """``xtrack.frozen_test_keys()`` cached under a key of the code + frozen split files that define it."""
    from sentinel.core.registry import file_sha256

    deps = [paths.REPO_ROOT / "sentinel/models/bioassess/xtrack.py"] + \
        [paths.REPO_ROOT / v for v in X.FROZEN_SPLIT_FILES.values()]
    key = _sha_text(SPLIT_SHA256, *[file_sha256(f) for f in deps])
    f = DEV_DIR / "frozen_keys.json"
    if f.exists() and not rebuild:
        d = json.loads(f.read_text())
        if d.get("key") == key:
            return {(s, i): r for s, i, r in d["rows"]}
    fr = X.frozen_test_keys()
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"key": key, "rows": [[s, i, r] for (s, i), r in fr.items()]}))
    return fr


def load_context(rebuild: bool = False):
    """(S, frozen, D): split table, frozen-test keys, unified design of allowed rows (cached pickle)."""
    from sentinel.core.registry import file_sha256
    from sentinel.models.bioassess import xdata

    S = load_split()
    frozen = frozen_keys_cached(rebuild)
    deps = [paths.REPO_ROOT / "sentinel/models/bioassess/xdata.py"]
    key = _sha_text(SPLIT_SHA256, len(frozen), *[file_sha256(f) for f in deps])
    f = DEV_DIR / "design_v1.pkl"
    D = None
    if f.exists() and not rebuild:
        with open(f, "rb") as fh:
            obj = pickle.load(fh)
        if obj.get("key") == key:
            D = obj["design"]
    if D is None:
        D = xdata.load_x_design(S, frozen)
        with open(f, "wb") as fh:
            pickle.dump({"key": key, "design": D}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    keys = [tuple(k.split(":", 1)) for k in D.samples.index]
    X.assert_fit_ids(keys, S, frozen)          # loader guard on every load (protocol §7)
    return S, frozen, D


def keys_of(ids) -> list[tuple[str, str]]:
    return [tuple(str(k).split(":", 1)) for k in ids]


# ------------------------------------------------------------------------------------------------- folds
@dataclass
class Fold:
    block: str
    g: str
    train: pd.Index
    test: pd.Index
    design: str = "dev"
    report: dict = field(default_factory=dict)


def _buffer_mask(smp: pd.DataFrame, te: pd.DataFrame, km: float = X.BUFFER_KM) -> np.ndarray:
    near = X._within_km(smp["lat"].to_numpy(float), smp["lon"].to_numpy(float), te["lat"].to_numpy(float),
                        te["lon"].to_numpy(float), km)
    return near | smp["location_id"].isin(set(te["location_id"])).to_numpy()


def pseudo_test_mask(S: pd.DataFrame, D, block: str) -> np.ndarray:
    s = D.samples
    lab = D.Y[list(TARGETS[block])].notna().any(axis=1).to_numpy()
    if block == "X-L":
        return ((s["source"] == "nla") & (s["survey"] == "nla17")).to_numpy() & lab & s["ag_eco9"].notna().to_numpy()
    role = S[S["source"] == "uk"].drop_duplicates("id").set_index("id")["uk_role"]
    r = pd.Series([i.split(":", 1)[1] for i in s.index], index=s.index).map(role)
    return ((s["source"] == "uk") & s["year"].between(2016, 2020) & (r == "pool_fit")).to_numpy() & lab & \
        s["ag_eco9"].notna().to_numpy()


def dev_folds(S: pd.DataFrame, frozen: dict, D, block: str) -> list[Fold]:
    """Development folds of amendment v4.1 (ids are design ids "<source>:<id>")."""
    s = D.samples
    pt = pseudo_test_mask(S, D, block)
    out = []
    for g in sorted(s.loc[pt, "ag_eco9"].astype(str).unique()):
        te_m = pt & (s["ag_eco9"].astype(str) == g).to_numpy()
        P = X.training_rows(S, block, g, frozen)
        pool = set(P["source"] + ":" + P["id"].astype(str))
        tr_m = s.index.isin(pool)
        if block == "X-L":
            tr_m &= ~((s["source"] == "nla") & (s["survey"] == "nla17")).to_numpy()
        else:
            tr_m &= ~((s["source"] == "uk") & (s["year"] >= 2016)).to_numpy()
        nb = int((tr_m & _buffer_mask(s, s[te_m])).sum())
        tr_m &= ~_buffer_mask(s, s[te_m])
        tr, te = s.index[tr_m], s.index[te_m]
        if set(tr) & set(te):
            raise AssertionError("dev train/test overlap")
        X.assert_fit_ids(keys_of(tr), S, frozen)
        X.assert_fit_ids(keys_of(te), S, frozen)
        out.append(Fold(block, g, tr, te, "dev", {"n_train": int(len(tr)), "n_test": int(len(te)),
                                                  "removed_buffer": nb}))
    return out


def save_folds(folds: dict[str, list[Fold]], f: Path) -> str:
    from sentinel.core.registry import file_sha256

    doc = {b: [{"g": fo.g, "train": list(map(str, fo.train)), "test": list(map(str, fo.test)), "report": fo.report}
               for fo in fl] for b, fl in folds.items()}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(doc))
    return file_sha256(f)


def read_folds(f: Path, S, frozen) -> dict[str, list[Fold]]:
    doc = json.loads(Path(f).read_text())
    out = {}
    for b, fl in doc.items():
        out[b] = []
        for d in fl:
            fo = Fold(b, d["g"], pd.Index(d["train"]), pd.Index(d["test"]), "dev", d.get("report", {}))
            X.assert_fit_ids(keys_of(fo.train), S, frozen)
            out[b].append(fo)
    return out


# ------------------------------------------------------------------------------------------------- labels
def labels(D, block: str, target: str, ids, variant: str = "same") -> np.ndarray:
    """Training label of ``target`` (a block target) for ``ids``: own-continent rows take their own column; with
    ``variant == "multi"`` rows of the other continent take the harmonised counterpart (shared targets only)."""
    ids = pd.Index(ids)
    y = D.Y.loc[ids, target].to_numpy(float).copy()
    own = D.samples.loc[ids, "source"].isin(CONTINENT[block]).to_numpy()
    y[~own] = np.nan
    other = SHARED.get(target) or SHARED_INV.get(target)
    if variant == "multi" and other is not None:
        y[~own] = D.Y.loc[ids, other].to_numpy(float)[~own]
    return y


def variants_for(target: str) -> tuple[str, ...]:
    return ("same", "multi") if (target in SHARED or target in SHARED_INV) else ("same",)


def stratified_subsample(groups: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    """Indices of a seeded proportional subsample stratified by ``groups`` (all rows if len <= n_max)."""
    n = len(groups)
    if n <= n_max:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    cnt = np.bincount(inv)
    alloc = np.floor(cnt * n_max / n).astype(int)
    rem = n_max - alloc.sum()
    frac = cnt * n_max / n - alloc
    alloc[np.argsort(-frac, kind="stable")[:rem]] += 1
    pick = [rng.choice(np.flatnonzero(inv == k), alloc[k], replace=False) for k in range(len(codes)) if alloc[k]]
    return np.sort(np.concatenate(pick))


# ------------------------------------------------------------------------------------------------- features
class TaxonMatrix:
    """Sparse sample × ``mixed_otu`` count matrix of the whole design (an index only; taxa are chosen per fold)."""

    def __init__(self, D, key: str = "mixed_otu"):
        from scipy import sparse

        t = D.tokens.dropna(subset=[key])
        self.rows = pd.Index(D.samples.index)
        self.cols = pd.Index(sorted(t[key].astype(str).unique()))
        r = self.rows.get_indexer(t["visit_id"])
        c = self.cols.get_indexer(t[key].astype(str))
        ok = r >= 0
        self.M = sparse.csr_matrix((t["count"].to_numpy(float)[ok], (r[ok], c[ok])),
                                   shape=(len(self.rows), len(self.cols)))
        self.M.sum_duplicates()
        self.total = np.asarray(self.M.sum(1)).ravel()

    def sub(self, ids):
        return self.M[self.rows.get_indexer(pd.Index(ids))]

    def prevalence(self, ids) -> pd.Series:
        A = self.sub(ids)
        return pd.Series(np.asarray((A > 0).sum(0)).ravel(), index=self.cols)

    def hellinger(self, ids, taxa) -> np.ndarray:
        r = self.rows.get_indexer(pd.Index(ids))
        A = self.M[r][:, self.cols.get_indexer(pd.Index(taxa))].toarray()
        tot = self.total[r]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(np.where(tot[:, None] > 0, A / tot[:, None], 0.0))

    def presence(self, ids, taxa) -> np.ndarray:
        r = self.rows.get_indexer(pd.Index(ids))
        return (self.M[r][:, self.cols.get_indexer(pd.Index(taxa))].toarray() > 0)


@dataclass
class XFeaturizer:
    """M5 ``CommunityFeaturizer`` semantics (taxa with prevalence >= ``min_samples`` in the fit rows, Hellinger of
    the full-count relative abundance) + optional descriptors, source / method one-hots, geography/context."""

    block: str
    min_samples: int = 10
    max_taxa: int | None = None
    community: bool = True
    descriptors: bool = True
    onehots: bool = True
    numeric: tuple = ()
    taxa_: list = field(default_factory=list)
    fill_: dict = field(default_factory=dict)

    def fit(self, TM: TaxonMatrix, D, ids) -> "XFeaturizer":
        if self.community:
            prev = TM.prevalence(ids)
            prev = prev[prev >= self.min_samples].sort_values(ascending=False, kind="stable")
            if self.max_taxa:
                prev = prev.iloc[: self.max_taxa]
            self.taxa_ = sorted(prev.index)
        s = D.samples.loc[pd.Index(ids)]
        self.fill_ = {c: float(pd.to_numeric(s[c], errors="coerce").median()) if c in s else 0.0 for c in self.numeric}
        self.fill_ = {c: (v if np.isfinite(v) else 0.0) for c, v in self.fill_.items()}
        return self

    def transform(self, TM: TaxonMatrix, D, ids) -> np.ndarray:
        ids = pd.Index(ids)
        s = D.samples.loc[ids]
        parts = []
        if self.community:
            parts.append(TM.hellinger(ids, self.taxa_))
        if self.descriptors:
            parts.append(s[["log10_total_count", "log10_n_taxa"]].to_numpy(float))
        if self.onehots:
            parts.append(np.column_stack([(s["source"] == k).to_numpy(float) for k in ("nla", "nrsa", "uk")]))
            am = s["analysis_method"].astype(object) if "analysis_method" in s else pd.Series(None, index=ids)
            uk = (s["source"] == "uk").to_numpy()
            oh = [(am == m).to_numpy(float) for m in METHODS]
            oh.append((uk & ~am.isin(METHODS).to_numpy()).astype(float))
            parts.append(np.column_stack(oh))
        if self.numeric:
            num = np.column_stack([pd.to_numeric(s[c], errors="coerce").to_numpy(float) if c in s
                                   else np.full(len(s), np.nan) for c in self.numeric])
            for j, c in enumerate(self.numeric):
                num[~np.isfinite(num[:, j]), j] = self.fill_[c]
            parts.append(num)
        return np.hstack(parts) if parts else np.zeros((len(ids), 0))


# ------------------------------------------------------------------------------------------------- models
COMMUNITY_MODELS = {   # name -> (featurizer kwargs builder, learner)
    "rf_community": ("rf", dict()),
    "gbm_community": ("gbm", dict()),
    "rf_community_geography": ("rf", dict(numeric="geo")),
    "rf_community_context": ("rf", dict(numeric="context")),
    "tabpfn_community": ("tabpfn", dict(max_taxa=TABPFN_PARAMS["max_taxa"])),
}
POOLED_MODELS = tuple(COMMUNITY_MODELS) + ("wapls2", "knn_community")
SIMPLE_MODELS = ("region_mean", "rf_geography", "nearest_reference")
INDEX_MODELS = ("idx_linear", "idx_linear_geo", "idx_rf", "idx_rf_geo")
REGISTRY_NAME = {
    "rf_community": "RandomForest[community]", "gbm_community": "LightGBM[community]",
    "rf_community_geography": "RandomForest[community+geography]",
    "rf_community_context": "RandomForest[community+context]", "tabpfn_community": "TabPFNv2[community]",
    "wapls2": "WA-PLS2[community]", "knn_community": "kNN-MAT[community]", "region_mean": "RegionMean",
    "rf_geography": "RandomForest[geography]", "nearest_reference": "NearestReferenceSite",
    "idx_linear": "UKIndices-Linear", "idx_linear_geo": "UKIndices-Linear+geography", "idx_rf": "UKIndices-RF",
    "idx_rf_geo": "UKIndices-RF+geography", "bioinfer": "bio.infer[EPA 1.3.6]", "gjam": "GJAM-inverse",
}
STOCHASTIC = {"rf_community", "gbm_community", "rf_community_geography", "rf_community_context", "tabpfn_community",
              "rf_geography", "idx_rf", "idx_rf_geo"}


def models_for(block: str) -> list[str]:
    m = list(SIMPLE_MODELS if block == "X-L" else SIMPLE_MODELS[:2]) + list(POOLED_MODELS)
    return m + (list(INDEX_MODELS) if block == "X-U" else [])


def _num_cols(block, kind):
    return {"geo": GEO, "context": CONTEXT[block]}.get(kind, ())


def _rf(seed, n_jobs):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**RF_PARAMS, n_jobs=n_jobs, random_state=seed)


def _fit_learner(kind, Xtr, y, Xte, seed, n_jobs, device="cuda"):
    if kind == "rf":
        return _rf(seed, n_jobs).fit(Xtr, y).predict(Xte)
    if kind == "gbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**GBM_PARAMS, random_state=seed, n_jobs=n_jobs).fit(Xtr, y).predict(Xte)
    if kind == "tabpfn":
        from sentinel.models.chemtox.tabpfn_model import load_tabpfn

        TabPFNRegressor, _ = load_tabpfn()
        m = TabPFNRegressor(n_estimators=TABPFN_PARAMS["n_estimators"], device=device, random_state=seed,
                            ignore_pretraining_limits=True)
        m.fit(Xtr.astype(np.float32), y)
        ch = TABPFN_PARAMS["predict_chunk"]
        Xq = Xte.astype(np.float32)
        return np.concatenate([m.predict(Xq[i:i + ch]) for i in range(0, len(Xq), ch)])
    if kind == "ols":
        from sklearn.linear_model import LinearRegression
        return LinearRegression().fit(Xtr, y).predict(Xte)
    raise ValueError(kind)


def fit_predict(name: str, block: str, target: str, D, TM: TaxonMatrix, train_ids, test_ids, *, variant: str = "same",
                seed: int = 0, n_jobs: int = 6, device: str = "cuda") -> np.ndarray:
    """Predictions of one comparator for one target (fit on ``train_ids`` only; labels per ``variant``)."""
    train_ids, test_ids = pd.Index(train_ids), pd.Index(test_ids)
    if set(train_ids) & set(test_ids):
        raise AssertionError("train and test ids overlap")
    y_all = labels(D, block, target, train_ids, variant)
    lab = train_ids[np.isfinite(y_all)]
    y = y_all[np.isfinite(y_all)]
    s_lab = D.samples.loc[lab]
    if len(lab) == 0:
        return np.full(len(test_ids), np.nan)
    if name == "region_mean":
        return np.full(len(test_ids), float(np.mean(y)))      # leave-region-out: region g unseen (v4.5)
    if name == "nearest_reference":
        from sklearn.neighbors import BallTree

        ref = (s_lab["reference_class"] == "R").to_numpy()
        tree = BallTree(np.radians(s_lab.loc[ref, ["lat", "lon"]].to_numpy(float)), metric="haversine")
        _, idx = tree.query(np.radians(D.samples.loc[test_ids, ["lat", "lon"]].to_numpy(float)), k=1)
        return y[ref][idx[:, 0]]
    groups = (s_lab["source"].astype(str) + "|" + s_lab["ag_eco9"].astype(str)).to_numpy()
    if name == "rf_geography":
        fz = XFeaturizer(block, community=False, descriptors=False, onehots=False, numeric=GEO).fit(TM, D, lab)
        pick = stratified_subsample(groups, RF_MAX_ROWS, seed)
        return _fit_learner("rf", fz.transform(TM, D, lab[pick]), y[pick], fz.transform(TM, D, test_ids), seed, n_jobs)
    if name in INDEX_MODELS:
        num = INDICES + (GEO if name.endswith("_geo") else ())
        fz = XFeaturizer(block, community=False, descriptors=False, onehots=True, numeric=num).fit(TM, D, lab)
        pick = stratified_subsample(groups, RF_MAX_ROWS, seed)
        kind = "ols" if name.startswith("idx_linear") else "rf"
        return _fit_learner(kind, fz.transform(TM, D, lab[pick]), y[pick], fz.transform(TM, D, test_ids), seed, n_jobs)
    if name in ("wapls2", "knn_community"):
        from sentinel.models.bioassess.baselines import wapls_fit_predict

        fz = XFeaturizer(block, descriptors=False, onehots=False).fit(TM, D, lab)
        Htr, Hte = fz.transform(TM, D, lab), fz.transform(TM, D, test_ids)
        if name == "wapls2":
            return wapls_fit_predict(Htr ** 2, y, Hte ** 2, 2)
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=n_jobs).fit(Htr, y).predict(Hte)
    if name in COMMUNITY_MODELS:
        kind, kw = COMMUNITY_MODELS[name]
        fz = XFeaturizer(block, max_taxa=kw.get("max_taxa"), numeric=_num_cols(block, kw.get("numeric"))).fit(
            TM, D, lab)
        cap = TABPFN_MAX_ROWS if kind == "tabpfn" else RF_MAX_ROWS
        pick = stratified_subsample(groups, cap, seed)
        return _fit_learner(kind, fz.transform(TM, D, lab[pick]), y[pick], fz.transform(TM, D, test_ids), seed,
                            n_jobs, device)
    raise ValueError(f"unknown comparator {name}")


# ------------------------------------------------------------------------------------------------- scoring
def score(y, p, region, cluster, n_boot: int = 2000, seed: int = 0) -> dict:
    from sentinel.models.bioassess import evaluate as EV

    sm = EV.summary(y, p, region)
    ci = EV.cis(y, p, region, cluster, n_boot=n_boot, seed=seed, metrics=("r2_anomaly", "r2"))
    sm.pop("r2_by_region", None)
    return {**sm, "ci": ci}


def hash_ids(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------- test design
def load_test_rows(S: pd.DataFrame, block: str):
    """The block's frozen TEST rows (tokens + labels) as a BioDesign. Called ONLY by the one-time scorer in its
    authorised test mode, after the fixed marker has been claimed (``scripts/eval/bioassess_x_final.py``)."""
    from sentinel.models.bioassess import xdata
    from sentinel.models.bioassess import xtestlook as XL
    from sentinel.models.bioassess.data import BioDesign

    XL.require_open_claim()                  # single-look guard: only under an open, unscored claim of this freeze
    T = S[(S["block"] == block) & (S["role"] == "test")]
    keys = set(zip(T["source"], T["id"].astype(str)))
    src, loader = {"X-L": ("nla", xdata._nla), "X-U": ("uk", xdata._uk)}[block]
    s, tok, y = loader(keys)
    cl = T.drop_duplicates(["source", "id"]).set_index(["source", "id"])["cluster"]
    key = src + ":" + s.index.astype(str)
    s = s.copy()
    s["location_id"] = cl.reindex(list(zip([src] * len(s), s.index.astype(str)))).to_numpy()
    tok = tok.assign(visit_id=src + ":" + tok["visit_id"].astype(str))
    s.index, y.index = key, key
    s.index.name = "visit_id"
    for c in set(TARGET_NAMES) | set(X.UK_PRIMARY) | set(SHARED.values()):
        if c not in y:
            y[c] = np.nan
    return BioDesign(s, tok[[c for c in xdata.TOKEN_COLS if c in tok.columns]], y.reindex(s.index),
                     {"n_test_rows_loaded": int(len(s)), "n_test_rows_split": int(len(T))})


def combine(D, Dt):
    from sentinel.models.bioassess.data import BioDesign

    if set(D.samples.index) & set(Dt.samples.index):
        raise AssertionError("test rows already in the training design")
    smp = pd.concat([D.samples, Dt.samples], sort=False)
    Y = pd.concat([D.Y, Dt.Y.reindex(columns=D.Y.columns)], sort=False).reindex(smp.index)
    return BioDesign(smp, pd.concat([D.tokens, Dt.tokens], ignore_index=True), Y, {**D.info, **Dt.info})


def test_folds(S: pd.DataFrame, frozen: dict, D, Dt, block: str) -> list[Fold]:
    """Test design: fold g = test rows of region g; training = ``training_rows(S, block, g, frozen)`` (allowed
    rows of the training design only)."""
    T = S[(S["block"] == block) & (S["role"] == "test")]
    tfold = dict(zip(T["source"] + ":" + T["id"].astype(str), T["fold"].astype(str)))
    te_all = pd.Series([tfold.get(i) for i in Dt.samples.index], index=Dt.samples.index)
    out = []
    for g in sorted(T["fold"].astype(str).unique()):
        P = X.training_rows(S, block, g, frozen)
        pool = set(P["source"] + ":" + P["id"].astype(str))
        tr = D.samples.index[D.samples.index.isin(pool)]
        te = te_all.index[(te_all == g).to_numpy()]
        X.assert_fit_ids(keys_of(tr), S, frozen)
        out.append(Fold(block, g, tr, te, "test", {"n_train": int(len(tr)), "n_test": int(len(te))}))
    return out
