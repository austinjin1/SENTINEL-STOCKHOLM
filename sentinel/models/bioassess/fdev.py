# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-F UK development design (protocol ``bioassess_f_protocol.md`` §9.4) and the comparator features of the
comparator table (``docs/preregistration/bioassess_f_comparators.md``).

Design: the M5-X X-U development design restricted to the allowed UK rows. One fold per Environment Agency area
``g``. Pseudo-test of ``g``: invertebrate rows of ``pool_fit`` sites in ``g`` dated 2016-2020 with at least one F-Z
primary label. Training pool of ``g``: allowed UK rows (invertebrate and diatom-only) dated up to 2015-12-31, outside
``g``, farther than 1 km from every pseudo-test station of ``g`` and in no location cluster of one. No French row
exists in this design. Every fit passes ``UKDev.assert_fit_ids`` (which calls ``ftrack.assert_fit_ids`` first).

Nothing here reads a French file, a locked row or a label of a disallowed UK row: the design comes from
``fdata.load_uk_allowed`` only.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.models.bioassess import fdata as FD
from sentinel.models.bioassess import ftrack as F

UKDEV_TRAIN_TO = dt.date(2015, 12, 31)
UKDEV_TEST_FROM = dt.date(2016, 1, 1)
UKDEV_TEST_TO = dt.date(2020, 12, 31)
TRACK = "F-Z"
GUARD_DESIGN = "dev"            # ftrack design name; the F-Z pool of ftrack is the same in both designs
MIN_TAXON_ROWS = 10
RF_MAX_ROWS = 60000
TABPFN_MAX_ROWS = 6000
TABPFN_TAXA = {"invertebrate": 300, "diatom": 150}
TABPFN_MAX_COLS = 500
EPT = ("Ephemeroptera", "Plecoptera", "Trichoptera")
UKDEV_DIR = paths.RUNS_ROOT / "bioassess" / "f_track" / "ukdev"


def fold_tag(area: str) -> str:
    """Fold name written in checkpoint records of this design."""
    return f"ukdev:{area}"


def _uk_roles() -> pd.Series:
    """site_id -> role of the M5 UK split (metadata only), as in ``ftrack.uk_allowed``."""
    from sentinel.models.bioassess import uk as UK

    P = paths.processed("uk_ea_ecology")
    sites = pd.read_parquet(P / "sites.parquet", columns=["site_id", "assemblage", "waterbody_type", "lat", "lon"])
    sites = sites.assign(site_id=sites["site_id"].astype(str))
    inv = sites[(sites["assemblage"] == "invertebrate") & (sites["waterbody_type"] != "WBES")]
    return UK.uk_site_roles(inv)


@dataclass
class UKDev:
    """The UK development design over an ``fdata.MADesign`` of allowed UK rows."""

    D: FD.MADesign
    role: pd.Series                                   # row key -> site role (invertebrate rows), else None
    primary: tuple = FD.PRIMARY[TRACK]
    _cache: dict = field(default_factory=dict)

    @classmethod
    def build(cls, D: FD.MADesign | None = None, roles: pd.Series | None = None) -> "UKDev":
        D = D or FD.load_uk_allowed()
        S = D.samples
        if not S["source"].isin(["uk", "ukd"]).all() or not S["uk_allowed"].all():
            raise F.FitGuardError("the UK development design holds allowed UK rows only")
        roles = _uk_roles() if roles is None else roles
        role = S["station"].astype(str).map(roles).where(S["has_inv"].to_numpy(bool))
        return cls(D, role)

    # ---------------------------------------------------------------------------------------------- rows
    def _dates(self) -> np.ndarray:
        return pd.to_datetime(self.D.samples["date"]).dt.date.to_numpy()

    def _period(self) -> np.ndarray:
        d = self._dates()
        return (d >= UKDEV_TEST_FROM) & (d <= UKDEV_TEST_TO)

    def test_mask(self) -> np.ndarray:
        S = self.D.samples
        lab = self.D.Y[list(self.primary)].notna().any(axis=1).to_numpy()
        return (S["source"] == "uk").to_numpy() & S["has_inv"].to_numpy(bool) & self._period() \
            & (self.role == "pool_fit").to_numpy() & lab & S["region"].notna().to_numpy()

    @property
    def areas(self) -> list:
        S = self.D.samples
        return sorted(S.loc[self.test_mask(), "region"].astype(str).unique())

    def test_ids(self, area: str) -> pd.Index:
        S = self.D.samples
        return S.index[self.test_mask() & (S["region"].astype(str) == area).to_numpy()]

    def context_ids(self, area: str) -> pd.Index:
        """Region-context candidates of the mirror rule (§9.1): the allowed invertebrate rows of the held-out area
        in its pseudo-test period. Inputs only; no label of these rows is read by a model."""
        S = self.D.samples
        return S.index[(S["source"] == "uk").to_numpy() & S["has_inv"].to_numpy(bool) & self._period()
                       & (S["region"].astype(str) == area).to_numpy()]

    def training_ids(self, area: str) -> pd.Index:
        if area in self._cache:
            return self._cache[area]
        from sklearn.neighbors import BallTree

        from sentinel.eval.splits import EARTH_RADIUS_KM

        S = self.D.samples
        pool = S.index.isin(F.training_ids(S, TRACK, None, GUARD_DESIGN))
        ok = pool & (self._dates() <= UKDEV_TRAIN_TO) & ~(S["region"].astype(str) == area).to_numpy()
        te = self.test_ids(area)
        if len(te) == 0:
            raise ValueError(f"area {area!r} has no pseudo-test row")
        q = S.loc[te, ["lat", "lon"]].drop_duplicates().to_numpy(float)
        tree = BallTree(np.radians(q), metric="haversine")
        idx = np.flatnonzero(ok)
        dist, _ = tree.query(np.radians(S.iloc[idx][["lat", "lon"]].to_numpy(float)), k=1)
        ok[idx[dist[:, 0] * EARTH_RADIUS_KM <= F.BUFFER_KM]] = False
        ok &= ~S["location_id"].isin(set(S.loc[te, "location_id"])).to_numpy()
        out = S.index[ok]
        if len(out.intersection(te)):
            raise AssertionError("development training and pseudo-test rows overlap")
        self._cache[area] = out
        return out

    def assert_fit_ids(self, ids, area: str) -> None:
        """THE guard of this design: inside the F-Z pool of ``ftrack`` and inside the fold's development pool."""
        ids = pd.Index(ids)
        F.assert_fit_ids(ids, self.D.samples, TRACK, None, GUARD_DESIGN)
        bad = ids.difference(self.training_ids(area))
        if len(bad):
            raise F.FitGuardError(f"refusing to fit on {len(bad)} rows outside the UK development pool of {area!r} "
                                  f"(e.g. {list(bad[:3])})")

    def report(self) -> dict:
        S, Y = self.D.samples, self.D.Y
        out = {}
        for k, a in enumerate(self.areas):
            te, tr = self.test_ids(a), self.training_ids(a)
            out[a] = {"k": k, "n_test": int(len(te)), "n_test_clusters": int(S.loc[te, "location_id"].nunique()),
                      "n_context": int(len(self.context_ids(a))), "n_train": int(len(tr)),
                      "n_train_inv": int(S.loc[tr, "has_inv"].sum()),
                      "n_test_labelled": {c: int(Y.loc[te, c].notna().sum()) for c in self.primary},
                      "n_train_labelled": {c: int(Y.loc[tr, c].notna().sum()) for c in self.primary}}
        return out


# ------------------------------------------------------------------------------------------------ taxon blocks
class TaxonBlocks:
    """Sparse modality-sample x ``mixed_otu`` count matrices of a design (an index; taxa are chosen per fold)."""

    def __init__(self, D: FD.MADesign, key: str = "mixed_otu"):
        from scipy import sparse

        self.S = D.samples
        self.M, self.keys, self.cols, self.total, self.ntax, self.index = {}, {}, {}, {}, {}, {}
        for m in FD.MODALITIES:
            t = D.tokens[m].dropna(subset=[key])
            keys = pd.Index(sorted(t["key"].astype(str).unique()))
            cols = pd.Index(sorted(t[key].astype(str).unique()))
            M = sparse.csr_matrix((t["count"].to_numpy(float), (keys.get_indexer(t["key"].astype(str)),
                                                               cols.get_indexer(t[key].astype(str)))),
                                  shape=(len(keys), len(cols)))
            M.sum_duplicates()
            self.M[m], self.keys[m], self.cols[m] = M, keys, cols
            self.total[m] = np.asarray(M.sum(1)).ravel()
            self.ntax[m] = np.asarray((M > 0).sum(1)).ravel()
        kc = {"invertebrate": "inv_key", "diatom": "diatom_key"}
        for m in FD.MODALITIES:
            k = self.S[kc[m]]
            self.index[m] = pd.Series(self.keys[m].get_indexer(k.astype(object).where(k.notna(), "").astype(str)),
                                      index=self.S.index)
        self._idx = None
        self._tok = D.tokens["invertebrate"]

    def pos(self, m: str, ids) -> np.ndarray:
        """Row positions in the matrix of modality ``m`` (-1: the row has no sample of that modality)."""
        return self.index[m].reindex(pd.Index(ids)).to_numpy(np.int64)

    def prevalence(self, m: str, ids) -> pd.Series:
        r = self.pos(m, ids)
        A = self.M[m][r[r >= 0]]
        return pd.Series(np.asarray((A > 0).sum(0)).ravel(), index=self.cols[m])

    def hellinger(self, m: str, ids, taxa, missing: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """(Hellinger values of ``taxa``: square root of the share of the sample total count, row-has-sample mask)."""
        r = self.pos(m, ids)
        has = r >= 0
        out = np.full((len(r), len(taxa)), missing, float)
        if has.any() and len(taxa):
            A = self.M[m][r[has]][:, self.cols[m].get_indexer(pd.Index(taxa))].toarray()
            tot = self.total[m][r[has]]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[has] = np.sqrt(np.where(tot[:, None] > 0, A / tot[:, None], 0.0))
        return out, has

    def descriptors(self, m: str, ids, missing: float = np.nan) -> np.ndarray:
        """log10 total count and log10 number of taxa of the sample (M5-X count descriptors)."""
        r = self.pos(m, ids)
        has = r >= 0
        out = np.full((len(r), 2), missing, float)
        out[has, 0] = np.log10(np.maximum(self.total[m][r[has]], 1.0))
        out[has, 1] = np.log10(np.maximum(self.ntax[m][r[has]], 1.0))
        return out

    def indices(self, ids) -> np.ndarray:
        """Computed invertebrate indices (comparator table §5, F-Z): richness, EPT richness, EPT share (EPT taxa /
        all taxa), Shannon (natural log, on the relative abundance of the harmonised taxa). NaN without a sample."""
        if self._idx is None:
            t = self._tok.dropna(subset=["mixed_otu"])
            g = t.groupby(["key", "mixed_otu"], as_index=False).agg(count=("count", "sum"), order=("order", "first"))
            g["ept"] = g["order"].isin(EPT)
            tot = g.groupby("key")["count"].transform("sum")
            p = g["count"] / tot
            g["h"] = -(p * np.log(p.where(p > 0, 1.0)))
            a = g.groupby("key").agg(richness=("mixed_otu", "size"), ept_richness=("ept", "sum"), shannon=("h", "sum"))
            a["ept_share"] = a["ept_richness"] / a["richness"]
            self._idx = a[["richness", "ept_richness", "ept_share", "shannon"]].astype(float)
        k = self.S["inv_key"].reindex(pd.Index(ids))
        return self._idx.reindex(k.astype(object).where(k.notna(), "").astype(str)).to_numpy(float)


INDEX_COLS = ("richness", "ept_richness", "ept_share", "shannon")


@dataclass
class FFeaturizer:
    """Comparator features of the comparator table. Fitted on the labelled training rows of one fit; no label read.

    ``modalities``: taxon blocks used. ``descriptors``: count descriptors per used modality. ``numeric``: numeric
    context columns (missing = training median); an altitude column adds its missing flag. ``tokens``: source and
    field-protocol one-hots and the diatom offset. ``max_taxa``: per-modality cap (most prevalent). ``max_cols``:
    diatom taxa are dropped from the least prevalent until the column count fits. ``nan_blocks``: True = a missing
    assemblage block is NaN (TabPFN, LightGBM); False = 0 plus one indicator column per block that can be missing."""

    modalities: tuple = ("invertebrate",)
    descriptors: bool = True
    numeric: tuple = ()
    tokens: bool = False
    max_taxa: dict | None = None
    max_cols: int | None = None
    nan_blocks: bool = False
    taxa_: dict = field(default_factory=dict)
    fill_: dict = field(default_factory=dict)
    cats_: dict = field(default_factory=dict)
    flag_: dict = field(default_factory=dict)
    info_: dict = field(default_factory=dict)

    def _other_cols(self) -> int:
        n = 0
        for m in self.modalities:
            n += (2 if self.descriptors else 0) + (0 if self.nan_blocks else int(self.flag_.get(m, False)))
        n += len(self.numeric) + int("altitude_m" in self.numeric)
        if self.tokens:
            n += len(self.cats_["source"]) + sum(len(self.cats_["protocol_" + m]) for m in self.modalities)
            n += int("diatom" in self.modalities)
        return n

    def fit(self, TB: TaxonBlocks, ids) -> "FFeaturizer":
        ids = pd.Index(ids)
        S = TB.S.loc[ids]
        for m in self.modalities:
            prev = TB.prevalence(m, ids)
            prev = prev[prev >= MIN_TAXON_ROWS].sort_values(ascending=False, kind="stable")
            cap = (self.max_taxa or {}).get(m)
            self.taxa_[m] = list(prev.index[:cap] if cap else prev.index)
            self.flag_[m] = bool((TB.pos(m, ids) < 0).any()) or m == "diatom"
        self.fill_ = {}
        for c in self.numeric + (("diatom_offset_days",) if self.tokens and "diatom" in self.modalities else ()):
            v = float(pd.to_numeric(S[c], errors="coerce").median())
            self.fill_[c] = v if np.isfinite(v) else 0.0
        if self.tokens:
            self.cats_ = {"source": sorted(S["source"].astype(str).unique())}
            for m in self.modalities:
                self.cats_["protocol_" + m] = sorted(S["protocol_" + m].dropna().astype(str).unique())
        self.info_ = {"n_taxa": {m: len(v) for m, v in self.taxa_.items()}, "diatom_taxa_dropped": 0}
        if self.max_cols is not None:
            over = sum(len(v) for v in self.taxa_.values()) + self._other_cols() - int(self.max_cols)
            if over > 0:
                if "diatom" not in self.taxa_ or len(self.taxa_["diatom"]) < over:
                    raise ValueError("column limit cannot be met by dropping diatom taxa")
                self.taxa_["diatom"] = self.taxa_["diatom"][:-over]
                self.info_["diatom_taxa_dropped"] = int(over)
                self.info_["n_taxa"]["diatom"] = len(self.taxa_["diatom"])
        return self

    def transform(self, TB: TaxonBlocks, ids) -> np.ndarray:
        ids = pd.Index(ids)
        S = TB.S.loc[ids]
        miss = np.nan if self.nan_blocks else 0.0
        parts = []
        for m in self.modalities:
            H, has = TB.hellinger(m, ids, self.taxa_[m], miss)
            parts.append(H)
            if self.descriptors:
                parts.append(TB.descriptors(m, ids, miss))
            if not self.nan_blocks and self.flag_[m]:
                parts.append((~has).astype(float)[:, None])
        for c in self.numeric:
            x = pd.to_numeric(S[c], errors="coerce").to_numpy(float)
            bad = ~np.isfinite(x)
            parts.append(np.where(bad, self.fill_[c], x)[:, None])
            if c == "altitude_m":
                parts.append(bad.astype(float)[:, None])
        if self.tokens:
            parts.append(np.column_stack([(S["source"].astype(str) == k).to_numpy(float)
                                          for k in self.cats_["source"]]))
            for m in self.modalities:
                p = S["protocol_" + m].astype(object)
                if self.cats_["protocol_" + m]:
                    parts.append(np.column_stack([(p == k).to_numpy(float) for k in self.cats_["protocol_" + m]]))
            if "diatom" in self.modalities:
                x = pd.to_numeric(S["diatom_offset_days"], errors="coerce").to_numpy(float)
                x = np.where(S["diatom_key"].notna().to_numpy() & np.isfinite(x), x, self.fill_["diatom_offset_days"])
                parts.append(x[:, None])
        X = np.hstack(parts) if parts else np.zeros((len(ids), 0))
        if self.max_cols is not None and X.shape[1] > int(self.max_cols):
            raise AssertionError(f"{X.shape[1]} columns exceed the limit of {self.max_cols}")
        return X


# ------------------------------------------------------------------------------------------------ comparators
ALT_GEO = ("lat", "lon", "altitude_m")
# name -> (learner, featurizer keyword arguments). F-Z input sets by ``ftrack.FZ_INPUT_SETS`` (lead ruling (a)).
ML_SPECS = {
    "rf__community": ("rf", {}),
    "rf__coords": ("rf", {"numeric": F.FZ_INPUT_SETS["coords"]}),
    "rf__nocoords": ("rf", {"numeric": F.FZ_INPUT_SETS["nocoords"]}),
    "gbm__community": ("gbm", {"nan_blocks": True}),
    "gbm__coords": ("gbm", {"numeric": F.FZ_INPUT_SETS["coords"], "nan_blocks": True}),
    "gbm__nocoords": ("gbm", {"numeric": F.FZ_INPUT_SETS["nocoords"], "nan_blocks": True}),
    "tabpfn__community": ("tabpfn", {"max_taxa": {"invertebrate": 300}, "nan_blocks": True}),
    "tabpfn__coords": ("tabpfn", {"numeric": F.FZ_INPUT_SETS["coords"], "max_taxa": {"invertebrate": 300},
                                  "nan_blocks": True}),
    "tabpfn__nocoords": ("tabpfn", {"numeric": F.FZ_INPUT_SETS["nocoords"], "max_taxa": {"invertebrate": 300},
                                    "nan_blocks": True}),
    # fixed members of #8 ``ens_zs`` (lead ruling B2; comparator table §7): both assemblages, context without
    # coordinates, no count descriptors
    "b2_rf_zs": ("rf", {"modalities": FD.MODALITIES, "descriptors": False, "numeric": F.CONTEXT_COLS_NO_COORDS,
                        "tokens": True}),
    "b2_tabpfn_zs": ("tabpfn", {"modalities": FD.MODALITIES, "descriptors": False,
                                "numeric": F.CONTEXT_COLS_NO_COORDS, "tokens": True, "max_taxa": dict(TABPFN_TAXA),
                                "max_cols": TABPFN_MAX_COLS, "nan_blocks": True}),
}
SIMPLE = ("pool_mean", "month_clim", "rf_geography", "nearest_station")
TRANSFER = ("wapls2", "knn_mat")
INDEX_MODELS = ("idx_ols", "idx_ols_geo", "idx_rf", "idx_rf_geo")
# UK design: only the ``nocoords`` ensembles (comparator table §9, line of 2026-11-11T00:00Z)
ENSEMBLES = {"ens_rf_tabpfn__nocoords": ("rf__nocoords", "tabpfn__nocoords"),
             "ens_rf_gbm_tabpfn__nocoords": ("rf__nocoords", "gbm__nocoords", "tabpfn__nocoords")}
MEMBERS_B2 = ("b2_rf_zs", "b2_tabpfn_zs")
# reference set of the UK development design (comparator table §9, line of 2026-11-10): the comparator development
# table of gate G5 for this design is complete when each of these has stored seed-0 predictions for every area
UK_REFERENCES = SIMPLE + TRANSFER + INDEX_MODELS + ("tabpfn__community", "tabpfn__coords", "tabpfn__nocoords",
                                                    "rf__nocoords", "gbm__nocoords")
B2_POOL = "all"                 # members train on the rows of #7 (invertebrate and diatom-only rows)


def _strata(S: pd.DataFrame) -> np.ndarray:
    return (S["source"].astype(str) + "|" + S["region"].astype(str)).to_numpy()


def fit_predict(name: str, target: str, dev_or_design, TB: TaxonBlocks, train_ids, test_ids, *, seed: int = 0,
                n_jobs: int = 5, device: str = "cuda", info: dict | None = None) -> np.ndarray:
    """Predictions of one comparator (or B2 member) for one target column. Fit on the labelled rows of
    ``train_ids`` only; the caller has passed ``train_ids`` through the fit guard of its design."""
    from sentinel.models.bioassess import xdev as XD

    D = dev_or_design.D if hasattr(dev_or_design, "D") else dev_or_design
    train_ids, test_ids = pd.Index(train_ids), pd.Index(test_ids)
    if len(train_ids.intersection(test_ids)):
        raise AssertionError("train and test ids overlap")
    S = D.samples
    if name not in MEMBERS_B2:                                # invertebrate methods: rows with an invertebrate list
        train_ids = train_ids[S.loc[train_ids, "has_inv"].to_numpy(bool)]
    y_all = D.Y.loc[train_ids, target].to_numpy(float)
    ok = np.isfinite(y_all)
    lab, y = train_ids[ok], y_all[ok]
    if len(lab) == 0:
        return np.full(len(test_ids), np.nan)
    s_lab, s_te = S.loc[lab], S.loc[test_ids]
    if name == "pool_mean":
        return np.full(len(test_ids), float(np.mean(y)))
    if name == "month_clim":
        mo = pd.to_datetime(s_lab["date"]).dt.month.to_numpy()
        g = pd.Series(y).groupby(mo).agg(["mean", "size"])
        clim = g["mean"].where(g["size"] >= 10)
        out = pd.to_datetime(s_te["date"]).dt.month.map(clim).to_numpy(float)
        return np.where(np.isfinite(out), out, float(np.mean(y)))
    if name == "nearest_station":
        from sklearn.neighbors import BallTree

        st = pd.DataFrame({"station": s_lab["station"].astype(str).to_numpy(), "lat": s_lab["lat"].to_numpy(float),
                           "lon": s_lab["lon"].to_numpy(float), "y": y}).groupby("station").mean()
        tree = BallTree(np.radians(st[["lat", "lon"]].to_numpy(float)), metric="haversine")
        _, idx = tree.query(np.radians(s_te[["lat", "lon"]].to_numpy(float)), k=1)
        return st["y"].to_numpy(float)[idx[:, 0]]
    strata = _strata(s_lab)
    if name == "rf_geography":
        fz = FFeaturizer(modalities=(), descriptors=False, numeric=ALT_GEO).fit(TB, lab)
        pick = XD.stratified_subsample(strata, RF_MAX_ROWS, seed)
        Xtr, Xte = fz.transform(TB, lab[pick]), fz.transform(TB, test_ids)
        return XD._fit_learner("rf", Xtr[:, :3], y[pick], Xte[:, :3], seed, n_jobs)
    if name in INDEX_MODELS:
        pick = XD.stratified_subsample(strata, RF_MAX_ROWS, seed)

        def feats(ids, fill=None):
            A = TB.indices(ids)
            if name.endswith("_geo"):
                A = np.hstack([A, S.loc[ids, list(ALT_GEO)].to_numpy(float)])
            return A

        Xtr, Xte = feats(lab[pick]), feats(test_ids)
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr, Xte = np.where(np.isfinite(Xtr), Xtr, med), np.where(np.isfinite(Xte), Xte, med)
        return XD._fit_learner("ols" if name.startswith("idx_ols") else "rf", Xtr, y[pick], Xte, seed, n_jobs)
    if name in TRANSFER:
        fz = FFeaturizer(descriptors=False).fit(TB, lab)
        Htr, Hte = fz.transform(TB, lab), fz.transform(TB, test_ids)
        if name == "wapls2":
            from sentinel.models.bioassess.baselines import wapls_fit_predict

            return wapls_fit_predict(Htr ** 2, y, Hte ** 2, 2)
        from sklearn.neighbors import KNeighborsRegressor

        return KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=n_jobs).fit(Htr, y).predict(Hte)
    if name in ML_SPECS:
        kind, kw = ML_SPECS[name]
        cap = TABPFN_MAX_ROWS if kind == "tabpfn" else RF_MAX_ROWS
        pick = XD.stratified_subsample(strata, cap, seed)
        fz = FFeaturizer(**kw).fit(TB, lab)
        if info is not None:
            info.update(fz.info_, n_fit_rows=int(len(pick)), n_labelled=int(len(lab)))
        return XD._fit_learner(kind, fz.transform(TB, lab[pick]), y[pick], fz.transform(TB, test_ids), seed, n_jobs,
                               device)
    raise ValueError(f"unknown comparator {name}")


__all__ = ["UKDev", "TaxonBlocks", "FFeaturizer", "fit_predict", "fold_tag", "ML_SPECS", "SIMPLE", "TRANSFER",
           "INDEX_MODELS", "ENSEMBLES", "MEMBERS_B2", "UK_REFERENCES", "UKDEV_DIR", "INDEX_COLS", "TRACK",
           "GUARD_DESIGN"]
