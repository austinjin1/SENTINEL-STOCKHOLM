# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Fuse data layer: modality specs, targets, fold-wise preprocessing.

Everything that is *fit* (scalers, 16S CLR/PCA, target standardisation) is fit on the training rows
of a fold only (R4) through :class:`FoldPreprocessor`. Absent modalities never get values: their
features stay NaN for GBMs and become absent tokens for the deep model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...core import paths
from ...data import fuse_tiers as ftd

# --------------------------------------------------------------------------- targets
@dataclass(frozen=True)
class Target:
    name: str          # column in the tier table
    kind: str          # 'gauss' | 'binary'
    family: str        # variable family (query embedding)
    horizon: int       # days (0 = nowcast / same-event)
    cens_col: str | None = None   # left-censored flag (value is the bound)


def tier_d_targets(df: pd.DataFrame | None = None) -> list[Target]:
    out = []
    for v in ftd.TARGET_VARS + ["do_mgl_min"]:
        for h in ftd.HORIZONS:
            out.append(Target(f"y__{v}__h{h}", "gauss", v, h))
    for a in ftd.LAB_TARGETS:
        out.append(Target(f"ylab__{a}", "gauss", f"lab_{a}", 0, cens_col=f"ylab__{a}__cens"))
    for e in ("H1_hypoxia__any_h7", "H1_hypoxia__onset_h7", "H1_hypoxia_t4__any_h7",
              "H5b_chl_bloom__any_h7", "H5b_chl_bloom__onset_h7"):
        out.append(Target(f"ev__{e}", "binary", e.split("__")[0], 7))
    if df is not None:
        out = [t for t in out if t.name in df.columns and df[t.name].notna().any()]
    return out


P1_TARGETS = ["y__do_mgl__h7", "y__turb_fnu__h7", "y__no3n_mgl__h7", "y__chl_ugl__h7"]


def tier_d_target_matrix(df: pd.DataFrame, targets: list[Target]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Y [N,T] model-space values, observed mask [N,T], censored mask [N,T]).

    Lab targets are moved to model space (log(x + offset)); a censored lab value is replaced by its
    detection limit, a left-censoring bound (true value < DL); censored rows without a DL are dropped.
    """
    N, T = len(df), len(targets)
    Y = np.full((N, T), np.nan, np.float64)
    C = np.zeros((N, T), bool)
    for j, t in enumerate(targets):
        v = df[t.name].to_numpy(float)
        if t.name.startswith("ylab__"):
            a = t.name[len("ylab__"):]
            cens = df[t.cens_col].fillna(False).to_numpy(bool)
            dl = df[f"ylab__{a}__dl"].to_numpy(float)
            # Left-censored (below detection): the bound is the detection limit, never the published
            # sub-DL value (NEON publishes a number for most BDL analyses). Censored without a DL -> dropped.
            v = np.where(cens, dl, v)
            v = ftd.lab_to_model(a, v)
            C[:, j] = cens & np.isfinite(v)
        Y[:, j] = v
    M = np.isfinite(Y)
    return Y, M, C


# --------------------------------------------------------------------------- modalities
@dataclass
class ModalitySpec:
    name: str
    columns: list[str]
    groups: list[list[int]]
    group_names: list[str]
    present_col: str
    age_col: str | None = None
    hub: bool = False

    @property
    def n_features(self) -> int:
        return len(self.columns)


def group_columns(mod: str, cols: list[str], chunk: int = 6) -> tuple[list[list[int]], list[str]]:
    """Token groups: ``mod__X__feat`` columns are grouped by X; ``mod__X`` columns are chunked."""
    by_key: dict[str, list[int]] = {}
    loose: list[int] = []
    for i, c in enumerate(cols):
        parts = c.split("__")
        if len(parts) >= 3:
            by_key.setdefault(parts[1], []).append(i)
        else:
            loose.append(i)
    groups = list(by_key.values())
    names = list(by_key.keys())
    for k in range(0, len(loose), chunk):
        groups.append(loose[k:k + chunk])
        names.append(f"{mod}_chunk{k // chunk}")
    return groups, names


TIER_D_EXCLUDE_FEATURES = {"micro__sample_id"}


def tier_d_modalities(df: pd.DataFrame, n_micro_pcs: int = 32, include: list[str] | None = None) -> list[ModalitySpec]:
    cols = json.loads((paths.processed("fuse_tier_d") / "columns.json").read_text())
    specs = []
    for m in ftd.NON_HUB_D + ftd.HUB_D:
        if include is not None and m not in include:
            continue
        if f"{m}__present" not in df.columns and m not in ftd.HUB_D:
            continue
        cc = [c for c in cols.get(m, []) if c in df.columns and c not in TIER_D_EXCLUDE_FEATURES]
        age = f"{m}__age" if f"{m}__age" in df.columns else None
        if m == "micro":
            cc = [c for c in cc if c != "micro__age"] + [f"micro__pc{k}" for k in range(n_micro_pcs)]
        elif age in cc:
            cc = [c for c in cc if c != age]
        if m in ("chem",):
            pass
        if not cc and m not in ("micro",):
            continue
        if m not in ftd.HUB_D and not df[f"{m}__present"].any():
            continue
        chunk = 8 if m == "micro" else 6
        g, n = group_columns(m, cc, chunk=chunk)
        specs.append(ModalitySpec(m, cc, g, n, f"{m}__present", age, hub=m in ftd.HUB_D))
    return specs


# --------------------------------------------------------------------------- fold preprocessing
@dataclass
class FoldPreprocessor:
    """Fit on training rows only: robust scaling per feature, 16S CLR → PCA, target standardisation."""
    specs: list[ModalitySpec]
    n_micro_pcs: int = 32
    center_: dict = field(default_factory=dict)
    scale_: dict = field(default_factory=dict)
    y_mean_: np.ndarray | None = None
    y_std_: np.ndarray | None = None
    clr_: object = None
    pca_: object = None

    # ---- 16S
    def fit_micro(self, df: pd.DataFrame, train_mask: np.ndarray, genus: pd.DataFrame) -> None:
        from sklearn.decomposition import PCA
        ids = pd.unique(df.loc[train_mask & df["micro__present"].to_numpy(bool), "micro__sample_id"].dropna())
        C = genus.reindex(ids).fillna(0.0)
        prev = (C > 0).mean(0)
        keep = list(prev[prev >= 0.05].sort_values(ascending=False, kind="stable").index[:500])
        keep = [k for k in keep if k != "__unassigned__"]
        self.clr_ = keep
        X = self._clr(C)
        k = min(self.n_micro_pcs, X.shape[0] - 1, X.shape[1])
        self.pca_ = PCA(n_components=k, random_state=0).fit(X)

    def _clr(self, C: pd.DataFrame) -> np.ndarray:
        kept = C.reindex(columns=self.clr_, fill_value=0.0)
        other = C.sum(1) - kept.sum(1)
        X = np.log(np.column_stack([kept.to_numpy(float), other.to_numpy(float)]) + 0.5)
        return X - X.mean(1, keepdims=True)

    def micro_features(self, df: pd.DataFrame, genus: pd.DataFrame) -> pd.DataFrame:
        sid = df["micro__sample_id"]
        out = pd.DataFrame(np.nan, index=df.index, columns=[f"micro__pc{k}" for k in range(self.n_micro_pcs)])
        ok = sid.notna().to_numpy() & df["micro__present"].to_numpy(bool)
        if ok.any() and self.pca_ is not None:
            u = pd.unique(sid[ok])
            Z = self.pca_.transform(self._clr(genus.reindex(u).fillna(0.0)))
            zmap = pd.DataFrame(Z, index=u)
            vals = zmap.reindex(sid[ok].to_numpy()).to_numpy()
            out.iloc[np.flatnonzero(ok), :Z.shape[1]] = vals
        return out

    def augment(self, df: pd.DataFrame, genus: pd.DataFrame | None) -> pd.DataFrame:
        """Return ``df`` plus this fold's 16S PC columns (call after :meth:`fit_micro`)."""
        if genus is None or not any(s.name == "micro" for s in self.specs):
            return df
        mf = self.micro_features(df, genus)
        return pd.concat([df.drop(columns=[c for c in mf.columns if c in df.columns]), mf], axis=1)

    # ---- features
    def fit(self, df: pd.DataFrame, train_mask: np.ndarray, Y: np.ndarray | None = None) -> "FoldPreprocessor":
        for s in self.specs:
            pres = df[s.present_col].to_numpy(bool) if s.present_col in df else np.ones(len(df), bool)
            sub = df.loc[train_mask & pres, s.columns].astype(float).to_numpy(float)
            med = np.nanmedian(sub, axis=0) if len(sub) else np.zeros(len(s.columns))
            q75, q25 = (np.nanpercentile(sub, 75, axis=0), np.nanpercentile(sub, 25, axis=0)) if len(sub) else (1, 0)
            sd = np.nanstd(sub, axis=0) if len(sub) else np.ones(len(s.columns))
            sc = np.where(np.isfinite(q75 - q25) & ((q75 - q25) > 1e-9), (q75 - q25) / 1.349, sd)
            sc = np.where(np.isfinite(sc) & (sc > 1e-9), sc, 1.0)
            self.center_[s.name] = np.where(np.isfinite(med), med, 0.0)
            self.scale_[s.name] = sc
        if Y is not None:
            yt = Y[train_mask]
            self.y_mean_ = np.nanmean(yt, axis=0)
            self.y_std_ = np.nanstd(yt, axis=0)
            self.y_mean_ = np.where(np.isfinite(self.y_mean_), self.y_mean_, 0.0)
            self.y_std_ = np.where(np.isfinite(self.y_std_) & (self.y_std_ > 1e-9), self.y_std_, 1.0)
        return self

    def tensors(self, df: pd.DataFrame) -> dict:
        """Per modality: x [N, G, K] (scaled, NaN→0), m [N, G, K] observed mask, present [N],
        age [N] (days, NaN→0). Absent modality ⇒ present = False and m = 0."""
        out = {}
        for s in self.specs:
            X = (df[s.columns].astype(float).to_numpy(np.float64) - self.center_[s.name]) / self.scale_[s.name]
            X = np.clip(X, -10, 10)
            obs = np.isfinite(X)
            pres = df[s.present_col].to_numpy(bool) if s.present_col in df else np.ones(len(df), bool)
            obs &= pres[:, None]
            G = len(s.groups)
            K = max(len(g) for g in s.groups)
            xg = np.zeros((len(df), G, K), np.float32)
            mg = np.zeros((len(df), G, K), np.float32)
            for gi, g in enumerate(s.groups):
                xg[:, gi, :len(g)] = np.where(obs[:, g], X[:, g], 0.0)
                mg[:, gi, :len(g)] = obs[:, g]
            pres = pres & (mg.reshape(len(df), -1).sum(1) > 0)
            age = df[s.age_col].to_numpy(float) if s.age_col else np.zeros(len(df))
            age = np.where(np.isfinite(age), age, 0.0).astype(np.float32)
            out[s.name] = {"x": xg, "m": mg, "present": pres, "age": age}
        return out

    def standardize_y(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.y_mean_) / self.y_std_


def load_tier_d() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = paths.processed("fuse_tier_d")
    df = pd.read_parquet(root / "daily.parquet")
    genus = pd.read_parquet(root / "micro_genus.parquet")
    return df, genus


def tier_d_feature_frame(df: pd.DataFrame, specs: list[ModalitySpec], pre: FoldPreprocessor | None,
                         genus: pd.DataFrame | None) -> pd.DataFrame:
    """Raw model-space feature frame (NaN where absent) for GBM-type baselines; 16S PCs from ``pre``."""
    cols = []
    for s in specs:
        cols += [c for c in s.columns if not c.startswith("micro__pc")]
    X = df[cols].copy()
    for s in specs:
        if s.name not in ("time", "static"):
            pres = df[s.present_col].to_numpy(bool)
            sc = [c for c in s.columns if c in X.columns]
            X.loc[~pres, sc] = np.nan
    if pre is not None and genus is not None and any(s.name == "micro" for s in specs):
        mf = pre.micro_features(df, genus)
        X = pd.concat([X, mf], axis=1)
    for c in X.columns:
        if X[c].dtype != np.float64 and X[c].dtype != np.float32:
            X[c] = pd.to_numeric(X[c].astype(object).where(X[c].notna(), np.nan), errors="coerce").astype(float)
    return X
