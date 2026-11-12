# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""BioSet-MA: multi-assemblage set transformer with Student-t heads (M5-F; protocol §9.1).

New module. The frozen ``bioset.py`` is imported (vocabulary, set encoding) and never edited.

**Tokens.** One token per taxon of a modality sample (invertebrates, benthic diatoms):
flat name embedding + sum of lineage embeddings (phylum .. genus, small initial scale, so the model starts at the
flat embedding) + modality embedding + a linear map of six token features: the four BioSet abundance features, the
*pool prevalence* (share of the training samples of the same modality within 200 km that hold the taxon, the
sample itself left out) and a prevalence-missing flag (pool smaller than 10 samples, or a name unseen in the fold).

**Encoder.** Two pre-LN self-attention blocks shared by both modalities, PMA(1) ⊕ abundance-weighted mean.
It runs only on the rows that have the modality.

**Fusion.** One pre-LN self-attention block over four slots [invertebrate vector, diatom vector, context token,
region token] with a padding mask, then the masked mean. The optional region token is attention-pooled from the
community vectors of other rows of the same (pseudo-)region; labels never enter it.

**Heads.** Trunk 2 × 256 GELU; per target a Student-t (ν = 4) location and log-scale with a per-source scale
offset; masked negative log-likelihood on training-fold standardised targets. Optional hinge penalties keep the
predicted locations chemically coherent.

**Fit discipline.** :class:`MAFeaturizer` is fitted on training ids only (vocabulary, rare-name mask, prevalence
tables, standardisers). Callers pass every id through ``ftrack.assert_fit_ids`` first. Checkpoints are saved with a
training-id record and are loaded only through ``ftrack.load_checkpoint``.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.models.bioassess import bioset as BS
from sentinel.models.bioassess.bioset import LEVELS, N_ABUND, PAD
from sentinel.models.bioassess.ftrack import CONTEXT_COLS, CONTEXT_COLS_NO_COORDS   # F-Z: no coordinates

MODALITIES = ("invertebrate", "diatom")
KEY_COL = {"invertebrate": "inv_key", "diatom": "diatom_key"}
N_TOKEN_FEAT = N_ABUND + 2          # abundance features, pool prevalence, prevalence-missing flag
I_PREV, I_MISS = N_ABUND, N_ABUND + 1
N_MOD_FEAT = 3                      # log10 total count, log10 number of taxa (standardised), date offset / 90
N_SLOTS = 4                         # invertebrate, diatom, context, region
NAME = LEVELS.index("name")
LOG_SCALE_RANGE = (-6.0, 4.0)

ALTITUDE_COL = "altitude_m"

DEFAULTS = {
    "d_model": 96, "n_layers": 2, "n_heads": 4, "dropout": 0.1, "trunk": 256, "nu": 4.0,
    "modalities": MODALITIES, "token_cap": {"invertebrate": 150, "diatom": 100},
    "min_name_count": 3, "lineage_std": 0.1,
    "context_cols": CONTEXT_COLS,
    "token_dropout": 0.1, "modality_dropout": 0.3, "source_dropout": 0.1,
    "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 256, "max_epochs": 100, "patience": 10, "amp": True,
    "inner_val_frac": 0.15, "inner_block_km": 50.0,
    "prevalence_km": 200.0, "prevalence_cell_km": 50.0, "prevalence_min_pool": 10,
    "region_context": False, "region_max": 256, "pseudo_region_km": (50.0, 150.0), "episode_rows": 64,
    "val_region_km": 100.0,
    "coherence_weight": 0.0, "do_tol": 0.06,
    "embed_dim": 32,
}
# The BioSet-MA recipes of protocol §9.3 (differences from DEFAULTS only).
RECIPES = {
    "ma_inv": {"modalities": ("invertebrate",)},
    "ma_full": {},
    "ma_full_rc": {"region_context": True},
    "ma_full_coh": {"coherence_weight": 0.1},
    "ma_zs": {"region_context": True, "context_cols": CONTEXT_COLS_NO_COORDS},
}


def recipe(name: str, **over) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(copy.deepcopy(RECIPES[name]))
    cfg.update(over)
    cfg["name"] = name
    return cfg


# ------------------------------------------------------------------------------------------------ geometry
def _km_per_deg() -> float:
    from sentinel.eval.splits import EARTH_RADIUS_KM

    return math.pi * EARTH_RADIUS_KM / 180.0


def grid_cells(lat, lon, cell_km: float, dlat: float = 0.0, dlon: float = 0.0):
    """(row, col, centre_lat, centre_lon) of ~square cells of side ``cell_km`` (per-row cos(lat)); ``dlat``/``dlon``
    shift the grid origin (degrees)."""
    k = _km_per_deg()
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    row = np.floor((lat + 90.0 + dlat) * k / cell_km).astype(np.int64)
    clat = (row + 0.5) * cell_km / k - 90.0 - dlat
    kx = k * np.maximum(np.cos(np.radians(clat)), 1e-3)
    col = np.floor((lon + 180.0 + dlon) * kx / cell_km).astype(np.int64)
    clon = (col + 0.5) * cell_km / kx - 180.0 - dlon
    return row, col, clat, clon


def cell_codes(lat, lon, cell_km: float, dlat: float = 0.0, dlon: float = 0.0) -> np.ndarray:
    """Dense integer codes (0..n-1) of the grid cells of the points."""
    row, col, _, _ = grid_cells(lat, lon, cell_km, dlat, dlon)
    _, inv = np.unique(np.column_stack([row, col]), axis=0, return_inverse=True)
    return np.asarray(inv, np.int64).reshape(-1)


def pseudo_regions(lat, lon, rng, km=(50.0, 150.0)) -> np.ndarray:
    """Random spatial clusters for episodic training: a grid of random side in ``km`` and random origin."""
    side = float(rng.uniform(km[0], km[1]))
    span = side / _km_per_deg()
    return cell_codes(lat, lon, side, float(rng.uniform(0, span)), float(rng.uniform(0, span)))


# ------------------------------------------------------------------------------------------------ pool prevalence
@dataclass
class PoolPrevalence:
    """Per-cell document frequencies of name ids among the training samples of one modality.

    The pool of a sample is every training sample in a grid cell whose centre lies within ``radius_km`` of the centre
    of the sample's own cell. Built from inputs only (taxon lists and coordinates)."""

    cell_km: float
    radius_km: float
    min_pool: int
    centres: np.ndarray        # [C, 2] lat, lon of the training cells
    counts: object             # scipy csr [C, V]: samples of the cell that hold the name id
    n: np.ndarray              # [C] samples per cell
    keys: frozenset            # sample keys in the pool (left out of their own prevalence)

    @classmethod
    def fit(cls, keys, lat, lon, name_ids: np.ndarray, mask: np.ndarray, n_names: int, *, cell_km: float = 50.0,
            radius_km: float = 200.0, min_pool: int = 10) -> "PoolPrevalence":
        """``name_ids`` [U, T] and ``mask`` [U, T] of the training samples ``keys`` at (``lat``, ``lon``)."""
        from scipy import sparse

        U = len(keys)
        if U == 0:
            return cls(float(cell_km), float(radius_km), int(min_pool), np.zeros((0, 2)),
                       sparse.csr_matrix((0, n_names), dtype=np.float32), np.zeros(0, np.float32), frozenset())
        row, col, clat, clon = grid_cells(lat, lon, cell_km)
        cells, cidx = np.unique(np.column_stack([row, col]), axis=0, return_inverse=True)
        cidx = np.asarray(cidx).reshape(-1)
        centres = np.zeros((len(cells), 2))
        centres[cidx, 0], centres[cidx, 1] = clat, clon
        r, t = np.nonzero(mask & (name_ids != PAD))
        B = sparse.coo_matrix((np.ones(len(r), np.float32), (r, name_ids[r, t])), shape=(U, n_names)).tocsr()
        B.data[:] = 1.0                                           # presence (a name listed twice counts once)
        M = sparse.coo_matrix((np.ones(U, np.float32), (cidx, np.arange(U))), shape=(len(cells), U)).tocsr()
        return cls(float(cell_km), float(radius_km), int(min_pool), centres, (M @ B).tocsr(),
                   np.bincount(cidx, minlength=len(cells)).astype(np.float32), frozenset(map(str, keys)))

    def lookup(self, keys, lat, lon, name_ids: np.ndarray, mask: np.ndarray):
        """(prevalence [U, T] float32, missing [U, T] bool) for the samples ``keys``; padding gets (0, False)."""
        from scipy import sparse
        from sklearn.neighbors import BallTree

        from sentinel.eval.splits import EARTH_RADIUS_KM

        U, T = name_ids.shape
        prev = np.zeros((U, T), np.float32)
        miss = np.zeros((U, T), bool)
        if U == 0:
            return prev, miss
        row, col, clat, clon = grid_cells(lat, lon, self.cell_km)
        q, qidx = np.unique(np.column_stack([row, col]), axis=0, return_inverse=True)
        qidx = np.asarray(qidx).reshape(-1)
        qc = np.zeros((len(q), 2))
        qc[qidx, 0], qc[qidx, 1] = clat, clon
        if len(self.centres):
            tree = BallTree(np.radians(self.centres), metric="haversine")
            nb = tree.query_radius(np.radians(qc), r=(self.radius_km + 1e-6) / EARTH_RADIUS_KM)
        else:
            nb = [np.zeros(0, int)] * len(q)
        rr = np.concatenate([np.full(len(i), k) for k, i in enumerate(nb)]) if len(nb) else np.zeros(0, int)
        cc = np.concatenate(nb) if len(nb) else np.zeros(0, int)
        A = sparse.coo_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(len(q), len(self.centres))).tocsr()
        npool = np.asarray(A @ self.n).reshape(-1)
        own = np.fromiter((str(k) in self.keys for k in keys), bool, U).astype(np.float32)
        n_eff = npool[qidx] - own                                           # [U]
        P = A @ self.counts                                                 # csr [Q, V]
        for s in range(0, len(q), 256):                                     # dense by blocks of query cells
            blk = np.asarray(P[s:s + 256].todense(), np.float32)
            rows = np.flatnonzero((qidx >= s) & (qidx < s + 256))
            if len(rows) == 0:
                continue
            c = blk[(qidx[rows] - s)[:, None], name_ids[rows]] - own[rows, None]
            prev[rows] = np.clip(c / np.maximum(n_eff[rows], 1.0)[:, None], 0.0, 1.0)
        miss = (n_eff < self.min_pool)[:, None] | (name_ids == PAD)
        prev[miss | ~mask] = 0.0
        miss = miss & mask
        return prev, miss


# ------------------------------------------------------------------------------------------------ featurizer
@dataclass
class ModalityData:
    lev: np.ndarray            # [U, T, 6] int32 lineage ids of the unique modality samples
    feat: np.ndarray           # [U, T, N_TOKEN_FEAT] float32
    mask: np.ndarray           # [U, T] bool
    mfeat: np.ndarray          # [U, 2] float32 standardised log10 total count, log10 number of taxa
    index: np.ndarray          # [N] row -> unique modality sample (-1 = modality absent)
    offset: np.ndarray         # [N] float32 date offset of the attached sample / 90 days
    proto: np.ndarray          # [N] int64 protocol id (0 = unknown)


@dataclass
class MAData:
    ids: np.ndarray
    mod: dict                  # modality -> ModalityData
    ctx: np.ndarray            # [N, 2C] float32 standardised context (missing -> 0) + missing flags
    src: np.ndarray            # [N] int64 source id (0 = unknown)
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray = None     # [N] station altitude in metres, NaN = missing (pressure factor of the DO term)
    _dev: dict = field(default_factory=dict, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.ids)

    def on(self, device) -> dict:
        """The large arrays as tensors on ``device`` (built once; one device at a time). Batches are assembled
        there by indexing (lead ruling (o))."""
        key = str(device)
        if key not in self._dev:
            torch, _ = BS._torch()
            t = {m: (torch.as_tensor(md.lev, device=device), torch.as_tensor(md.feat, device=device))
                 for m, md in self.mod.items()}
            t["ctx"] = torch.as_tensor(self.ctx, device=device)
            self._dev = {key: t}
        return self._dev[key]

    def present(self) -> np.ndarray:
        return np.column_stack([m.index >= 0 for m in self.mod.values()]).any(1)


def nan_stats(x: np.ndarray):
    """Column mean and standard deviation over finite entries (0 and 1 where a column has none / no spread)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    n = ok.sum(0)
    mu = np.where(ok, x, 0.0).sum(0) / np.maximum(n, 1)
    var = (np.where(ok, x - mu, 0.0) ** 2).sum(0) / np.maximum(n, 1)
    sd = np.sqrt(var)
    return mu, np.where(sd > 0, sd, 1.0)


def usable_ids(samples: pd.DataFrame, ids, modalities=MODALITIES) -> pd.Index:
    """Rows of ``ids`` that hold at least one of the modalities (a diatom-only row is unusable for ``ma_inv``)."""
    S = samples.loc[pd.Index(ids)]
    ok = np.zeros(len(S), bool)
    for m in modalities:
        ok |= S[KEY_COL[m]].notna().to_numpy()
    return S.index[ok]


def _encode_modality(tokens: pd.DataFrame, keys: np.ndarray, vocab, allowed, cap: int, chunk: int = 20000):
    t = tokens[tokens["key"].isin(set(keys))]
    n_tok = t.groupby("key").size()
    T = int(min(cap, n_tok.max())) if len(n_tok) else 1
    U = len(keys)
    lev = np.zeros((U, T, len(LEVELS)), np.int32)
    ab = np.zeros((U, T, N_ABUND), np.float32)
    mask = np.zeros((U, T), bool)
    for s in range(0, U, chunk):
        ks = keys[s:s + chunk]
        e = BS.restrict(BS.encode(t, ks, vocab, max_tokens=cap, id_col="key"), allowed)
        w = min(T, e.lev.shape[1])
        lev[s:s + len(ks), :w] = e.lev[:, :w]
        ab[s:s + len(ks), :w] = e.ab[:, :w]
        mask[s:s + len(ks), :w] = e.mask[:, :w]
    agg = t.groupby("key")["count"].agg(["sum", "size"]).reindex(keys)
    stats = np.column_stack([np.log10(np.maximum(agg["sum"].to_numpy(float), 1.0)),
                             np.log10(np.maximum(agg["size"].to_numpy(float), 1.0))])
    return lev, ab, mask, np.nan_to_num(stats).astype(np.float32)


def _unique_keys(S: pd.DataFrame, modality: str):
    """(unique keys, row -> unique index with -1 where absent, first row of each unique key)."""
    k = S[KEY_COL[modality]].to_numpy(object)
    has = np.array([x is not None and x == x for x in k], bool)
    index = np.full(len(S), -1, np.int64)
    if not has.any():
        return np.zeros(0, object), index, np.zeros(0, np.int64)
    uniq, first, inv = np.unique(k[has].astype(str), return_index=True, return_inverse=True)
    index[has] = np.asarray(inv).reshape(-1)
    return uniq.astype(object), index, np.flatnonzero(has)[first]


@dataclass
class MAFeaturizer:
    """Everything that turns design rows into model inputs. Fitted on training ids only; no label is read."""

    cfg: dict
    vocab: object = None
    allowed: list = None
    prevalence: dict = field(default_factory=dict)
    source_map: dict = field(default_factory=dict)
    proto_map: dict = field(default_factory=dict)
    ctx_mean: np.ndarray = None
    ctx_std: np.ndarray = None
    mod_mean: dict = field(default_factory=dict)
    mod_std: dict = field(default_factory=dict)
    train_keys: frozenset = frozenset()

    @property
    def modalities(self) -> tuple:
        return tuple(m for m in MODALITIES if m in tuple(self.cfg["modalities"]))

    def fit(self, design, train_ids) -> "MAFeaturizer":
        cfg = self.cfg
        ids = pd.Index(train_ids)
        if not ids.is_unique:
            raise ValueError("training ids must be unique")
        S = design.samples.loc[ids]
        self.train_keys = frozenset(map(str, ids))
        uk = {m: _unique_keys(S, m) for m in self.modalities}
        toks = {m: design.tokens[m][design.tokens[m]["key"].isin(set(uk[m][0]))] for m in self.modalities}
        self.vocab = BS.TaxonVocab.build(*toks.values())
        self.allowed = self.vocab.allowed_from(pd.concat(list(toks.values()), ignore_index=True),
                                               int(cfg["min_name_count"]))
        lat, lon = S["lat"].to_numpy(float), S["lon"].to_numpy(float)
        for m in self.modalities:
            keys, _, first = uk[m]
            lev, _, mask, stats = _encode_modality(toks[m], keys, self.vocab, self.allowed, int(cfg["token_cap"][m]))
            self.prevalence[m] = PoolPrevalence.fit(
                keys, lat[first], lon[first], lev[..., NAME], mask, self.vocab.size("name"),
                cell_km=float(cfg["prevalence_cell_km"]), radius_km=float(cfg["prevalence_km"]),
                min_pool=int(cfg["prevalence_min_pool"]))
            self.mod_mean[m] = stats.mean(0) if len(stats) else np.zeros(2, np.float32)
            sd = stats.std(0) if len(stats) else np.ones(2, np.float32)
            self.mod_std[m] = np.where(sd > 0, sd, 1.0).astype(np.float32)
            names = sorted(set(S["protocol_" + m].dropna().astype(str))) if "protocol_" + m in S else []
            self.proto_map[m] = {p: i + 1 for i, p in enumerate(names)}
        self.source_map = {s: i + 1 for i, s in enumerate(sorted(set(S["source"].astype(str))))}
        self.ctx_mean, self.ctx_std = nan_stats(S[list(cfg["context_cols"])].to_numpy(float))
        return self

    @property
    def n_ctx(self) -> int:
        return 2 * len(self.cfg["context_cols"])

    def transform(self, design, ids) -> MAData:
        ids = pd.Index(ids)
        if not ids.is_unique:
            raise ValueError("ids must be unique")
        S = design.samples.loc[ids]
        lat, lon = S["lat"].to_numpy(float), S["lon"].to_numpy(float)
        mod = {}
        for m in self.modalities:
            keys, index, first = _unique_keys(S, m)
            lev, ab, mask, stats = _encode_modality(design.tokens[m], keys, self.vocab, self.allowed,
                                                    int(self.cfg["token_cap"][m]))
            empty = ~mask.any(1)                       # a key without any token row counts as an absent modality
            if empty.any():
                index[np.isin(index, np.flatnonzero(empty))] = -1
            prev, miss = self.prevalence[m].lookup(keys, lat[first], lon[first], lev[..., NAME], mask)
            feat = np.concatenate([ab, prev[..., None], miss[..., None].astype(np.float32)], -1)
            off = np.zeros(len(S), np.float32)
            if m == "diatom" and "diatom_offset_days" in S:
                off = np.nan_to_num(S["diatom_offset_days"].to_numpy(float) / 90.0).astype(np.float32)
                off[index < 0] = 0.0
            pcol = "protocol_" + m
            proto = S[pcol].map(self.proto_map[m]).fillna(0).to_numpy(np.int64) if pcol in S \
                else np.zeros(len(S), np.int64)
            proto[index < 0] = 0
            mod[m] = ModalityData(lev, feat.astype(np.float32), mask,
                                  ((stats - self.mod_mean[m]) / self.mod_std[m]).astype(np.float32), index, off,
                                  proto)
        x = S[list(self.cfg["context_cols"])].to_numpy(float)
        bad = ~np.isfinite(x)
        z = np.where(bad, 0.0, (x - self.ctx_mean) / self.ctx_std)
        src = S["source"].astype(str).map(self.source_map).fillna(0).to_numpy(np.int64)
        alt = S[ALTITUDE_COL].to_numpy(float) if ALTITUDE_COL in S else np.full(len(S), np.nan)
        out = MAData(ids.to_numpy(object), mod, np.concatenate([z, bad.astype(float)], 1).astype(np.float32), src,
                     lat, lon, alt)
        if len(out) and not out.present().all():
            raise ValueError(f"{int((~out.present()).sum())} rows hold none of the modalities {self.modalities}; "
                             f"filter with usable_ids first")
        return out


def inner_split(samples: pd.DataFrame, ids, frac: float = 0.15, block_km: float = 50.0, seed: int = 0):
    """(fit ids, inner-validation ids): whole 50-km blocks are held out, per country, until ``frac`` of the
    country's 1-km location clusters are in validation. A cluster never straddles the two sides."""
    from sentinel.eval.splits import spatial_blocks

    ids = pd.Index(ids)
    S = samples.loc[ids]
    cl = S["location_id"].astype(str).to_numpy()
    blocks = spatial_blocks(S["lat"].to_numpy(float), S["lon"].to_numpy(float), block_km)
    cb = pd.Series(blocks, index=cl).groupby(level=0, sort=False).first().reindex(cl).to_numpy()
    country = S["source"].astype(str).str[:2].to_numpy()
    rng = np.random.default_rng(seed)
    val = np.zeros(len(S), bool)
    for c in sorted(set(country)):
        m = country == c
        tab = pd.DataFrame({"cluster": cl[m], "block": cb[m]}).drop_duplicates("cluster")
        per = tab.groupby("block").size().sort_index()
        if len(per) < 2:                              # one block only: hold out clusters instead
            cs = np.sort(tab["cluster"].to_numpy())
            k = int(round(frac * len(cs)))
            pick = set(rng.permutation(cs)[:k]) if len(cs) > 1 else set()
            val |= m & np.isin(cl, list(pick))
            continue
        order = rng.permutation(per.index.to_numpy())
        target, acc, chosen = frac * per.sum(), 0, []
        for b in order[:-1]:                          # never every block
            if acc >= target:
                break
            chosen.append(b)
            acc += int(per[b])
        val |= m & np.isin(cb, chosen)
    return ids[~val], ids[val]


# ------------------------------------------------------------------------------------------------ losses
def student_t_nll(loc, log_scale, y, nu: float = 4.0):
    """Element-wise negative log-likelihood of a location-scale Student-t; (nll with 0 where y is missing, mask)."""
    torch, _ = BS._torch()
    ok = torch.isfinite(y)
    z = (torch.nan_to_num(y) - loc) * torch.exp(-log_scale)
    c = math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * math.log(nu * math.pi)
    nll = -(c - log_scale - 0.5 * (nu + 1.0) * torch.log1p(z * z / nu))
    return torch.where(ok, nll, torch.zeros_like(nll)), ok


def pooled_nll(nll, ok):
    return nll.sum() / ok.sum().clamp_min(1)


def column_balanced_nll(nll, ok, cols) -> float:
    """Mean over the columns ``cols`` (those with at least one finite target) of the column mean NLL."""
    n = ok[:, cols].sum(0)
    per = nll[:, cols].sum(0) / n.clamp_min(1)
    have = n > 0
    return float(per[have].mean()) if bool(have.any()) else float("nan")


def altitude_pressure_atm(alt_m):
    """Standard-atmosphere pressure (atm) at altitude ``alt_m`` metres: (1 − 2.25577e-5·h)^5.25588.

    A missing altitude gives 1 atm (no correction; the caller counts such rows). The altitude is clipped to
    [−100, 5000] m as in ``hydromet.physics.pressure_atm_from_elevation`` (unit-tested equal). numpy or torch."""
    if hasattr(alt_m, "nan_to_num"):
        h = alt_m.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 5000.0)
    else:
        h = np.asarray(alt_m, dtype="float64")
        h = np.where(np.isfinite(h), np.clip(h, -100.0, 5000.0), 0.0)
    return (1.0 - 2.25577e-5 * h) ** 5.25588


def o2_solubility_mg_l(temp_c, pressure_atm=1.0):
    """Oxygen solubility in fresh water (Benson & Krause 1984), mg/L, at ``pressure_atm``. numpy or torch.

    This is the shared routine ``sentinel.models.hydromet.physics.do_sat_mgl`` (called, not copied)."""
    from sentinel.models.hydromet.physics import do_sat_mgl

    return do_sat_mgl(temp_c, pressure_atm)


def default_constraints(columns, do_tol: float = 0.06) -> list[dict]:
    """Coherence constraints of protocol §9.1 for the target columns present, per label window."""
    col = {c: i for i, c in enumerate(columns)}
    out = []
    for w in sorted({c.split("__")[1] for c in columns if "__" in c}):
        def ix(t):
            return col.get(f"{t}__{w}")

        for hi, lo, sources in (("log10_tp_mg_l", "log10_po4p_mg_l", None),
                                ("log10_ton_mg_l", "log10_no3n_mg_l", ("uk", "ukd")),
                                ("log10_nkj_mg_l", "log10_nh4n_mg_l", ("fr",))):
            if ix(hi) is not None and ix(lo) is not None:
                out.append({"kind": "ge", "hi": ix(hi), "lo": ix(lo), "sources": sources, "window": w})
        if all(ix(t) is not None for t in ("log10_do_mg_l", "do_sat_pct", "temp_c")):
            out.append({"kind": "do", "do": ix("log10_do_mg_l"), "sat": ix("do_sat_pct"), "temp": ix("temp_c"),
                        "tol": float(do_tol), "sources": None, "window": w})
    return out


def coherence_penalty(P, constraints, src=None, source_map=None, alt=None):
    """Sum over constraints of the row-mean hinge violation of the predicted locations ``P`` (original units).

    ``alt``: station altitude per row in metres (NaN = missing -> 1 atm). It sets the pressure of the DO term."""
    torch, _ = BS._torch()
    tot = P.new_zeros(())
    for c in constraints:
        if c["kind"] == "ge":
            v = torch.relu(P[:, c["lo"]] - P[:, c["hi"]])
        else:
            temp = P[:, c["temp"]].clamp(0.0, 35.0)
            sat = P[:, c["sat"]].clamp(1.0, 300.0)
            press = 1.0 if alt is None else altitude_pressure_atm(alt.to(P.dtype))
            expect = torch.log10(sat / 100.0 * o2_solubility_mg_l(temp, press))
            v = torch.relu((P[:, c["do"]] - expect).abs() - c["tol"])
        if c.get("sources") is not None and src is not None and source_map is not None:
            ids = [source_map[s] for s in c["sources"] if s in source_map]
            sel = torch.isin(src, torch.as_tensor(ids, device=src.device, dtype=src.dtype)).to(v.dtype) if ids \
                else torch.zeros_like(v)
            tot = tot + (v * sel).sum() / sel.sum().clamp_min(1.0)
        else:
            tot = tot + v.mean()
    return tot


# ------------------------------------------------------------------------------------------------ model
def build_ma(cfg: dict, vocab_sizes: dict, n_ctx: int, n_src: int, n_proto: dict, n_out: int):
    torch, nn = BS._torch()
    d, drop, H, W = int(cfg["d_model"]), float(cfg["dropout"]), int(cfg["n_heads"]), int(cfg["trunk"])
    mods = tuple(m for m in MODALITIES if m in tuple(cfg["modalities"]))

    class TokenEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.name = nn.Embedding(vocab_sizes["name"], d, padding_idx=PAD)
            nn.init.normal_(self.name.weight, 0.0, 1.0)
            self.lineage = nn.ModuleDict({lv: nn.Embedding(vocab_sizes[lv], d, padding_idx=PAD)
                                          for lv in LEVELS[:-1]})
            for e in self.lineage.values():
                nn.init.normal_(e.weight, 0.0, float(cfg["lineage_std"]))
            with torch.no_grad():
                self.name.weight[PAD].zero_()
                for e in self.lineage.values():
                    e.weight[PAD].zero_()
            self.modality = nn.Parameter(torch.randn(len(MODALITIES), d) * 0.1)
            self.feat = nn.Linear(N_TOKEN_FEAT, d)

        def forward(self, lev, feat, m: int):
            h = self.name(lev[..., NAME])
            for j, lv in enumerate(LEVELS[:-1]):
                h = h + self.lineage[lv](lev[..., j])
            return h + self.feat(feat) + self.modality[m]

    class SAB(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.att = nn.MultiheadAttention(d, H, dropout=drop, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
            self.drop = nn.Dropout(drop)

        def forward(self, h, pad):
            x = self.ln1(h)
            a, _ = self.att(x, x, x, key_padding_mask=pad, need_weights=False)
            h = h + self.drop(a)
            return h + self.drop(self.ff(self.ln2(h)))

    class SetEncoder(nn.Module):
        """Shared across modalities: token embedding, SAB blocks, PMA(1) ⊕ abundance-weighted mean -> [n, 2d]."""

        def __init__(self):
            super().__init__()
            self.emb = TokenEmbedding()
            self.blocks = nn.ModuleList([SAB() for _ in range(int(cfg["n_layers"]))])
            self.ln = nn.LayerNorm(d)
            self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.pma = nn.MultiheadAttention(d, H, dropout=drop, batch_first=True)

        def forward(self, lev, feat, mask, m: int):
            h = self.emb(lev, feat, m)
            pad = ~mask
            for b in self.blocks:
                h = b(h, pad)
            h = self.ln(h)
            w = feat[..., 1] * mask                               # relative abundance
            w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
            wmean = (h.float() * w.unsqueeze(-1)).sum(1)
            q = self.seed.expand(h.shape[0], -1, -1)
            p, _ = self.pma(q, h, h, key_padding_mask=pad, need_weights=False)
            return torch.cat([wmean, p[:, 0].float()], -1)

    class RegionContext(nn.Module):
        """Region token of row i: attention pooling over the community vectors of the other rows of its group."""

        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(d)
            self.seed = nn.Parameter(torch.randn(d) * 0.02)
            self.att = nn.MultiheadAttention(d, H, dropout=drop, batch_first=True)
            self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

        def allowed(self, group, key_ok=None):
            B = group.shape[0]
            same = (group[:, None] == group[None, :]) & (group[:, None] >= 0)
            same = same & ~torch.eye(B, dtype=torch.bool, device=group.device)
            if key_ok is not None:
                same = same & key_ok[None, :]
            return same

        def forward(self, h, group, key_ok=None):
            same = self.allowed(group, key_ok)
            has = same.any(1)
            block = ~same
            block[~has] = False                                   # rows without context: output discarded below
            x = self.ln(h)
            q = (x + self.seed).unsqueeze(0)
            kv = x.unsqueeze(0)
            out, _ = self.att(q, kv, kv, attn_mask=block, need_weights=False)
            out = self.ff(out[0])
            return out * has.unsqueeze(-1).to(out.dtype), has

    class BioSetMA(nn.Module):
        def __init__(self):
            super().__init__()
            self.mods = mods
            self.encoder = SetEncoder()
            self.mod_proj = nn.ModuleDict({m: nn.Linear(2 * d + N_MOD_FEAT, d) for m in mods})
            self.proto = nn.ModuleDict({m: nn.Embedding(int(n_proto.get(m, 0)) + 1, d) for m in mods})
            self.ctx = nn.Linear(n_ctx, d)
            self.src = nn.Embedding(n_src + 1, d)
            for e in [self.src, *self.proto.values()]:
                nn.init.normal_(e.weight, 0.0, 0.1)
            self.slot = nn.Parameter(torch.randn(N_SLOTS, d) * 0.1)
            self.region = RegionContext() if cfg.get("region_context") else None
            self.fusion = SAB()
            self.ln = nn.LayerNorm(d)
            self.trunk = nn.Sequential(nn.Linear(d, W), nn.GELU(), nn.Dropout(drop), nn.Linear(W, W), nn.GELU(),
                                       nn.Dropout(drop))
            self.loc = nn.Linear(W, n_out)
            self.log_scale = nn.Linear(W, n_out)
            self.src_scale = nn.Embedding(n_src + 1, n_out)
            nn.init.zeros_(self.src_scale.weight)
            nn.init.zeros_(self.log_scale.weight)
            nn.init.zeros_(self.log_scale.bias)

        def community(self, b):
            """Per-modality vectors [B, 2, d] and their validity [B, 2] (zeros where the modality is absent)."""
            B = b["ctx"].shape[0]
            dev = b["ctx"].device
            vec, ok = [], []
            for j, m in enumerate(MODALITIES):
                v = torch.zeros(B, d, device=dev)
                o = torch.zeros(B, dtype=torch.bool, device=dev)
                if m in self.mods and m in b["mod"]:
                    lev, feat, mask, rows, mfeat = b["mod"][m]
                    e = self.encoder(lev, feat, mask, j)
                    e = self.mod_proj[m](torch.cat([e, mfeat], -1)) + self.proto[m](b["proto"][m][rows])
                    v = v.index_copy(0, rows, e.float())
                    o = o.index_fill(0, rows, True)
                vec.append(v)
                ok.append(o)
            return torch.stack(vec, 1), torch.stack(ok, 1)

        def forward(self, b):
            vec, ok = self.community(b)
            B = vec.shape[0]
            ctx = (self.ctx(b["ctx"]) + self.src(b["src"])).float()
            okf = ok.float()
            if self.region is not None and b.get("group") is not None:
                base = (vec * okf.unsqueeze(-1)).sum(1) / okf.sum(1, keepdim=True).clamp_min(1.0)
                r, has = self.region(base, b["group"], b.get("key_ok"))
                r = r.float()
            else:
                r = torch.zeros_like(ctx)
                has = torch.zeros(B, dtype=torch.bool, device=ctx.device)
            X = torch.cat([vec, ctx.unsqueeze(1), r.unsqueeze(1)], 1) + self.slot
            V = torch.cat([ok, torch.ones(B, 1, dtype=torch.bool, device=ctx.device), has.unsqueeze(1)], 1)
            h = self.ln(self.fusion(X, ~V)).float()
            Vf = V.float().unsqueeze(-1)
            z = (h * Vf).sum(1) / Vf.sum(1)
            t = self.trunk(z)
            return self.loc(t), self.log_scale(t) + self.src_scale(b["src"]), t

    return BioSetMA()


# ------------------------------------------------------------------------------------------------ batches
def make_batch(data: MAData, rows, device, *, cfg: dict | None = None, rng=None, group=None, key_ok=None) -> dict:
    """Tensors of the rows ``rows``. With ``rng`` (training): token dropout (>= 1 token kept), modality dropout
    (one modality always kept) and source dropout (source, protocol and pool prevalence set to unknown)."""
    torch, _ = BS._torch()
    rows = np.asarray(rows, np.int64)
    n = len(rows)
    train = rng is not None
    src = data.src[rows].copy()
    sdrop = (rng.random(n) < float(cfg["source_dropout"])) if train else np.zeros(n, bool)
    src[sdrop] = 0
    dev = data.on(device)
    present = {m: data.mod[m].index[rows] >= 0 for m in data.mod}
    if train and len(data.mod) == 2 and float(cfg["modality_dropout"]) > 0:
        a, b = MODALITIES
        hit = present[a] & present[b] & (rng.random(n) < float(cfg["modality_dropout"]))
        which = rng.random(n) < 0.5
        present[a] = present[a] & ~(hit & which)
        present[b] = present[b] & ~(hit & ~which)
    mod, proto = {}, {}
    for m, md in data.mod.items():
        p = md.proto[rows].copy()
        p[sdrop] = 0
        proto[m] = torch.as_tensor(p, device=device)
        loc = np.flatnonzero(present[m])
        if len(loc) == 0:
            continue
        u = md.index[rows[loc]]
        mask = md.mask[u]
        T = max(1, int(mask.sum(1).max()))                # tokens are packed to the left, most abundant first
        mask = mask[:, :T].copy()
        ut = torch.as_tensor(u, device=device, dtype=torch.long)
        lev, feat = dev[m][0][ut, :T], dev[m][1][ut, :T]       # gathered on the device; new tensors
        if train:
            td = float(cfg["token_dropout"])
            if td > 0:
                drop = (rng.random(mask.shape) < td) & mask
                drop &= (mask & ~drop).any(1, keepdims=True)
                mask &= ~drop
            sd = sdrop[loc]
            if sd.any():
                sdt = torch.as_tensor(np.flatnonzero(sd), device=device, dtype=torch.long)
                feat[sdt, :, I_PREV] = 0.0
                feat[sdt, :, I_MISS] = torch.as_tensor(md.mask[u, :T][sd].astype(np.float32), device=device)
        mfeat = np.column_stack([md.mfeat[u], md.offset[rows[loc]]]).astype(np.float32)
        mod[m] = (lev.long(), feat, torch.as_tensor(mask, device=device), torch.as_tensor(loc, device=device),
                  torch.as_tensor(mfeat, device=device))
    alt = None if data.alt is None else torch.as_tensor(data.alt[rows].astype(np.float32), device=device)
    out = {"mod": mod, "proto": proto, "alt": alt,
           "ctx": dev["ctx"][torch.as_tensor(rows, device=device, dtype=torch.long)],
           "src": torch.as_tensor(src, device=device),
           "src_true": torch.as_tensor(data.src[rows], device=device), "group": None, "key_ok": None}
    if group is not None:
        out["group"] = torch.as_tensor(np.asarray(group, np.int64), device=device)
    if key_ok is not None:
        out["key_ok"] = torch.as_tensor(np.asarray(key_ok, bool), device=device)
    return out


def epoch_batches(n: int, batch_size: int, rng, groups=None, episode_rows: int = 64):
    """Yield (rows, group). Without ``groups``: shuffled batches. With ``groups`` (pseudo-region codes): episodic
    batches made of chunks of at most ``episode_rows`` rows of one pseudo-region; the chunk is the context group."""
    if groups is None:
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            yield perm[s:s + batch_size], None
        return
    order = rng.permutation(n)
    o = order[np.argsort(groups[order], kind="stable")]
    g = groups[o]
    start = np.r_[True, g[1:] != g[:-1]]
    pos = np.arange(n) - np.maximum.accumulate(np.where(start, np.arange(n), 0))
    chunk = np.cumsum(start | (pos % episode_rows == 0)) - 1
    bounds = np.flatnonzero(np.r_[True, chunk[1:] != chunk[:-1]])
    chunks = np.split(o, bounds[1:])
    rows, grp, k = [], [], 0
    for c in rng.permutation(len(chunks)):
        ch = chunks[c]
        if rows and sum(map(len, rows)) + len(ch) > batch_size:
            yield np.concatenate(rows), np.concatenate(grp)
            rows, grp, k = [], [], 0
        rows.append(ch)
        grp.append(np.full(len(ch), k, np.int64))
        k += 1
    if rows:
        yield np.concatenate(rows), np.concatenate(grp)


def _autocast(device, cfg):
    torch, _ = BS._torch()
    on = bool(cfg.get("amp", True)) and str(device).startswith("cuda")
    return torch.autocast(device_type="cuda" if on else "cpu", dtype=torch.float16 if on else torch.bfloat16,
                          enabled=on)


def run_model(model, data: MAData, cfg: dict, device="cpu", region=None, seed: int = 0, batch: int = 256,
              region_index: dict | None = None):
    """(loc, log-scale, trunk output) of every row in standardised target units (eval mode, no gradient).

    ``region``: one label per row (None / NaN = no region context). The region token of a row is pooled from at
    most ``cfg['region_max']`` other rows with the same label: all of them when the region is that small, else
    the one sample of the mirror rule (protocol §9.1; ``ftrack.mirror_pick``): rows sorted by key in byte order,
    drawn with ``default_rng([20261110, k, seed])``. ``region_index`` maps a region label to its index ``k`` in
    the split file (scored rows); without it ``k`` is the position of the label among the sorted labels (inner
    validation grid). The caller passes exactly the candidate rows of the rule in ``data``."""
    from sentinel.models.bioassess.ftrack import mirror_pick

    torch, _ = BS._torch()
    model.eval()
    N = len(data)
    out = [None, None, None]

    def put(rows, res, keep):
        for i, a in enumerate(res):
            a = a.float().cpu().numpy()[:keep]
            if out[i] is None:
                out[i] = np.zeros((N, a.shape[1]), np.float32)
            out[i][rows[:keep]] = a

    plain = np.arange(N)
    use_region = getattr(model, "region", None) is not None and region is not None
    with torch.no_grad():
        if use_region:
            lab = pd.Series(np.asarray(region, object))
            has = lab.notna().to_numpy()
            plain = np.flatnonzero(~has)
            cap = int(cfg["region_max"])
            names = lab[has].astype(str).to_numpy()
            pos = np.flatnonzero(has)
            for k, name in enumerate(sorted(set(names))):
                R = pos[names == name]
                if len(R) <= cap:
                    with _autocast(device, cfg):
                        res = model(make_batch(data, R, device, group=np.zeros(len(R), np.int64)))
                    put(R, res, len(R))
                    continue
                kk = k if region_index is None else int(region_index[name])
                C = np.sort(R[mirror_pick(data.ids[R], kk, seed, cap)])
                with _autocast(device, cfg):
                    res = model(make_batch(data, C, device, group=np.zeros(cap, np.int64)))
                put(C, res, cap)
                Q = np.setdiff1d(R, C)
                for s in range(0, len(Q), batch):
                    q = Q[s:s + batch]
                    rows = np.concatenate([q, C])
                    with _autocast(device, cfg):
                        res = model(make_batch(data, rows, device, group=np.zeros(len(rows), np.int64),
                                               key_ok=np.r_[np.zeros(len(q), bool), np.ones(cap, bool)]))
                    put(rows, res, len(q))
        for s in range(0, len(plain), batch):
            rows = plain[s:s + batch]
            with _autocast(device, cfg):
                res = model(make_batch(data, rows, device))
            put(rows, res, len(rows))
    if out[0] is None:
        return (np.zeros((0, 0), np.float32),) * 3
    return np.asarray(out[0]), np.clip(out[1], *LOG_SCALE_RANGE), np.asarray(out[2])


# ------------------------------------------------------------------------------------------------ fit / predict
@dataclass
class FittedMA:
    model: object
    cfg: dict
    columns: list
    y_mean: np.ndarray
    y_std: np.ndarray
    history: list
    best_epoch: int
    n_params: int
    pca: dict | None = None
    info: dict = field(default_factory=dict)

    def predict(self, data: MAData, region=None, device="cpu", seed: int = 0, region_index=None) -> dict:
        """Predictive Student-t location and scale per target, in the units of the target columns."""
        loc, ls, _ = run_model(self.model, data, self.cfg, device, region, seed, region_index=region_index)
        return {"loc": loc * self.y_std + self.y_mean, "scale": np.exp(ls) * self.y_std, "nu": float(self.cfg["nu"])}

    def embed(self, data: MAData, region=None, device="cpu", seed: int = 0, region_index=None) -> np.ndarray:
        """Trunk output [N, trunk]."""
        return run_model(self.model, data, self.cfg, device, region, seed, region_index=region_index)[2]

    def fit_embedding(self, train: MAData, region=None, device="cpu", seed: int = 0) -> None:
        """PCA of the trunk output on training rows -> the low-dimensional sample embedding of the hybrid."""
        from sklearn.decomposition import PCA

        Z = self.embed(train, region, device, seed).astype(np.float64)
        k = int(min(self.cfg["embed_dim"], Z.shape[1], max(1, Z.shape[0] - 1)))
        p = PCA(n_components=k, svd_solver="full").fit(Z)
        comp = p.components_ * np.where(p.components_[np.arange(k), np.abs(p.components_).argmax(1)] < 0, -1.0,
                                        1.0)[:, None]                        # fixed sign convention
        self.pca = {"mean": p.mean_, "components": comp}

    def embed_low(self, data: MAData, region=None, device="cpu", seed: int = 0) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("fit_embedding(train) first")
        Z = self.embed(data, region, device, seed).astype(np.float64)
        return ((Z - self.pca["mean"]) @ self.pca["components"].T).astype(np.float32)


FIT_STATE_VERSION = 1


def _fit_signature(cfg, columns, primary, seed, n_train, n_val) -> dict:
    return {"cfg": repr(sorted((k, repr(v)) for k, v in cfg.items())), "columns": list(columns),
            "primary": list(primary), "seed": int(seed), "n_train": int(n_train), "n_val": int(n_val)}


def fit_ma_chunk(cfg: dict, feat: MAFeaturizer, tr: MAData, Ytr: np.ndarray, va: MAData, Yva: np.ndarray, *, columns,
                 primary, seed: int, device: str = "cpu", constraints: list | None = None, log=None,
                 state: dict | None = None, chunk_epochs: int | None = None):
    """One chunk of a resumable fit (lead ruling (o)). Returns ``(fitted, state)``.

    Runs at most ``chunk_epochs`` epochs (None = to the end) from ``state`` (None = a new fit). ``fitted`` is the
    :class:`FittedMA` once the fit has ended (early stop or ``max_epochs``), else None. ``state`` holds everything
    needed to go on: weights, optimiser and gradient-scaler states, the numpy and torch generator states and the
    early-stopping record. It is saved with :func:`save_fit_state` and read back only through the guarded loader
    (:func:`load_fit_state`). A resumed fit must get the same data, columns, recipe and seed (checked).

    Train on (tr, Ytr); early-stop on the column-balanced NLL of the ``primary`` columns of (va, Yva).
    Standardisers come from ``Ytr`` only. ``tr`` / ``va`` must come from ``feat`` fitted on training ids.
    On one device type a chunked fit follows the same generator streams as an uninterrupted one (tested equal on
    CPU); bit equality on a GPU is not guaranteed and is disclosed in the protocol."""
    torch, _ = BS._torch()
    from sentinel.core.seed import seed_everything

    columns = list(columns)
    Ytr, Yva = np.asarray(Ytr, float), np.asarray(Yva, float)
    if Ytr.shape != (len(tr), len(columns)) or Yva.shape[1:] != (len(columns),) or len(Yva) != len(va):
        raise ValueError("target arrays do not match the data / columns")
    if chunk_epochs is not None and int(chunk_epochs) < 1:
        raise ValueError("chunk_epochs must be >= 1")
    sig = _fit_signature(cfg, columns, primary, seed, len(tr), len(va))
    if state is not None:
        if state.get("version") != FIT_STATE_VERSION or state.get("signature") != sig:
            raise ValueError("fit state was written for another recipe, seed, column list or data size")
        if state.get("done"):
            raise ValueError("this fit has already ended; nothing to resume")
    ym, ys = nan_stats(Ytr)
    pcols = [columns.index(c) for c in primary if c in columns]
    if not pcols:
        raise ValueError("no primary column among the targets")
    lam = float(cfg.get("coherence_weight", 0.0))
    if lam > 0 and constraints is None:
        constraints = default_constraints(columns, float(cfg["do_tol"]))
    info = {}
    if lam > 0 and constraints and any(c["kind"] == "do" for c in constraints):
        miss = len(tr) if tr.alt is None else int((~np.isfinite(tr.alt)).sum())
        info["do_coherence"] = {"n_train_rows": len(tr), "n_altitude_missing": miss,
                                "note": "rows with a missing altitude use 1 atm (no pressure correction)"}
        if log:
            log(f"DO coherence: {miss} of {len(tr)} training rows have no altitude (1 atm used)")
    sizes = {lv: feat.vocab.size(lv) for lv in LEVELS}
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = build_ma(cfg, sizes, feat.n_ctx, len(feat.source_map), {m: len(v) for m, v in feat.proto_map.items()},
                     len(columns)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    amp = bool(cfg.get("amp", True)) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, best_ep, bad, hist, start, chunks = math.inf, None, -1, 0, [], 0, []
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        rng.bit_generator.state = state["np_rng"]
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        best, best_state, best_ep, bad = state["best"], state["best_state"], state["best_epoch"], state["bad"]
        hist, start, chunks = list(state["history"]), int(state["epoch"]), list(state["chunks"])
    ytr = torch.as_tensor(((Ytr - ym) / ys).astype(np.float32))
    yva = torch.as_tensor(((Yva - ym) / ys).astype(np.float32))
    ym_t = torch.as_tensor(ym.astype(np.float32), device=device)
    ys_t = torch.as_tensor(ys.astype(np.float32), device=device)
    nu = float(cfg["nu"])
    rc = bool(cfg.get("region_context"))
    val_region = cell_codes(va.lat, va.lon, float(cfg["val_region_km"])) if (rc and len(va)) else None
    n, bs, max_ep = len(tr), int(cfg["batch_size"]), int(cfg["max_epochs"])
    done, ep = start >= max_ep, start - 1
    for ep in range(start, max_ep):
        model.train()
        groups = pseudo_regions(tr.lat, tr.lon, rng, cfg["pseudo_region_km"]) if rc else None
        tot, cnt = 0.0, 0
        for rows, grp in epoch_batches(n, bs, rng, groups, int(cfg["episode_rows"])):
            b = make_batch(tr, rows, device, cfg=cfg, rng=rng, group=grp)
            with _autocast(device, cfg):
                loc, ls, _ = model(b)
            loc, ls = loc.float(), ls.float().clamp(*LOG_SCALE_RANGE)
            nll, ok = student_t_nll(loc, ls, ytr[rows].to(device), nu)
            loss = pooled_nll(nll, ok)
            if lam > 0 and constraints:
                loss = loss + lam * coherence_penalty(loc * ys_t + ym_t, constraints, b["src_true"], feat.source_map,
                                                      b["alt"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()) * len(rows)
            cnt += len(rows)
        if len(va):
            vloc, vls, _ = run_model(model, va, cfg, device, val_region, seed)
            vn, vok = student_t_nll(torch.as_tensor(vloc), torch.as_tensor(vls), yva, nu)
            vl = column_balanced_nll(vn, vok, pcols)
        else:
            vl = tot / max(cnt, 1)
        hist.append((ep, tot / max(cnt, 1), vl))
        if math.isfinite(vl) and vl < best - 1e-4:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best, best_ep, bad = vl, ep, 0
        else:
            bad += 1
        if bad >= int(cfg["patience"]) or ep + 1 >= max_ep:
            done = True
            break
        if chunk_epochs is not None and ep + 1 - start >= int(chunk_epochs):
            break
    chunks.append({"start_epoch": start, "end_epoch": ep + 1})
    state = {"version": FIT_STATE_VERSION, "signature": sig, "epoch": ep + 1, "done": bool(done),
             "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
             "opt": opt.state_dict(), "scaler": scaler.state_dict(), "np_rng": rng.bit_generator.state,
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None,
             "best": best, "best_state": best_state, "best_epoch": best_ep, "bad": bad, "history": list(hist),
             "chunks": chunks}
    if not done:
        if log:
            log(f"chunk ended after epoch {ep} (resume point {len(chunks)}); best so far epoch {best_ep}")
        return None, state
    if best_state is None:
        raise FloatingPointError("inner-validation NLL never finite (no finite primary target in validation?)")
    model.load_state_dict(best_state)
    n_params = int(sum(p.numel() for p in model.parameters()))
    if log:
        log(f"best epoch {best_ep} inner-val NLL {best:.4f} ({len(hist)} epochs, {n_params} params, "
            f"{len(chunks)} chunk(s))")
    info["chunks"] = chunks
    return FittedMA(model, copy.deepcopy(cfg), columns, ym, ys, hist, best_ep, n_params, None, info), state


def fit_ma(cfg: dict, feat: MAFeaturizer, tr: MAData, Ytr: np.ndarray, va: MAData, Yva: np.ndarray, *, columns,
           primary, seed: int, device: str = "cpu", constraints: list | None = None, log=None) -> FittedMA:
    """Uninterrupted fit: :func:`fit_ma_chunk` run to the end in one call."""
    return fit_ma_chunk(cfg, feat, tr, Ytr, va, Yva, columns=columns, primary=primary, seed=seed, device=device,
                        constraints=constraints, log=log)[0]


def save_fit_state(path, state: dict, train_ids, *, track: str, g, design: str):
    """Save a resume point with the M5-F training-id record (the guarded loader refuses it otherwise)."""
    torch, _ = BS._torch()
    from sentinel.models.bioassess import ftrack as F

    torch.save({k: v for k, v in state.items() if k != "f_guard"}, path)
    return F.write_train_ids_sidecar(path, train_ids, track=track, g=g, design=design)


def load_fit_state(path, S: pd.DataFrame, track: str, g, design: str) -> dict:
    """A resume point, through ``ftrack.load_checkpoint`` only."""
    from sentinel.models.bioassess import ftrack as F

    obj = F.load_checkpoint(path, S, track, g, design)
    F.require_guarded(obj, track, g, design)
    return {k: v for k, v in obj.items() if k != "f_guard"}


# ------------------------------------------------------------------------------------------------ checkpoints
def save_checkpoint(path, fitted: FittedMA, feat: MAFeaturizer, train_ids, *, track: str, g, design: str):
    """Save model + featurizer and the M5-F training-id record that the guarded loader requires."""
    torch, _ = BS._torch()
    from sentinel.models.bioassess import ftrack as F

    state = {k: v.detach().cpu() for k, v in fitted.model.state_dict().items()}
    torch.save({"state": state, "cfg": fitted.cfg, "columns": fitted.columns, "y_mean": fitted.y_mean,
                "y_std": fitted.y_std, "history": fitted.history, "best_epoch": fitted.best_epoch,
                "n_params": fitted.n_params, "pca": fitted.pca, "info": fitted.info, "featurizer": feat}, path)
    return F.write_train_ids_sidecar(path, train_ids, track=track, g=g, design=design)


def load_fitted(path, S: pd.DataFrame, track: str, g, design: str, device: str = "cpu"):
    """(FittedMA, MAFeaturizer) through ``ftrack.load_checkpoint`` (refuses checkpoints without a valid record)."""
    from sentinel.models.bioassess import ftrack as F

    obj = F.load_checkpoint(path, S, track, g, design)
    F.require_guarded(obj, track, g, design)
    feat, cfg = obj["featurizer"], obj["cfg"]
    model = build_ma(cfg, {lv: feat.vocab.size(lv) for lv in LEVELS}, feat.n_ctx, len(feat.source_map),
                     {m: len(v) for m, v in feat.proto_map.items()}, len(obj["columns"]))
    model.load_state_dict(obj["state"])
    fitted = FittedMA(model.to(device), cfg, obj["columns"], obj["y_mean"], obj["y_std"], obj["history"],
                      obj["best_epoch"], obj["n_params"], obj["pca"], obj.get("info", {}))
    return fitted, feat


__all__ = ["MODALITIES", "DEFAULTS", "RECIPES", "recipe", "PoolPrevalence", "MAFeaturizer", "MAData", "ModalityData",
           "usable_ids", "inner_split", "grid_cells", "cell_codes", "pseudo_regions", "student_t_nll", "pooled_nll",
           "column_balanced_nll", "o2_solubility_mg_l", "default_constraints", "coherence_penalty", "build_ma",
           "make_batch", "epoch_batches", "run_model", "FittedMA", "fit_ma", "fit_ma_chunk", "save_fit_state",
           "load_fit_state", "altitude_pressure_atm", "save_checkpoint", "load_fitted"]
