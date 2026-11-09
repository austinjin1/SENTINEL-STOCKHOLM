# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Chemprop v2 D-MPNN for SENTINEL-ChemTox, with taxonomy embedding and conditioned head.

Pinned to ``chemprop==2.3.1`` (installed with ``rdkit==2025.9.3`` so the shared environment's
RDKit is untouched). Chemprop v2 needs neither PyG nor DGL, so it avoids the broken
``torch-cluster`` in this environment (docs/research/04 §E.2).

Architecture (RESEARCH_PLAN §5.4):

* a bond-message D-MPNN encoder over the molecular graph built from ``chem_rdkit_can_smiles``;
* **non-structural features as extra FFN inputs** — the published 37-column
  experimental/chemical/taxonomic block is passed as chemprop's datapoint-level ``x_d`` vector and
  concatenated to the pooled graph encoding;
* a **species/taxonomy embedding** — a learned vector per ``tax_gs`` (with an out-of-vocabulary
  slot for species unseen in the training folds) concatenated to the same representation;
* a **duration/endpoint-conditioned head** — a FiLM layer whose scale and shift are produced from
  the scaled observation duration and the one-hot endpoint/effect/concentration-type vector, so the
  head is conditioned on the experiment rather than only informed of it (TRIDENT's design borrow);
* optionally, **CheMeleon foundation initialisation** of the message-passing block
  (the API equivalent of the CLI's ``--from-foundation CheMeleon``: the released
  ``chemeleon_mp.pt`` from Zenodo record 15460715 — cite DOI 10.48550/arXiv.2506.15792 — loaded
  into a ``BondMessagePassing`` block with mean aggregation, which requires the V2 multi-hot atom
  featuriser).

The conditioning inputs ride inside the ``x_d`` vector: columns ``[0, n_desc)`` are the descriptor
block, column ``n_desc`` is the species index, and the remaining columns are the conditioning
vector. :class:`ChemToxMPNN` splits them again inside ``fingerprint``.

Phase 1 scope: this module is implemented and smoke-tested only — no tuning, no test scoring.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from torch import nn

from sentinel.core import paths

CHEMPROP_PIN = "chemprop==2.3.1"
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHEMELEON_CITATION = "CheMeleon foundation model, DOI 10.48550/arXiv.2506.15792"
SMILES_COL = "chem_rdkit_can_smiles"
SPECIES_COL = "tax_gs"
DURATION_COL = "result_obs_duration_mean"
CONDITION_CATEGORICAL = ("result_endpoint", "result_effect", "result_conc1_type")
TAXON_GROUP_COL = "tax_group"
#: multitask output order; ADORE writes these three codes in ``tax_group``
TASK_GROUPS = ("fish", "crusta", "algae")


# ----------------------------------------------------------------------------- conditioning inputs


@dataclass
class ConditionEncoder:
    """Species vocabulary + duration/endpoint conditioning vector, fit on training rows only."""

    species_: dict[str, int] = field(default_factory=dict)
    levels_: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, rows: pd.DataFrame) -> ConditionEncoder:
        sp = rows[SPECIES_COL].astype("string").fillna("<NA>")
        self.species_ = {s: i for i, s in enumerate(sorted(sp.unique()))}  # index len() = OOV slot
        self.levels_ = {
            c: sorted(rows[c].astype("string").fillna("<NA>").unique()) for c in CONDITION_CATEGORICAL
            if c in rows.columns
        }
        return self

    @property
    def n_species(self) -> int:
        return len(self.species_) + 1  # + out-of-vocabulary

    @property
    def n_condition(self) -> int:
        return 1 + sum(len(v) for v in self.levels_.values())

    def species_index(self, rows: pd.DataFrame) -> np.ndarray:
        sp = rows[SPECIES_COL].astype("string").fillna("<NA>")
        oov = len(self.species_)
        return np.array([self.species_.get(s, oov) for s in sp], dtype=np.float32)

    def condition(self, rows: pd.DataFrame) -> np.ndarray:
        d = pd.to_numeric(rows[DURATION_COL], errors="coerce").to_numpy(dtype=float)
        blocks = [((np.nan_to_num(d, nan=48.0) - 24.0) / 72.0).astype(np.float32).reshape(-1, 1)]
        for c, lv in self.levels_.items():
            v = rows[c].astype("string").fillna("<NA>").to_numpy()
            blocks.append(np.column_stack([(v == x).astype(np.float32) for x in lv]))
        return np.column_stack(blocks).astype(np.float32)


# ----------------------------------------------------------------------------- the model


def _chemprop():
    from chemprop import data, featurizers, models
    from chemprop import nn as cnn

    return data, featurizers, models, cnn


def load_chemeleon_message_passing(cache_dir: Path | None = None):
    """CheMeleon-initialised ``BondMessagePassing`` block (downloads the Zenodo checkpoint once)."""
    _, _, _, cnn = _chemprop()
    cache = Path(cache_dir) if cache_dir else paths.ensure(paths.RUNS_ROOT / "chemtox" / "pretrained")
    path = cache / "chemeleon_mp.pt"
    if not path.exists():
        urlretrieve(CHEMELEON_URL, path)
    ckpt = torch.load(path, weights_only=True)
    block = cnn.BondMessagePassing(**ckpt["hyper_parameters"])
    block.load_state_dict(ckpt["state_dict"])
    return block


def _pretrain_x_d(rows: pd.DataFrame) -> np.ndarray:
    """Conditioning vector for the pretraining task: duration + endpoint + taxon group."""
    d = pd.to_numeric(rows["duration_h"], errors="coerce").to_numpy(dtype=float)
    blocks = [((np.nan_to_num(d, nan=48.0) - 24.0) / 72.0).astype(np.float32).reshape(-1, 1)]
    ep = rows["endpoint"].astype("string").fillna("<NA>").to_numpy()
    blocks.append(np.column_stack([(ep == v).astype(np.float32) for v in ("LC50", "EC50")]))
    tg = rows["taxon_group"].astype("string").str.lower().fillna("<NA>").to_numpy()
    blocks.append(np.column_stack([(tg == v).astype(np.float32)
                                   for v in ("fish", "crustaceans", "algae")]))
    return np.column_stack(blocks).astype(np.float32)


def pretrain_ecotox_encoder(
    rows: pd.DataFrame,
    *,
    message_hidden: int = 300,
    depth: int = 3,
    dropout: float = 0.0,
    epochs: int = 10,
    batch_size: int = 128,
    accelerator: str = "cpu",
    seed: int = 0,
    exclude_cas=(),
    exclude_inchikey=(),
    cache_dir: Path | None = None,
) -> dict:
    """Pretrain a message-passing encoder on extra ECOTOX rows and return its state dict.

    ``rows`` must already have the evaluated challenge's **test** chemicals removed
    (:func:`sentinel.models.chemtox.aux.ecotox_auxiliary` does that and asserts it);
    ``exclude_cas`` / ``exclude_inchikey`` additionally remove the current CV fold's *validation*
    chemicals, so a fold's own molecules cannot reach the encoder either. The result is cached on
    disk under a key covering the architecture, the pretraining settings and the excluded fold
    chemicals, so the five per-fold encoders are computed once and reused by every configuration
    that shares them.
    """
    from . import aux as AUX

    data, _, _, cnn = _chemprop()
    import lightning.pytorch as pl
    from rdkit import Chem

    rows = AUX.drop_chemicals(rows, exclude_cas, exclude_inchikey, cas_col="cas",
                              inchikey_col="chem_inchikey")
    key_src = json.dumps({
        "n_rows": len(rows),
        "cas_sha": hashlib.sha256("|".join(sorted(map(str, set(exclude_cas)))).encode()).hexdigest()[:16],
        "mh": message_hidden, "depth": depth, "dropout": dropout, "epochs": epochs,
        "batch_size": batch_size, "seed": seed,
    }, sort_keys=True)
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    cache = Path(cache_dir) if cache_dir else paths.ensure(paths.RUNS_ROOT / "chemtox" / "pretrained")
    path = cache / f"ecotox_mp_{key}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)

    torch.manual_seed(seed)
    x_d = _pretrain_x_d(rows)
    y = pd.to_numeric(rows["y"], errors="coerce").to_numpy(dtype=float)
    mu, sd = float(np.nanmean(y)), float(np.nanstd(y) or 1.0)
    smiles = rows["smiles"].astype("string").fillna("").to_list()
    pts = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not np.isfinite(y[i]):
            continue
        pts.append(data.MoleculeDatapoint(mol=mol, y=np.array([(y[i] - mu) / sd]), x_d=x_d[i]))
    if not pts:
        raise ValueError("no usable ECOTOX pretraining rows after filtering")
    _, featurizers, models, _ = _chemprop()
    fz = featurizers.SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    dset = data.MoleculeDataset(pts, featurizer=fz)
    dset.cache = True
    loader = data.build_dataloader(dset, batch_size=batch_size, num_workers=0, seed=seed, shuffle=True)
    mp = cnn.BondMessagePassing(d_v=fz.atom_fdim, d_e=fz.bond_fdim, d_h=message_hidden, depth=depth,
                                dropout=dropout)
    agg = cnn.MeanAggregation()
    predictor = cnn.RegressionFFN(input_dim=mp.output_dim + x_d.shape[1], hidden_dim=300, n_layers=2)
    model = models.MPNN(mp, agg, predictor, batch_norm=True)
    trainer = pl.Trainer(max_epochs=int(epochs), accelerator=accelerator, devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.fit(model, loader)
    state = {k: v.detach().cpu() for k, v in model.message_passing.state_dict().items()}
    torch.save(state, path)
    return state


def build_model(
    *,
    n_descriptors: int,
    n_species: int,
    n_condition: int,
    d_v: int = 72,
    d_e: int = 14,
    species_dim: int = 16,
    message_hidden: int = 300,
    depth: int = 3,
    ffn_hidden: int = 300,
    ffn_layers: int = 2,
    dropout: float = 0.0,
    from_foundation: str | None = None,
    batch_norm: bool = True,
    max_lr: float = 1e-3,
    use_conditioning: bool = True,
    n_tasks: int = 1,
    encoder_state: dict | None = None,
):
    """Assemble :class:`ChemToxMPNN`.

    ``from_foundation="CheMeleon"`` starts from the released foundation encoder;
    ``encoder_state`` instead loads a message-passing state dict we pretrained ourselves (the
    ECOTOX route); ``species_dim=0`` removes the taxonomy embedding and ``use_conditioning=False``
    removes the FiLM head — both are ablations the improvement cycle needs.
    """
    _, _, _models, cnn = _chemprop()
    if from_foundation and from_foundation.lower() == "chemeleon":
        mp = load_chemeleon_message_passing()
        agg = cnn.MeanAggregation()
    elif from_foundation:
        raise ValueError(f"unknown foundation model {from_foundation!r} (supported: CheMeleon)")
    else:
        mp = cnn.BondMessagePassing(d_v=d_v, d_e=d_e, d_h=message_hidden, depth=depth,
                                    dropout=dropout)
        agg = cnn.MeanAggregation()
        if encoder_state is not None:
            mp.load_state_dict(encoder_state)
    d_in = mp.output_dim + n_descriptors + species_dim
    predictor = cnn.RegressionFFN(n_tasks=int(n_tasks), input_dim=d_in, hidden_dim=ffn_hidden,
                                  n_layers=ffn_layers, dropout=dropout)
    model = ChemToxMPNN(mp, agg, predictor, batch_norm=batch_norm, max_lr=max_lr,
                        n_descriptors=n_descriptors, n_species=n_species, n_condition=n_condition,
                        species_dim=species_dim, use_conditioning=use_conditioning)
    return model


def _mpnn_base():
    _, _, models, _ = _chemprop()
    return models.MPNN


class ChemToxMPNN(_mpnn_base()):
    """D-MPNN whose head sees the descriptors, a species embedding and FiLM conditioning."""

    def __init__(self, message_passing, agg, predictor, *, n_descriptors: int, n_species: int,
                 n_condition: int, species_dim: int = 16, batch_norm: bool = True,
                 max_lr: float = 1e-3, use_conditioning: bool = True, **kw):
        super().__init__(message_passing, agg, predictor, batch_norm=batch_norm, max_lr=max_lr, **kw)
        self.n_descriptors = int(n_descriptors)
        self.n_condition = int(n_condition)
        self.species_dim = int(species_dim)
        self.use_conditioning = bool(use_conditioning)
        self.species_embedding = (nn.Embedding(int(n_species), self.species_dim)
                                  if self.species_dim > 0 else None)
        d_z = message_passing.output_dim + self.n_descriptors + self.species_dim
        if self.use_conditioning:
            self.film = nn.Sequential(nn.Linear(self.n_condition, 64), nn.ReLU(),
                                      nn.Linear(64, 2 * d_z))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        else:
            self.film = None

    def split_x_d(self, X_d: torch.Tensor):
        n = self.n_descriptors
        desc = X_d[:, :n]
        species = X_d[:, n].long()
        cond = X_d[:, n + 1:]
        return desc, species, cond

    def fingerprint(self, bmg, V_d=None, X_d=None) -> torch.Tensor:
        H = self.bn(self.agg(self.message_passing(bmg, V_d), bmg.batch))
        if X_d is None:
            raise ValueError("ChemToxMPNN requires x_d (descriptors + species index + conditioning)")
        desc, species, cond = self.split_x_d(X_d)
        parts = [H, desc]
        if self.species_embedding is not None:
            parts.append(self.species_embedding(species))
        Z = torch.cat(parts, dim=1)
        if self.film is None:
            return Z
        gamma, beta = self.film(cond).chunk(2, dim=1)
        return Z * (1.0 + gamma) + beta


# ----------------------------------------------------------------------------- fit/predict wrapper


@dataclass
class ChempropToxModel:
    """``fit``/``predict`` wrapper so the D-MPNN runs inside :func:`sentinel.models.chemtox.cv.run_cv`.

    ``params``: ``max_epochs``, ``batch_size``, ``species_dim`` (0 drops the taxonomy embedding),
    ``use_conditioning`` (False drops the FiLM head), ``message_hidden``, ``depth``,
    ``ffn_hidden``, ``ffn_layers``, ``dropout``, ``max_lr``, ``from_foundation``
    (``None`` or ``"CheMeleon"``), ``accelerator`` (``cpu``/``gpu``), ``num_workers``,
    ``n_seeds`` (>1 averages that many seeds — a seed ensemble), ``multitask`` with ``aux_rows``
    (other ADORE challenges' training rows, already test-chemical filtered) and ``pretrain``
    (``"ecotox"``) with ``ecotox_rows`` + ``pretrain_epochs``.

    Leakage control inside ``fit``: auxiliary and pretraining rows are filtered again against the
    **current validation fold's** chemicals (CAS and InChIKey) before anything is trained, and the
    filter is asserted; the counts land in ``self.aux_info_``.
    """

    params: dict = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self):
        self.model_ = None
        self.models_: list = []
        self.enc_ = None
        self.y_mean_ = 0.0
        self.y_sd_ = 1.0
        self.n_tasks_ = 1
        self.aux_info_: dict = {}

    # ------------------------------------------------------------------ data plumbing
    def _datapoints(self, X, rows: pd.DataFrame, y=None):
        data, _featurizers, _, _ = _chemprop()
        from rdkit import Chem

        desc = np.asarray(X, dtype=np.float32)
        sp = self.enc_.species_index(rows).reshape(-1, 1)
        cond = self.enc_.condition(rows)
        x_d = np.column_stack([desc, sp, cond]).astype(np.float32)
        smiles = rows[SMILES_COL].astype("string").fillna("").to_list()
        task = self._task_index(rows) if self.n_tasks_ > 1 else None
        pts = []
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"unparseable SMILES at row {i}: {smi!r}")
            if y is None:
                target = None
            else:
                z = (float(y[i]) - self.y_mean_) / self.y_sd_
                if self.n_tasks_ > 1:
                    target = np.full(self.n_tasks_, np.nan)
                    target[task[i]] = z          # NaN in the other tasks -> masked by the criterion
                else:
                    target = np.array([z])
            pts.append(data.MoleculeDatapoint(mol=mol, y=target, x_d=x_d[i]))
        return pts

    @staticmethod
    def _task_index(rows: pd.DataFrame) -> np.ndarray:
        """Row -> multitask output index, from ``tax_group`` (fish / crusta / algae)."""
        g = rows[TAXON_GROUP_COL].astype("string").str.lower().fillna("<NA>").to_numpy()
        idx = np.zeros(len(g), dtype=int)
        for j, name in enumerate(TASK_GROUPS):
            idx[g == name] = j
        return idx

    def _featurizer(self):
        """V2 multi-hot atom featurisation (chemprop's default, and what CheMeleon requires)."""
        _, featurizers, _, _ = _chemprop()
        return featurizers.SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2()
        )

    def _dataset(self, pts):
        """Molecule dataset with the molecular graphs cached (re-featurising every epoch is the
        bottleneck on this fleet: the GPU sat at 5% utilisation without the cache)."""
        data, _, _, _ = _chemprop()
        dset = data.MoleculeDataset(pts, featurizer=self._featurizer())
        dset.cache = True
        return dset

    # ------------------------------------------------------------------ sklearn-style API
    def fit(self, X, y, *, train_rows: pd.DataFrame, valid_rows: pd.DataFrame | None = None,
            builder=None, **_ctx):
        from . import aux as AUX

        data, _, _, _ = _chemprop()
        import lightning.pytorch as pl

        p = dict(self.params)
        y = np.asarray(y, dtype=float)
        self.y_mean_, self.y_sd_ = float(np.mean(y)), float(np.std(y) or 1.0)
        self.enc_ = ConditionEncoder().fit(train_rows)
        self.n_tasks_ = len(TASK_GROUPS) if bool(p.get("multitask", False)) else 1
        self.aux_info_ = {}
        val_cas = () if valid_rows is None else valid_rows[AUX.CAS_COL].dropna().unique()
        val_ik = ()
        if valid_rows is not None and AUX.INCHIKEY_COL in valid_rows.columns:
            val_ik = valid_rows[AUX.INCHIKEY_COL].dropna().unique()

        # --- multitask: pool other challenges' training rows (already test-filtered upstream) ---
        rows_fit, X_fit, y_fit = train_rows, np.asarray(X, dtype=np.float32), y
        aux_rows = p.get("aux_rows")
        if self.n_tasks_ > 1 and aux_rows is not None and len(aux_rows):
            kept = AUX.drop_chemicals(aux_rows, val_cas, val_ik)
            self.aux_info_["multitask"] = {"n_in": len(aux_rows), "n_used": len(kept)}
            if len(kept):
                if builder is None:
                    raise ValueError("multitask needs the fitted FeatureBuilder (builder=...)")
                X_aux = builder.transform(kept)
                y_aux = pd.to_numeric(kept[builder_target(kept)], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(y_aux)
                rows_fit = pd.concat([train_rows, kept.loc[ok]], ignore_index=True)
                X_fit = np.vstack([X_fit, X_aux[ok]])
                y_fit = np.concatenate([y, y_aux[ok]])

        # --- optional ECOTOX encoder pretraining (per fold, cached) ---
        encoder_state = None
        if str(p.get("pretrain") or "").lower() == "ecotox":
            eco = p.get("ecotox_rows")
            if eco is None or not len(eco):
                raise ValueError("pretrain='ecotox' needs ecotox_rows")
            encoder_state = pretrain_ecotox_encoder(
                eco, message_hidden=int(p.get("message_hidden", 300)), depth=int(p.get("depth", 3)),
                dropout=float(p.get("dropout", 0.0)), epochs=int(p.get("pretrain_epochs", 10)),
                batch_size=int(p.get("batch_size", 128)),
                accelerator=str(p.get("accelerator", "cpu")), seed=self.seed,
                exclude_cas=val_cas, exclude_inchikey=val_ik,
            )
            self.aux_info_["pretrain"] = {"source": "ecotox", "n_rows_offered": len(eco),
                                          "epochs": int(p.get("pretrain_epochs", 10))}
            if p.get("from_foundation"):
                raise ValueError("choose either from_foundation or pretrain='ecotox', not both")

        fz = self._featurizer()
        n_seeds = max(1, int(p.get("n_seeds", 1)))
        self.models_ = []
        for s in range(n_seeds):
            seed = self.seed + s
            torch.manual_seed(seed)
            pts = self._datapoints(X_fit, rows_fit, y_fit)
            dset = self._dataset(pts)
            loader = data.build_dataloader(dset, batch_size=int(p.get("batch_size", 64)),
                                           num_workers=int(p.get("num_workers", 0)), seed=seed,
                                           shuffle=True)
            model = build_model(
                n_descriptors=int(X_fit.shape[1]),
                d_v=fz.atom_fdim,
                d_e=fz.bond_fdim,
                n_species=self.enc_.n_species,
                n_condition=self.enc_.n_condition,
                species_dim=int(p.get("species_dim", 16)),
                message_hidden=int(p.get("message_hidden", 300)),
                depth=int(p.get("depth", 3)),
                ffn_hidden=int(p.get("ffn_hidden", 300)),
                ffn_layers=int(p.get("ffn_layers", 2)),
                dropout=float(p.get("dropout", 0.0)),
                from_foundation=p.get("from_foundation"),
                max_lr=float(p.get("max_lr", 1e-3)),
                use_conditioning=bool(p.get("use_conditioning", True)),
                n_tasks=self.n_tasks_,
                encoder_state=encoder_state,
            )
            trainer = pl.Trainer(
                max_epochs=int(p.get("max_epochs", 5)),
                accelerator=str(p.get("accelerator", "cpu")),
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=bool(p.get("progress_bar", False)),
                enable_model_summary=False,
                deterministic=bool(p.get("deterministic", False)),
            )
            trainer.fit(model, loader)
            self.models_.append(model)
            self.trainer_ = trainer
        self.model_ = self.models_[0]
        return self

    def predict(self, X, *, rows: pd.DataFrame | None = None, **_ctx) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("ChempropToxModel.predict before fit")
        if rows is None:
            raise ValueError("ChempropToxModel.predict needs the row frame (SMILES, species, duration)")
        data, _, _, _ = _chemprop()
        pts = self._datapoints(X, rows)
        dset = self._dataset(pts)
        loader = data.build_dataloader(dset, batch_size=int(self.params.get("batch_size", 64)),
                                       num_workers=0, shuffle=False)
        task = self._task_index(rows) if self.n_tasks_ > 1 else None
        preds = []
        for model in (self.models_ or [self.model_]):
            model.eval()
            out = []
            with torch.no_grad():
                device = next(model.parameters()).device
                for batch in loader:
                    bmg, V_d, X_d = batch[0], batch[1], batch[2]
                    bmg.to(device)
                    X_d = X_d.to(device)
                    z = model(bmg, V_d, X_d).cpu().numpy()
                    out.append(z.reshape(len(z), -1))
            z = np.vstack(out)
            z = z[np.arange(len(z)), task] if task is not None else z.ravel()
            preds.append(z)
        return np.mean(preds, axis=0) * self.y_sd_ + self.y_mean_


def builder_target(rows: pd.DataFrame) -> str:
    """Target column on auxiliary ADORE rows (molar first, mass as the documented fallback)."""
    from sentinel.data.sources import adore as A

    if A.TARGET in rows.columns:
        return A.TARGET
    if A.TARGET_MASS in rows.columns:
        return A.TARGET_MASS
    raise KeyError("auxiliary rows carry neither the molar nor the mass target column")


def make_chemprop(seed: int = 0):
    return lambda params: ChempropToxModel(params=params, seed=seed)
