# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unseen-taxa test blocks for ChemTox (follow-on harder track, RESEARCH_PLAN §15.5 items 5-7).

Three test blocks — **insects**, **molluscs**, **amphibians** — curated from ECOTOX with ADORE's
own acute-lethality rules (the same rules :mod:`sentinel.models.chemtox.prospective` recovered), of
taxonomic classes that **no** SENTINEL ChemTox model has ever been trained or scored on. ADORE
covers fish, crustaceans and algae only.

Two independent kinds of hold-out, both decided before any fit (protocol
``docs/preregistration/chemtox_unseen_taxa_plan.md``):

* **Taxon hold-out at class level (the primary hold-out).** Every row of ``Insecta``,
  ``Gastropoda``, ``Bivalvia``, ``Polyplacophora``, ``Cephalopoda`` and ``Amphibia`` is removed from
  the training pool for *every* model, including the comparators and the auxiliary/augmentation
  rows. :func:`assert_no_test_classes` enforces it. Holding out whole classes (rather than species
  inside a class) is the conservative choice: a model can memorise a congener, not a class.
* **Chemical blocking as a reported stratum.** The training pool may legitimately contain a test
  chemical — it was measured in fish. ``chem_novel`` flags the test rows whose chemical is absent
  from the training pool by CAS **and** InChIKey, and ``parent_novel`` the stricter
  parent-structure version (salt/charge/stereo-insensitive, reusing
  :func:`sentinel.models.chemtox.prospective.parent_key`). Both are stored in the frozen file, so
  the strata are fixed by the data, not by a later decision.

Nothing here fits a model, and no function in this module reads a target value of a test block
except to compute the pre-registered power statistics, which are written into the protocol before
any model exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import prospective as PRO

#: Column-name prefixes of every taxonomy-derived feature. Everything a model could use to look up
#: the *taxon* — ADORE's Add-my-Pet traits (``tax_eco_*``, ``tax_lh_*``, ``tax_ps_*``), the
#: taxonomic ranks and the species key — is removed from every feature matrix on this track by
#: :class:`TaxonBlindBuilder` (protocol §1.1). The exposure block (duration, media, exposure type,
#: concentration type) is kept, which is why the drop cannot be expressed by
#: ``FeatureSpec.published=False``.
TRAIT_PREFIXES: tuple[str, ...] = (
    "tax_eco_", "tax_lh_", "tax_ps_", "tax_pdm", "tax_gs", "tax_group", "tax_name", "tax_all",
    "tax_class", "tax_order", "tax_family", "tax_genus", "tax_species", "species_number",
    "latin_name",
)

#: taxon_group -> the ECOTOX ``class`` values held out with it. The *class* is the unit of
#: hold-out; a block is named by its taxon group for readability.
BLOCK_CLASSES: dict[str, tuple[str, ...]] = {
    "insects": ("Insecta",),
    "molluscs": ("Gastropoda", "Bivalvia", "Polyplacophora", "Cephalopoda"),
    "amphibians": ("Amphibia",),
}
BLOCKS = tuple(BLOCK_CLASSES)
#: every held-out class, flattened: none of these may appear in any training row of any model
HELD_OUT_CLASSES: tuple[str, ...] = tuple(c for v in BLOCK_CLASSES.values() for c in v)
#: the taxa ADORE covers, i.e. the only groups a training row may come from. Both vocabularies are
#: accepted: ECOTOX writes ``crustaceans``, ADORE's challenge files write ``crusta`` (delta review
#: W-D-2), and a guard that rejected the ADORE spelling would block the auxiliary path at stage 2.
TRAINING_GROUPS = ("fish", "crustaceans", "crusta", "algae")
#: ADORE's chemical domain (the primary endpoint of each block is restricted to it, as in the
#: prospective plan §1.3: organometallics and metal salts are a different question)
ADORE_STRUCTURE_CLASSES = ("organic", "organic_multicomponent")

TARGET = "log10_conc_mol_L"


def _up(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip()


def curate(eco: pd.DataFrame) -> pd.DataFrame:
    """ADORE's acute-lethality curation rules, applied to a tidy ECOTOX frame.

    Identical to the rules recovered in ``docs/preregistration/chemtox_prospective_plan.md`` §1.1:
    effect ``MOR``; endpoint ``LC50``/``EC50``; duration exactly 24/48/72/96 h; media ``FW``/``SW``;
    concentration type ``A``/``F``/``T``; not censored (``~`` kept); a molar target; a
    RDKit-parseable structure. No taxon filter is applied here — that is the caller's hold-out.
    """
    m = (
        (eco["effect_norm"] == "MOR")
        & eco["endpoint_norm"].isin(["LC50", "EC50"])
        & eco["duration_h"].isin([24, 48, 72, 96])
        & eco["media_type_norm"].isin(["FW", "SW"])
        & eco["conc1_type_norm"].isin(["A", "F", "T"])
        & eco["censor_kind"].fillna("").isin(["", "approximate"])
        & eco[TARGET].notna()
        & eco["chem_smiles"].notna()
    )
    return eco.loc[m].copy()


def training_pool(curated: pd.DataFrame) -> pd.DataFrame:
    """Rows a model may train on: ADORE's taxa only, with every held-out class removed."""
    out = curated[curated["taxon_group"].isin(TRAINING_GROUPS)].copy()
    out = out[~out["class"].astype("string").isin(HELD_OUT_CLASSES)]
    assert_no_test_classes(out, what="training pool")
    return out


def test_block(curated: pd.DataFrame, block: str, pool: pd.DataFrame) -> pd.DataFrame:
    """One frozen test block with its pre-registered strata flags."""
    if block not in BLOCK_CLASSES:
        raise KeyError(block)
    sub = curated[curated["class"].astype("string").isin(BLOCK_CLASSES[block])].copy()
    sub["block"] = block
    sub["in_adore_domain"] = sub["chem_structure_class"].isin(ADORE_STRUCTURE_CLASSES)
    pool_cas, pool_key = set(_up(pool["test_cas"]).dropna()), set(_up(pool["chem_inchikey"]).dropna())
    sub["chem_novel"] = ~(_up(sub["test_cas"]).isin(pool_cas).fillna(False)
                          | _up(sub["chem_inchikey"]).isin(pool_key).fillna(False))
    pk = sub["chem_rdkit_can_smiles"] if "chem_rdkit_can_smiles" in sub.columns else sub["chem_smiles"]
    sub["parent_key"] = [PRO.parent_key(s) for s in pk]
    pool_pk = {PRO.parent_key(s) for s in (pool["chem_rdkit_can_smiles"]
                                           if "chem_rdkit_can_smiles" in pool.columns
                                           else pool["chem_smiles"])}
    sub["parent_novel"] = sub["chem_novel"] & ~sub["parent_key"].isin(pool_pk)
    return sub


#: ECOTOX ``taxon_group`` values that identify a held-out block even when ``class`` is missing
HELD_OUT_GROUPS = ("insects", "molluscs", "amphibians")


class TaxonHoldoutError(AssertionError):
    """A training/selection frame contains, or cannot be proven free of, held-out taxa."""


#: column names that may carry the taxon group, in order of preference
GROUP_COLUMNS = ("taxon_group", "tax_group")


def assert_no_test_classes(rows: pd.DataFrame, *, what: str = "rows",
                           class_col: str = "class", group_col: str | None = None) -> int:
    """Raise unless every row is provably outside the held-out classes.

    Hardened after the stage-1 review (C-M4U-5). Two changes from a naive guard:

    * **A missing keying column raises** instead of passing vacuously. Frames in ADORE's schema
      (``prospective.to_adore_schema``, ``aux.ecotox_auxiliary``) carry no ``class`` column, so a
      silent ``return`` would have made this guard a no-op on exactly the frames the pipeline
      builds. Such frames must be checked through ``group_col`` (``tax_group``/``taxon_group``),
      and at least one of the two columns must be present.
    * **``class`` = NA counts as held out** unless the row's taxon group is one of ADORE's
      (:data:`TRAINING_GROUPS`). An ECOTOX insect larva with a missing ``class`` would otherwise
      pass.

    Returns the number of rows it proved safe.
    """
    if rows is None:
        raise TaxonHoldoutError(f"{what}: frame is None")
    if not len(rows):
        return 0
    if group_col is None:
        group_col = next((c for c in GROUP_COLUMNS if c in rows.columns), GROUP_COLUMNS[0])
    cls = rows[class_col].astype("string") if class_col in rows.columns else None
    grp = rows[group_col].astype("string").str.lower() if group_col in rows.columns else None
    if cls is None and grp is None:
        raise TaxonHoldoutError(
            f"{what}: none of {class_col!r} or {GROUP_COLUMNS} is present, so the unseen-taxa "
            f"hold-out cannot be verified. Pass a frame that carries one of them (ADORE-schema "
            f"frames carry 'tax_group'); this guard must never pass vacuously.")
    bad = pd.Series(False, index=rows.index)
    if cls is not None:
        bad |= cls.isin(HELD_OUT_CLASSES).fillna(False)
        unknown = cls.isna()
        if unknown.any():
            if grp is None:
                raise TaxonHoldoutError(
                    f"{what}: {int(unknown.sum())} row(s) have {class_col!r} = NA and no "
                    f"{group_col!r} to fall back on — treated as held out (C-M4U-5)")
            bad |= unknown & ~grp.isin(TRAINING_GROUPS).fillna(False)
    if grp is not None:
        bad |= grp.isin(HELD_OUT_GROUPS).fillna(False)
        if cls is None:
            unknown_g = grp.isna() | ~grp.isin(TRAINING_GROUPS).fillna(False)
            bad |= unknown_g
    if bool(bad.any()):
        got = sorted(set(cls.loc[bad].astype(str)) if cls is not None else
                     set(grp.loc[bad].astype(str)))[:6]
        raise TaxonHoldoutError(
            f"{what}: {int(bad.sum())} of {len(rows)} row(s) are, or cannot be proven not to be, "
            f"held-out taxa {got} — the unseen-taxa blocks must not appear in training or "
            f"selection data for any model")
    return int(len(rows))


def assert_no_test_rows(rows: pd.DataFrame, test: pd.DataFrame, *, what: str = "rows",
                        id_col: str = "result_id") -> int:
    """Raise if any training row is a test row, or if row identity cannot be checked (C-M4U-5)."""
    if not len(rows):
        return 0
    if id_col not in rows.columns:
        raise TaxonHoldoutError(
            f"{what}: no {id_col!r} column, so test-row identity cannot be verified; this guard "
            f"must never pass vacuously")
    if id_col not in test.columns:
        raise TaxonHoldoutError(f"test frame has no {id_col!r} column")
    ids = set(test[id_col].astype("string").dropna())
    col = rows[id_col].astype("string")
    if bool(col.isna().all()):
        raise TaxonHoldoutError(f"{what}: every {id_col!r} is NA — identity cannot be verified")
    hit = col.isin(ids).fillna(False)
    if bool(hit.any()):
        raise TaxonHoldoutError(
            f"{what}: {int(hit.sum())} row(s) are test rows of an unseen-taxa block")
    return int(len(rows))


def replicate_floor(rows: pd.DataFrame) -> tuple[float, int]:
    """Pooled within-(chemical, species, duration) SD: the noise floor of a block."""
    g = rows.groupby(["test_cas", "latin_name", "duration_h"])[TARGET]
    sd = g.std(ddof=1).dropna()
    n = int(g.size().gt(1).sum())
    return (float(np.sqrt((sd ** 2).mean())) if len(sd) else float("nan")), n


def power_summary(rows: pd.DataFrame) -> dict:
    """Pre-registered power statistics of one block (computed before any model exists)."""
    dom = rows[rows["in_adore_domain"]]
    floor, groups = replicate_floor(dom)
    out = {
        "rows_all": int(len(rows)), "rows_primary": int(len(dom)),
        "chemicals_primary": int(dom["test_cas"].nunique()),
        "species_primary": int(dom["latin_name"].nunique()),
        "references_primary": int(dom["reference_number"].nunique()),
        "target_sd_primary": float(dom[TARGET].std()),
        "replicate_floor_primary": floor, "replicate_groups_primary": groups,
    }
    for flag in ("chem_novel", "parent_novel"):
        s = dom[dom[flag]]
        out[f"{flag}_rows"] = int(len(s))
        out[f"{flag}_chemicals"] = int(s["test_cas"].nunique())
        out[f"{flag}_target_sd"] = float(s[TARGET].std()) if len(s) > 1 else float("nan")
    return out


# --------------------------------------------------------------------- taxon-blind features (C-M4U-1)


class TaxonBlindBuilder:
    """A :class:`sentinel.models.chemtox.data.FeatureBuilder` with every taxonomy column removed.

    Implements protocol §1.1's trait drop **in new code**: the wrapped builder is fitted and used
    normally, and this class deletes the columns whose names start with any of
    :data:`TRAIT_PREFIXES` from the output matrix and from ``names_``. It is applied to **every**
    model on this track — candidates, comparators and baselines alike.

    Why a wrapper and not a feature flag: ``FeatureSpec.published`` is all-or-nothing, so the only
    in-tree alternatives were "traits in" or "traits *and* the exposure block out". Editing
    ``features.py`` was not an option while another workstream's frozen pass hashes it.

    Why it matters (the defect this fixes): ADORE's Add-my-Pet traits do not exist for a species
    outside ADORE, so with the traits left in, every test row would receive the *training mean*
    for each continuous trait and an **all-zero** multi-hot — one identical, fish-derived surrogate
    trait vector for the whole test block, in models that split on real trait values. That is a
    systematic train→test shift, not a neutral imputation.
    """

    def __init__(self, builder, prefixes: tuple[str, ...] = TRAIT_PREFIXES):
        self.builder = builder
        self.prefixes = tuple(prefixes)
        self.keep_: np.ndarray | None = None
        self.names_: list[str] = []
        self.dropped_: list[str] = []

    def _is_trait(self, name: str) -> bool:
        n = str(name)
        return any(n.startswith(p) or n.startswith(f"{p}=") for p in self.prefixes)

    def _compute_keep(self) -> None:
        names = list(self.builder.names_)
        self.keep_ = np.array([not self._is_trait(n) for n in names], dtype=bool)
        self.names_ = [n for n, k in zip(names, self.keep_) if k]
        self.dropped_ = [n for n, k in zip(names, self.keep_) if not k]

    def fit(self, train: pd.DataFrame) -> TaxonBlindBuilder:
        self.builder.fit(train)
        self._compute_keep()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.keep_ is None:
            raise RuntimeError("TaxonBlindBuilder.transform before fit")
        X = self.builder.transform(df)
        if X.shape[1] != len(self.keep_):
            raise RuntimeError(
                f"wrapped builder produced {X.shape[1]} columns but names_ has {len(self.keep_)}")
        return X[:, self.keep_]

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)

    @property
    def n_features(self) -> int:
        return len(self.names_)

    def __getattr__(self, item):           # delegate anything else (block_sizes, spec, ...)
        return getattr(self.builder, item)


def assert_no_trait_columns(names, *, what: str = "feature matrix") -> int:
    """Raise if any taxonomy-derived column survived into a model's feature names."""
    bad = [n for n in names if any(str(n).startswith(p) or str(n).startswith(f"{p}=")
                                  for p in TRAIT_PREFIXES)]
    if bad:
        raise TaxonHoldoutError(
            f"{what}: {len(bad)} taxonomy-derived column(s) reached the model, e.g. {bad[:5]} — "
            f"protocol §1.1 drops the trait block for every model on this track")
    return len(list(names))


#: chemprop's multitask head routes an unknown ``tax_group`` to output 0 (= fish), which would
#: silently predict every insect/mollusc/amphibian row with the fish head. On this track chemprop
#: is therefore pre-registered **single-task (taxon-agnostic)**: one head, no taxon routing.
def assert_taxon_agnostic(family: str, params: dict) -> None:
    """Raise if a configuration would route unseen taxa through a taxon-specific head (C-M4U-4)."""
    if str(family).startswith("chemprop") and bool(params.get("multitask", False)):
        raise TaxonHoldoutError(
            "chemprop multitask=True is not admissible on the unseen-taxa track: ChempropTox maps "
            "an unknown tax_group to output index 0 (fish), so every test row would be predicted "
            "by the fish head. The protocol registers chemprop as single-task (taxon-agnostic).")


# ------------------------------------------------------- chemical-matched surrogate comparators (C-M4U-2)

#: how a chemical's several surrogate (pool) values are reduced to one number
SURROGATE_REDUCERS = ("geomean", "most_sensitive")


class FishSurrogateBaseline:
    """Predict a test row from the **same chemical's** values in the training pool.

    The standard regulatory read-across for an untested taxon: take the chemical's measured acute
    value in the available surrogate species (here ADORE's fish and crustacean rows) and carry it
    across. In log10(mol/L) space the geometric mean is the arithmetic mean, and
    ``most_sensitive`` takes the minimum (the lowest lethal concentration), which is the
    conservative regulatory choice.

    ``reducer``: ``geomean`` or ``most_sensitive``. ``groups``: the surrogate taxon groups to use
    (default fish only; ``("fish", "crustaceans")`` uses both).

    Rows whose chemical has **no** surrogate value fall back to the pool's global mean, and the
    fallback is counted in :attr:`coverage_`, so the applicability of this comparator is reported
    rather than hidden (protocol §5.2).
    """

    def __init__(self, params: dict | None = None, seed: int = 0):
        p = dict(params or {})
        self.reducer = str(p.get("reducer", "geomean"))
        if self.reducer not in SURROGATE_REDUCERS:
            raise ValueError(f"reducer must be one of {SURROGATE_REDUCERS}")
        self.groups = tuple(p.get("groups", ("fish",)))
        self.seed = seed
        self.by_cas_: dict[str, float] = {}
        self.by_key_: dict[str, float] = {}
        self.global_: float = float("nan")
        self.coverage_: dict[str, int] = {"matched_cas": 0, "matched_parent": 0, "fallback": 0}

    def fit(self, X=None, y=None, *, train_rows: pd.DataFrame, sample_weight=None, **_ctx):
        rows = train_rows
        self.groups_present_ = True
        if "taxon_group" in rows.columns and self.groups:
            sub = rows[rows["taxon_group"].astype("string").str.lower().isin(self.groups)]
            # No silent substitution: if the requested surrogate taxa are absent from the fitting
            # frame (e.g. the Actinopterygii validation fold holds out every fish row), the
            # surrogate is **unavailable** and the baseline degenerates to the pool mean. Quietly
            # re-fitting on non-fish rows would make the "fish surrogate" a different estimator in
            # that fold, and would make the ICE regression circular (the response regressed on a
            # surrogate derived from the response's own rows).
            self.groups_present_ = bool(len(sub))
            rows = sub if len(sub) else rows.iloc[:0]
        target = rows[TARGET].astype(float)
        agg = "min" if self.reducer == "most_sensitive" else "mean"
        # the fallback level is always the FULL training frame's mean, never the subset's
        self.global_ = float(train_rows[TARGET].astype(float).mean())
        self.by_cas_ = (target.groupby(_up(rows["test_cas"])).agg(agg)).dropna().to_dict()
        if "parent_key" in rows.columns:
            self.by_key_ = (target.groupby(rows["parent_key"].astype("string")).agg(agg)
                            ).dropna().to_dict()
        return self

    def predict(self, X=None, *, rows: pd.DataFrame, **_ctx) -> np.ndarray:
        cas = _up(rows["test_cas"])
        key = rows["parent_key"].astype("string") if "parent_key" in rows.columns else None
        out, cov = np.empty(len(rows), dtype=float), {"matched_cas": 0, "matched_parent": 0,
                                                      "fallback": 0}
        for i in range(len(rows)):
            c = cas.iloc[i]
            if c in self.by_cas_:
                out[i] = self.by_cas_[c]
                cov["matched_cas"] += 1
                continue
            k = None if key is None else key.iloc[i]
            if k is not None and k in self.by_key_:
                out[i] = self.by_key_[k]
                cov["matched_parent"] += 1
                continue
            out[i] = self.global_
            cov["fallback"] += 1
        self.coverage_ = cov
        return out


class ICERegressionBaseline:
    """Interspecies correlation estimation (ICE / EPA Web-ICE), re-fitted on the training pool.

    Web-ICE predicts an untested taxon's acute value from a surrogate species' value with a
    least-squares regression in log space, ``log10(y_target) = a + b · log10(y_surrogate)``. The
    published models are species-pair specific and are not distributable here, so the **method's
    own form** is re-fitted on the pool: the surrogate value is the chemical's **fish** value (as in
    :class:`FishSurrogateBaseline`) and the slope/intercept are fitted on the pool's own
    cross-group pairs — fish → **every non-fish pool row** (crustaceans *and* algae), 17,081 usable
    pairs on the full pool — which is the only cross-taxon signal available without touching a test
    block. :class:`SurrogateFeatures` fits the identical regression, so the feature and the
    baseline never disagree.

    Applicability, as in Web-ICE: a prediction is made only where a surrogate value exists. Rows
    without one take the pool global mean and are counted in :attr:`coverage_`; the protocol
    reports the surrogate-available subset separately, which is how ICE results are normally read.
    """

    #: below this many usable pairs the regression is **not** fitted and ``a=0, b=1`` is kept, i.e.
    #: the ICE prediction degenerates to the surrogate value itself. This is pre-registered rather
    #: than silent (delta review W-D-4): it is what happens in the Actinopterygii validation fold,
    #: which has no fish surrogate at all (0 pairs).
    MIN_PAIRS = 30

    def __init__(self, params: dict | None = None, seed: int = 0):
        p = dict(params or {})
        self.surrogate = FishSurrogateBaseline({"reducer": p.get("reducer", "geomean"),
                                                "groups": p.get("groups", ("fish",))}, seed=seed)
        #: the pool rows the regression is fitted on: **all non-fish** pool rows (crustaceans and
        #: algae), which is what SurrogateFeatures does too — the two must agree (delta review W-D-4)
        self.target_groups = tuple(p.get("target_groups", ())) or None
        self.a_, self.b_ = 0.0, 1.0
        self.n_pairs_ = 0
        self.coverage_: dict[str, int] = {}

    def fit(self, X=None, y=None, *, train_rows: pd.DataFrame, sample_weight=None, **_ctx):
        self.surrogate.fit(train_rows=train_rows)
        grp = train_rows["taxon_group"].astype("string").str.lower()
        tgt = (train_rows[~grp.isin(self.surrogate.groups)] if self.target_groups is None
               else train_rows[grp.isin(self.target_groups)])
        if len(tgt):
            xs = _up(tgt["test_cas"]).map(self.surrogate.by_cas_).astype(float)
            ys = tgt[TARGET].astype(float)
            ok = xs.notna() & ys.notna()
            self.n_pairs_ = int(ok.sum())
            if self.n_pairs_ >= self.MIN_PAIRS:
                b, a = np.polyfit(xs[ok].to_numpy(), ys[ok].to_numpy(), 1)
                self.a_, self.b_ = float(a), float(b)
        return self

    def predict(self, X=None, *, rows: pd.DataFrame, **_ctx) -> np.ndarray:
        s = self.surrogate.predict(rows=rows)
        self.coverage_ = dict(self.surrogate.coverage_)
        matched = np.array([c in self.surrogate.by_cas_ for c in _up(rows["test_cas"])])
        out = np.where(matched, self.a_ + self.b_ * s, self.surrogate.global_)
        return out.astype(float)


def surrogate_available(rows: pd.DataFrame, pool: pd.DataFrame) -> pd.Series:
    """**The** frozen definition of "surrogate available" (delta review D-3).

    A row is surrogate-available iff its chemical has **any** record in the training pool — matched
    by CAS **or**, failing that, by ``parent_key``. This is the single rule used by
    :class:`FishSurrogateBaseline`'s fallback, by ``SurrogateFeatures.surr_available``, by the
    claim-gating subset of the protocol's §6(c) and by :func:`surrogate_coverage`. The earlier
    plan text quoted a *fish-only* coverage instead, which differed from the implementation; that
    discrepancy is what this function removes.
    """
    have_cas = set(_up(pool["test_cas"]).dropna())
    m = _up(rows["test_cas"]).isin(have_cas).fillna(False)
    if "parent_key" in pool.columns and "parent_key" in rows.columns:
        have_pk = set(pool["parent_key"].astype("string").dropna())
        m = m | rows["parent_key"].astype("string").isin(have_pk).fillna(False)
    return m


def surrogate_coverage(test: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """Pre-registered applicability of the surrogate comparators, per block (no target is read).

    Uses the single rule of :func:`surrogate_available`. ``fish_only_fraction`` is reported beside
    it for transparency, because the first plan revision quoted that number, but it gates nothing.
    """
    pool = add_parent_key(pool) if "parent_key" not in pool.columns else pool
    fish = pool[pool["taxon_group"].astype("string").str.lower() == "fish"]
    have_fish = set(_up(fish["test_cas"]).dropna())
    out = {}
    for b in sorted(test["block"].astype(str).unique()):
        t = test[(test["block"] == b) & test["in_adore_domain"]]
        m = surrogate_available(t, pool)
        f = _up(t["test_cas"]).isin(have_fish).fillna(False)
        out[b] = {"rows_primary": int(len(t)), "rows_with_surrogate": int(m.sum()),
                  "fraction": round(float(m.mean()), 4) if len(t) else float("nan"),
                  "chemicals_with_surrogate": int(t.loc[m, "test_cas"].nunique()),
                  "fish_only_fraction": round(float(f.mean()), 4) if len(t) else float("nan"),
                  "rule": "any pool record, by CAS or parent_key"}
    return out


def make_fish_surrogate(seed: int = 0):
    return lambda params: FishSurrogateBaseline(params, seed=seed)


def make_ice(seed: int = 0):
    return lambda params: ICERegressionBaseline(params, seed=seed)


# -------------------------------------------------------- reproducible GPBoost comparator (C-M4U-3)


class SeededGPBoost:
    """:class:`~sentinel.models.chemtox.gpboost_model.GPBoostTox` with the determinism holes closed.

    The closed prospective track found two GPBoost defects (plan deviations D1, D2 of
    ``chemtox_prospective_plan.md``): seed-0 predictions that were bit-identical **within** a
    process but differed **across** processes at the pass's data size, and a conjugate-gradient
    NaN/Inf that made the training-partition fit impossible. GPBoost (Zubrod et al. 2024; Yuan
    et al. 2026) is the strongest *published* method for cross-species prediction, so on this track
    it is a **mandatory comparator** — which means its fit has to be reproducible.

    Three changes, all in this new subclass (the in-tree wrapper is left untouched):

    * ``gpb.GPModel(..., seed=...)`` is given the model seed. The in-tree wrapper seeds only the
      boosting stage, leaving the random-effects solver unseeded.
    * ``num_parallel_threads`` is pinned to the configured thread budget, so a run is not
      thread-count dependent (fold-0 fits at 2 and 6 threads differ).
    * ``matrix_inversion_method="cholesky"`` replaces the iterative conjugate-gradient solver that
      produced the NaN/Inf, with ``cg_delta_conv``/``delta_rel_conv`` left at their defaults.

    :meth:`determinism_check` fits twice in-process and reports the maximum absolute difference;
    the protocol requires a **cross-process** check too, recorded in stage 2. If the fit still
    fails, the failure is reported as a failure (protocol §5.3) and never silently skipped.
    """

    def __init__(self, params: dict | None = None, seed: int = 0, threads: int = 2):
        self.params = dict(params or {})
        self.seed = int(seed)
        self.threads = int(threads)
        self.inner_ = None
        self.fit_error_: str | None = None

    def _build(self):
        from .gpboost_model import GPBoostTox

        return GPBoostTox(params=dict(self.params), seed=self.seed)

    def fit(self, X, y, **ctx):
        import gpboost as gpb

        inner = self._build()
        group_cols = tuple(inner.group_columns_)
        rows = ctx.get("train_rows")
        if rows is None:
            raise ValueError("SeededGPBoost needs train_rows")
        groups = inner._group_frame(rows, fit=True)
        p = dict(inner.params)
        p.pop("group_columns", None)
        rounds = int(p.pop("num_boost_round", 500))
        from .gpboost_model import GPBOOST_DEFAULTS

        train_params = {**GPBOOST_DEFAULTS, **p, "seed": self.seed,
                        "num_threads": self.threads}
        gp = gpb.GPModel(group_data=groups, likelihood="gaussian", seed=self.seed,
                         num_parallel_threads=self.threads,
                         matrix_inversion_method="cholesky")
        dset = gpb.Dataset(np.asarray(X, dtype=np.float32),
                           label=np.asarray(y, dtype=float),
                           weight=None if ctx.get("sample_weight") is None
                           else np.asarray(ctx["sample_weight"], dtype=float))
        inner.gp_ = gp
        inner.model_ = gpb.train(params=train_params, train_set=dset, gp_model=gp,
                                 num_boost_round=rounds)
        inner.group_columns_ = group_cols
        self.inner_ = inner
        return self

    def predict(self, X, rows: pd.DataFrame | None = None, **ctx) -> np.ndarray:
        if self.inner_ is None:
            raise RuntimeError("SeededGPBoost.predict before fit")
        return self.inner_.predict(X, rows=rows, **ctx)

    def determinism_check(self, X, y, rows: pd.DataFrame, X_pred, predict_rows: pd.DataFrame
                          ) -> dict:
        """Fit twice in this process and report the maximum absolute prediction difference.

        ``X``/``rows`` are the training matrix and frame; ``X_pred``/``predict_rows`` the frame the
        comparison predicts on and **its own** matrix. The two must correspond: an earlier version
        of this helper predicted with the *training* matrix while passing the validation frame's
        group data, which GPBoost rejects with a broadcast error (found while running the check the
        protocol requires, 2026-11-09).
        """
        if len(X_pred) != len(predict_rows):
            raise ValueError(
                f"X_pred has {len(X_pred)} rows but predict_rows has {len(predict_rows)}")
        a = SeededGPBoost(self.params, seed=self.seed, threads=self.threads)
        b = SeededGPBoost(self.params, seed=self.seed, threads=self.threads)
        pa = a.fit(X, y, train_rows=rows).predict(X_pred, rows=predict_rows)
        pb = b.fit(X, y, train_rows=rows).predict(X_pred, rows=predict_rows)
        d = float(np.max(np.abs(np.asarray(pa) - np.asarray(pb))))
        return {"max_abs_diff_in_process": d, "identical": bool(d == 0.0),
                "threads": self.threads, "seed": self.seed,
                "pythonhashseed": __import__("os").environ.get("PYTHONHASHSEED")}


def make_seeded_gpboost(seed: int = 0, threads: int = 2):
    return lambda params: SeededGPBoost(params, seed=seed, threads=threads)


# ------------------------------------------------------------- validation splits and power (W-M4U-2/3/4)

#: pool classes with ≥ 1,000 rows: the leave-one-class-out rotation mirrors the class-level test
#: hold-out far more closely than a single fish ↔ crustacean pair (algae are 0.17% of the pool).
VALIDATION_CLASSES = ("Actinopterygii", "Malacostraca", "Branchiopoda", "Maxillopoda")


def validation_splits(pool: pd.DataFrame, classes: tuple[str, ...] = VALIDATION_CLASSES):
    """Leave-one-class-out validation folds inside the training pool.

    Each fold holds out one pool **class** and removes the held-out rows' chemicals from the
    training side by CAS, InChIKey **and** ``parent_key`` — the last one added after the stage-1
    review (W-M4U-3), because a salt, counter-ion or stereoisomer shares neither CAS nor InChIKey
    and would otherwise make the selection splits easier than the test blocks they mirror.

    Yields ``(class_name, train_index, val_index)``. No test block is involved at any point.
    """
    cls = pool["class"].astype("string")
    pk = require_parent_key(pool, what="validation pool")
    for c in classes:
        val = np.flatnonzero((cls == c).fillna(False).to_numpy())
        if not len(val):
            continue
        v = pool.iloc[val]
        bad_cas = set(_up(v["test_cas"]).dropna())
        bad_key = set(_up(v["chem_inchikey"]).dropna())
        bad_pk = set(pk.iloc[val].dropna())
        keep = ((cls != c).fillna(True).to_numpy()
                & ~_up(pool["test_cas"]).isin(bad_cas).fillna(False).to_numpy()
                & ~_up(pool["chem_inchikey"]).isin(bad_key).fillna(False).to_numpy()
                & ~pk.isin(bad_pk).fillna(False).to_numpy())
        yield c, np.flatnonzero(keep), val


def mde(block: pd.DataFrame, *, rmse_level: float = 1.2, rho: float = 0.9, n_boot: int = 2000,
        seed: int = 0, cluster: str = "test_cas", alpha: float = 0.05) -> dict:
    """Minimum detectable effect of a block's primary endpoint (W-M4U-2).

    The quantity a claim needs is the sampling noise of a **paired** RMSE difference between two
    models of similar accuracy, under this block's cluster structure. It is estimated here without
    fitting anything: two synthetic predictors are given errors of the magnitude the closed track
    actually produced (``rmse_level``, default 1.2 log units — the prospective pass's range) and
    correlation ``rho`` (competing models' errors are strongly correlated; ``rho=0`` is the
    pessimistic bound and is reported too), the block's chemical clusters are resampled, and the
    half-width of the resulting paired-difference interval is the MDE.

    Returned ``mde_log_units`` is the smallest RMSE difference this block can resolve at
    ``1 - alpha``. The protocol forbids resting a claim on a difference below it, or below the
    block's replicate floor, whichever is larger.
    """
    rng = np.random.default_rng(seed)
    dom = block[block["in_adore_domain"]]
    y = dom[TARGET].to_numpy(dtype=float)
    g = dom[cluster].astype(str).to_numpy()
    uniq, inv = np.unique(g, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    out = {"n_clusters": int(len(uniq)), "n_rows": int(len(y)), "sd_target": float(y.std()),
           "rmse_level_assumed": float(rmse_level), "n_boot": n_boot, "cluster": cluster}
    for tag, r in (("mde_log_units", rho), ("mde_log_units_rho0", 0.0)):
        eA = rng.normal(0.0, rmse_level, len(y))
        eB = r * eA + np.sqrt(max(0.0, 1.0 - r * r)) * rng.normal(0.0, rmse_level, len(y))
        pA, pB = y + eA, y + eB
        d = np.empty(n_boot)
        for b in range(n_boot):
            take = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
            d[b] = (np.sqrt(np.mean((pA[take] - y[take]) ** 2))
                    - np.sqrt(np.mean((pB[take] - y[take]) ** 2)))
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
        out[tag] = float((hi - lo) / 2)
        out[f"{tag}_rho"] = float(r)
    return out


def shared_reference_stats(test: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """Study-level overlap between pool and test blocks (W-M4U-1), for the reference-clustered CI."""
    pr = set(pool["reference_number"].astype("string").dropna())
    out = {"pool_references": len(pr)}
    for b in sorted(test["block"].astype(str).unique()):
        t = test[(test["block"] == b) & test["in_adore_domain"]]
        ref = t["reference_number"].astype("string")
        shared = ref.isin(pr).fillna(False)
        top = ref.value_counts()
        out[b] = {"rows_primary": int(len(t)), "references": int(ref.nunique()),
                  "shared_references": int(ref[shared].nunique()),
                  "rows_from_shared_references": int(shared.sum()),
                  "fraction_rows_shared": round(float(shared.mean()), 4) if len(t) else None,
                  "largest_reference_rows": int(top.iloc[0]) if len(top) else 0,
                  "top10_reference_share": round(float(top.head(10).sum() / len(t)), 4)
                  if len(t) else None}
    return out


# ------------------------------------------------------------- ECOTOX -> model schema (W-M4U-5)

#: ECOTOX column -> the name the feature blocks expect. Only exposure, chemistry and the target are
#: mapped: **no taxonomy column is produced**, which is the §1.1 trait drop at the schema level.
SCHEMA_MAP = {
    "duration_h": "result_obs_duration_mean",
    "media_type_norm": "test_media_type",
    "exposure_type_norm": "test_exposure_type",
    "conc1_type_norm": "result_conc1_type",
    "chem_structure_class": "structure_class",
    "log10_conc_mol_L": "result_conc1_mean_mol_log",
}


#: Trait columns ``PublishedFeatures.fit`` requires to exist. On this track they carry a constant
#: **placeholder** (never a value): the multi-hot columns a single ``__dropped__`` level, the
#: continuous ones 0.0, identical in training and test rows. :class:`TaxonBlindBuilder` then deletes
#: every resulting column, and ``assert_no_trait_columns`` proves none survived. The placeholders
#: exist only because the in-tree block cannot be configured without them and ``features.py`` may
#: not be edited; they carry no information and are constant across every row of every frame.
TRAIT_PLACEHOLDER_MULTIHOT = ("tax_eco_climate", "tax_eco_ecozone", "tax_eco_food",
                              "tax_eco_migrate2")
TRAIT_PLACEHOLDER_CONTINUOUS = ("tax_lh_amd", "tax_lh_licm", "tax_ps_ampv", "tax_ps_ampkap",
                                "tax_ps_amppm")
TRAIT_PLACEHOLDER_VALUE = "__dropped__"


def add_parent_key(df: pd.DataFrame, *, smiles_col: str = "chem_rdkit_can_smiles") -> pd.DataFrame:
    """Attach ``parent_key`` (salt-, charge- and stereochemistry-insensitive structural key).

    The frozen ``training_pool.parquet`` does **not** carry this column (only the test blocks do),
    which made the fold blocking of §4 and the surrogate parent-level fallback silent no-ops
    (delta review D-1). It is derived here in new code, from the frozen file's own canonical SMILES,
    so no parquet is rebuilt.
    """
    if "parent_key" in df.columns:
        return df
    col = smiles_col if smiles_col in df.columns else "chem_smiles"
    if col not in df.columns:
        raise TaxonHoldoutError(
            f"cannot derive parent_key: neither {smiles_col!r} nor 'chem_smiles' is present")
    table = PRO.parent_keys(df[col])
    out = df.copy()
    out["parent_key"] = df[col].astype("string").map(table)
    return out


def require_parent_key(df: pd.DataFrame, *, what: str = "frame") -> pd.Series:
    """Return the frame's ``parent_key`` column, raising if it is missing or entirely null."""
    if "parent_key" not in df.columns:
        raise TaxonHoldoutError(
            f"{what}: no 'parent_key' column — parent-structure blocking would silently do nothing "
            f"(delta review D-1). Call add_parent_key() first.")
    col = df["parent_key"].astype("string")
    if bool(col.isna().all()):
        raise TaxonHoldoutError(f"{what}: every 'parent_key' is null")
    return col


def to_model_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ECOTOX columns to the names the ChemTox feature blocks expect.

    Keeps the ECOTOX names as well (so the guards and the surrogate baselines keep working), adds
    nothing taxonomic beyond the constant placeholders described at
    :data:`TRAIT_PLACEHOLDER_MULTIHOT`, and does **not** invent `chem_ws`/`chem_mp`: this track is
    pre-registered with ``FeatureSpec.chemical = "mw_logp"``, whose only inputs are `chem_mw` and
    `chem_rdkit_clogp`, both computed by RDKit for any molecule.
    """
    out = df.copy()
    for src, dst in SCHEMA_MAP.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    for c in TRAIT_PLACEHOLDER_MULTIHOT:
        out[c] = TRAIT_PLACEHOLDER_VALUE
    for c in TRAIT_PLACEHOLDER_CONTINUOUS:
        out[c] = 0.0
    return add_parent_key(out)


# ======================================================================================
# Cross-taxon channels (stage-1 amendment, 2026-11-09, lead decision before any numbers)
#
# The unamended track had no route by which taxon information could reach a model, while the
# surrogate/ICE comparators did (§9 of the protocol). Two channels are therefore pre-declared.
# **Fairness rule (the C-M4P-1 lesson):** every channel given to a candidate is also given to the
# mandatory comparators, and those channel-matched comparator variants are themselves the
# strongest-baseline comparators a claim must beat.
# ======================================================================================

# ------------------------------------------------------------------ Channel A: surrogate inputs

#: names of the surrogate feature columns, in order
SURROGATE_FEATURE_NAMES = (
    "surr_fish_geomean", "surr_fish_most_sensitive", "surr_pool_geomean",
    "surr_ice_prediction", "surr_available", "surr_n_records", "surr_n_pool_taxa",
)


class SurrogateFeatures:
    """Channel A — per-row, chemical-matched information drawn **only from the training pool**.

    For a row's chemical: the geometric mean and the most-sensitive (minimum) log10 value measured
    in the pool's fish rows, the same over all pool rows, the ICE-regression prediction, a
    surrogate-available flag and the number of contributing records and species. These are exactly
    the quantities the regulatory surrogate/ICE comparators use, handed to the models as features
    so a learner can weigh them against structure instead of being denied the channel.

    **Leakage properties, by construction:** :meth:`fit` reads only ``train_rows``; no test-block
    row, target or identifier is ever touched, and :meth:`transform` performs a lookup keyed on
    ``test_cas`` / ``parent_key``. Inside the selection rotation the block is re-fitted on each
    fold's *training* classes only, so the feature has exactly the availability it will have at
    test time. A chemical with no pool record gets ``surr_available = 0`` and the pool mean, which
    is the same rule the surrogate comparators use.
    """

    def __init__(self, fish_groups: tuple[str, ...] = ("fish",), with_ice: bool = True):
        self.fish_groups = tuple(fish_groups)
        self.with_ice = bool(with_ice)
        self.names_ = list(SURROGATE_FEATURE_NAMES)
        self.fish_geo_: dict[str, float] = {}
        self.fish_min_: dict[str, float] = {}
        self.pool_geo_: dict[str, float] = {}
        self.n_rec_: dict[str, int] = {}
        self.n_sp_: dict[str, int] = {}
        self.parent_geo_: dict[str, float] = {}
        self.global_: float = float("nan")
        self.ice_a_, self.ice_b_, self.ice_pairs_ = 0.0, 1.0, 0

    def fit(self, train: pd.DataFrame) -> SurrogateFeatures:
        assert_no_test_classes(train, what="SurrogateFeatures.fit")
        y = train[TARGET].astype(float)
        cas = _up(train["test_cas"])
        self.global_ = float(y.mean())
        self.fish_groups_present_ = True
        self.pool_geo_ = y.groupby(cas).mean().dropna().to_dict()
        self.n_rec_ = y.groupby(cas).size().to_dict()
        if "latin_name" in train.columns:
            self.n_sp_ = train.groupby(cas)["latin_name"].nunique().to_dict()
        if "parent_key" in train.columns:
            self.parent_geo_ = y.groupby(train["parent_key"].astype("string")).mean().dropna().to_dict()
        grp = train["taxon_group"].astype("string").str.lower() if "taxon_group" in train.columns else None
        fish = train[grp.isin(self.fish_groups)] if grp is not None else train
        self.fish_groups_present_ = bool(len(fish))
        if len(fish):
            fy, fc = fish[TARGET].astype(float), _up(fish["test_cas"])
            self.fish_geo_ = fy.groupby(fc).mean().dropna().to_dict()
            self.fish_min_ = fy.groupby(fc).min().dropna().to_dict()
        if self.with_ice and grp is not None:
            tgt = train[~grp.isin(self.fish_groups)]
            if len(tgt):
                xs = _up(tgt["test_cas"]).map(self.fish_geo_).astype(float)
                ys = tgt[TARGET].astype(float)
                ok = xs.notna() & ys.notna()
                self.ice_pairs_ = int(ok.sum())
                if self.ice_pairs_ >= ICERegressionBaseline.MIN_PAIRS:
                    b, a = np.polyfit(xs[ok].to_numpy(), ys[ok].to_numpy(), 1)
                    self.ice_a_, self.ice_b_ = float(a), float(b)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cas = _up(df["test_cas"])
        pk = df["parent_key"].astype("string") if "parent_key" in df.columns else None
        n = len(df)
        out = np.zeros((n, len(self.names_)), dtype=float)
        for i in range(n):
            c = cas.iloc[i]
            k = None if pk is None else pk.iloc[i]
            fg = self.fish_geo_.get(c)
            fm = self.fish_min_.get(c)
            pg = self.pool_geo_.get(c)
            if pg is None and k is not None:
                pg = self.parent_geo_.get(k)
            avail = 1.0 if (fg is not None or pg is not None) else 0.0
            base = fg if fg is not None else (pg if pg is not None else self.global_)
            out[i] = (
                fg if fg is not None else self.global_,
                fm if fm is not None else self.global_,
                pg if pg is not None else self.global_,
                self.ice_a_ + self.ice_b_ * base,
                avail,
                float(self.n_rec_.get(c, 0)),
                float(self.n_sp_.get(c, 0)),
            )
        return out

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)

    def surrogate_prediction(self, df: pd.DataFrame, kind: str = "ice") -> np.ndarray:
        """The surrogate/ICE point prediction for each row (the base of residual learning)."""
        X = self.transform(df)
        idx = {"fish_geomean": 0, "most_sensitive": 1, "pool_geomean": 2, "ice": 3}[kind]
        return X[:, idx]


# ------------------------------------------------------- Channel B: ranks ABOVE the held-out class

#: ECOTOX ``class`` -> ranks **strictly above class level**. The mapping is a pure function of the
#: class name, so by construction it cannot encode anything at class rank or below: the classes of
#: a test block stay unseen at every rank that defines them. Insects share ``Arthropoda`` with the
#: crustacean pool, amphibians share ``Chordata``/``Vertebrata`` with the fish pool, and molluscs
#: share only ``Protostomia`` — which is itself a result worth reporting, because the mollusc blocks
#: have the weakest higher-rank bridge to anything in the pool.
HIGHER_RANKS: dict[str, tuple[str, str, str]] = {
    # class:              (phylum,        subphylum/superclass, clade)
    "Insecta":            ("Arthropoda",  "Hexapoda",           "Protostomia"),
    "Malacostraca":       ("Arthropoda",  "Crustacea",          "Protostomia"),
    "Branchiopoda":       ("Arthropoda",  "Crustacea",          "Protostomia"),
    "Maxillopoda":        ("Arthropoda",  "Crustacea",          "Protostomia"),
    "Ostracoda":          ("Arthropoda",  "Crustacea",          "Protostomia"),
    "Gastropoda":         ("Mollusca",    "Conchifera",         "Protostomia"),
    "Bivalvia":           ("Mollusca",    "Conchifera",         "Protostomia"),
    "Polyplacophora":     ("Mollusca",    "Aculifera",          "Protostomia"),
    "Cephalopoda":        ("Mollusca",    "Conchifera",         "Protostomia"),
    "Amphibia":           ("Chordata",    "Vertebrata",         "Deuterostomia"),
    "Actinopterygii":     ("Chordata",    "Vertebrata",         "Deuterostomia"),
    "Chondrichthyes":     ("Chordata",    "Vertebrata",         "Deuterostomia"),
    "Osteichthyes":       ("Chordata",    "Vertebrata",         "Deuterostomia"),
    "Cephalaspidomorphi": ("Chordata",    "Vertebrata",         "Deuterostomia"),
}
HIGHER_RANK_NAMES = ("rank_phylum", "rank_subphylum", "rank_clade")
#: levels the one-hot uses, fixed here so train and test matrices always align
HIGHER_RANK_LEVELS = {
    "rank_phylum": ("Arthropoda", "Mollusca", "Chordata", "Other"),
    "rank_subphylum": ("Hexapoda", "Crustacea", "Conchifera", "Aculifera", "Vertebrata", "Other"),
    "rank_clade": ("Protostomia", "Deuterostomia", "Other"),
}


def higher_ranks(rows: pd.DataFrame, class_col: str = "class") -> pd.DataFrame:
    """The three above-class ranks per row (``Other`` for anything unmapped, e.g. algae)."""
    cls = rows[class_col].astype("string") if class_col in rows.columns else pd.Series(
        pd.NA, index=rows.index, dtype="string")
    tup = [HIGHER_RANKS.get(str(c), ("Other", "Other", "Other")) for c in cls]
    return pd.DataFrame(tup, columns=list(HIGHER_RANK_NAMES), index=rows.index)


class HigherRankFeatures:
    """Channel B — one-hot of the three above-class ranks, **restricted to levels the training
    frame actually varies over**.

    The only input is the class *name* and every output is a strictly higher rank, so no name can
    spell a class, order, family, genus or species (``assert_no_class_or_lower``). That alone is
    not enough, which the delta review (D-2) showed: four levels — `Mollusca`, `Hexapoda`,
    `Conchifera`, `Aculifera` — are **identically zero in the pool and non-zero only on test rows**,
    so inside this dataset each is a one-to-one proxy for a held-out class. Such a column leaks no
    *target* information (a constant cannot be fitted), but it is perfectly collinear with "is a
    test row", and for a neural model a weight on an always-zero input keeps its random
    initialisation and then multiplies 1 on test rows only — an arbitrary, seed-dependent, test-only
    offset that validation cannot see. It is the same defect as the all-zero trait multi-hot that
    §1.1 of the protocol rejects.

    **Fix, applied identically in every fold and at the final fit:** :meth:`fit` keeps only the
    levels that are **non-constant in the fitting frame**; a value whose level was dropped
    contributes zeros across that rank. So a model is never given a column it could not have
    learned from, and what Channel B can carry is exactly what the training taxa share with the
    test taxa.
    """

    def __init__(self):
        self.levels_: dict[str, tuple[str, ...]] = {}
        self.names_: list[str] = []
        self.dropped_: list[str] = []

    def fit(self, train: pd.DataFrame) -> HigherRankFeatures:
        assert_no_test_classes(train, what="HigherRankFeatures.fit")
        hr = higher_ranks(train)
        self.levels_, self.names_, self.dropped_ = {}, [], []
        for c in HIGHER_RANK_NAMES:
            v = hr[c].astype("string")
            keep = []
            for lv in HIGHER_RANK_LEVELS[c]:
                ind = (v == lv).to_numpy(dtype=float)
                if 0.0 < ind.mean() < 1.0:          # non-constant in the fitting frame
                    keep.append(lv)
                else:
                    self.dropped_.append(f"{c}={lv}")
            self.levels_[c] = tuple(keep)
            self.names_ += [f"{c}={lv}" for lv in keep]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.levels_:
            raise RuntimeError("HigherRankFeatures.transform before fit")
        hr = higher_ranks(df)
        cols = []
        for c in HIGHER_RANK_NAMES:
            v = hr[c].astype("string")
            for lv in self.levels_[c]:
                cols.append((v == lv).to_numpy(dtype=float))
        if not cols:
            return np.zeros((len(df), 0), dtype=float)
        return np.column_stack(cols)

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)


#: every rank at class level or below — none of these may appear in a feature name (Channel B guard)
CLASS_OR_LOWER_TOKENS = ("class", "tax_order", "order", "family", "genus", "species",
                         "latin_name", "tax_gs", "species_number")


def assert_no_class_or_lower(names, *, what: str = "feature matrix") -> int:
    """Raise if a feature name encodes the class rank or below (amendment Channel B guard)."""
    bad = []
    for n in names:
        s = str(n).lower()
        if s.startswith("rank_"):            # rank_phylum / rank_subphylum / rank_clade are allowed
            continue
        if any(tok in s for tok in CLASS_OR_LOWER_TOKENS):
            bad.append(n)
    if bad:
        raise TaxonHoldoutError(
            f"{what}: {len(bad)} feature(s) encode the class rank or below, e.g. {bad[:5]} — the "
            f"held-out classes must stay unseen at every rank that defines them")
    return len(list(names))


class ChannelBuilder:
    """``TaxonBlindBuilder`` plus the pre-declared cross-taxon channels.

    ``channels``: any of ``("surrogate", "higher_rank")``. Both blocks are fitted on the training
    rows only; inside the selection rotation that is the fold's training classes, so the surrogate
    features have exactly the coverage they will have at test time.
    """

    def __init__(self, builder, channels: tuple[str, ...] = (), *,
                 fish_groups: tuple[str, ...] = ("fish",)):
        self.base = TaxonBlindBuilder(builder)
        self.channels = tuple(channels)
        self.surrogate_ = SurrogateFeatures(fish_groups=fish_groups) if "surrogate" in channels else None
        self.higher_ = HigherRankFeatures() if "higher_rank" in channels else None
        self.names_: list[str] = []

    def fit(self, train: pd.DataFrame) -> ChannelBuilder:
        self.base.fit(train)
        names = list(self.base.names_)
        if self.surrogate_ is not None:
            self.surrogate_.fit(train)
            names += list(self.surrogate_.names_)
        if self.higher_ is not None:
            self.higher_.fit(train)
            names += list(self.higher_.names_)
        self.names_ = names
        assert_no_trait_columns(self.names_, what="ChannelBuilder")
        assert_no_class_or_lower(self.names_, what="ChannelBuilder")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = [self.base.transform(df)]
        if self.surrogate_ is not None:
            parts.append(self.surrogate_.transform(df))
        if self.higher_ is not None:
            parts.append(self.higher_.transform(df))
        return np.column_stack(parts).astype(np.float32)

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)

    @property
    def n_features(self) -> int:
        return len(self.names_)


class ResidualSurrogateModel:
    """Candidate variant: predict the surrogate/ICE value plus a learned correction.

    ``base_kind`` selects the offset (``ice`` by default); the wrapped model is fitted on the
    residual ``y - base`` and its prediction is added back. The base is computed by
    :class:`SurrogateFeatures`, i.e. from training-pool rows only.
    """

    def __init__(self, inner_factory, params: dict | None = None, *, base_kind: str = "ice",
                 fish_groups: tuple[str, ...] = ("fish",)):
        self.inner = inner_factory(dict(params or {}))
        self.base_kind = base_kind
        self.surrogate = SurrogateFeatures(fish_groups=fish_groups)
        self.fitted_ = False

    def fit(self, X, y, *, train_rows: pd.DataFrame, sample_weight=None, **ctx):
        self.surrogate.fit(train_rows)
        base = self.surrogate.surrogate_prediction(train_rows, self.base_kind)
        self.inner.fit(X, np.asarray(y, dtype=float) - base, train_rows=train_rows,
                       sample_weight=sample_weight, **ctx)
        self.fitted_ = True
        return self

    def predict(self, X, rows: pd.DataFrame | None = None, **ctx) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("ResidualSurrogateModel.predict before fit")
        if rows is None:
            raise ValueError("ResidualSurrogateModel needs rows")
        base = self.surrogate.surrogate_prediction(rows, self.base_kind)
        return np.asarray(self.inner.predict(X, rows=rows, **ctx), dtype=float) + base


def make_residual(inner_factory, *, base_kind: str = "ice"):
    return lambda params: ResidualSurrogateModel(inner_factory, params, base_kind=base_kind)
