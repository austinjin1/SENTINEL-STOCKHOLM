# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the unseen-taxa stage-2 selection runner (amendment A3 review fixes).

These cover the five blockers and the text invariants of the A3 review
(`REVIEW_REPORT_PRE_TRAINING.md`, "M4 unseen-taxa — amendment A3 review (2026-11-09)"). They touch
no test block: the only parquet they open is `training_pool.parquet`, and tests that need it skip
if the data root is absent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "eval" / "chemtox_unseen_select.py"


def _load():
    spec = importlib.util.spec_from_file_location("chemtox_unseen_select", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


def _pool():
    if not M.POOL.is_file():
        pytest.skip(f"frozen pool absent: {M.POOL}")
    from sentinel.models.chemtox import unseen_taxa as U

    return M.add_grouping_columns(U.to_model_schema(pd.read_parquet(M.POOL)))


def _cfg_key(cfgs):
    return json.dumps([[c["features"], c["params"], c["channels"]] for c in cfgs],
                      sort_keys=True, default=str)


# ------------------------------------------------------------------ T4: the forbidden-column rule

def test_adapter_never_creates_below_class_columns():
    """`tax_gs`/`tax_genus` must not exist: their absence is what blocks a below-class channel."""
    pool = _pool()
    for col in M.FORBIDDEN_COLUMNS:
        assert col not in pool.columns, f"{col} must never be created on this track"


def test_forbidden_column_guard_raises():
    pool = _pool()
    with pytest.raises(AssertionError):
        M.assert_no_forbidden_columns(pool.assign(tax_gs="Danio rerio"), what="unit test")


def test_rank_phylum_is_the_admitted_grouping_and_has_three_levels():
    pool = _pool()
    assert M.ADMITTED_GROUP_COLUMNS == ("rank_phylum",)
    assert pool["rank_phylum"].nunique() == 3


# --------------------------------------------------- C-A3-5: channel-matched comparators

def test_channel_matched_comparators_never_draw_the_empty_channel_set():
    """A `_chan` comparator that selects no channel makes claim condition (d2) vacuous."""
    assert () not in M.CHANNEL_CHOICES_MATCHED
    for name, meta in M.SEARCHED.items():
        if meta.get("channels") != "matched":
            continue
        for c in M.sample_entry(name, 32):
            assert c["channels"], f"{name} drew an empty channel set"


def test_candidates_may_still_draw_no_channel():
    """§1.4.4 keeps `channels in {none, surrogate, higher_rank, both}` for the candidates."""
    assert () in M.CHANNEL_CHOICES
    assert any(not c["channels"] for c in M.sample_entry("lightgbm", 32))


# --------------------------------------------------- C-A3-4: no comparator is a duplicate

def test_no_comparator_shares_a_configuration_draw():
    """A comparator identical to a candidate would make (b)/(d2) a self-comparison."""
    keys = {n: _cfg_key(M.sample_entry(n, 32)) for n in M.SEARCHED}
    comparators = [n for n, m in M.SEARCHED.items() if m["role"] == "comparator"]
    for c in comparators:
        for other, k in keys.items():
            if other != c and k == keys[c]:
                raise AssertionError(f"comparator {c} has the same draw as {other}")


def test_published_chan_entries_use_the_published_fixed_configuration():
    """§1.4.3: the published comparators vary only the channel setting (3), not 32 draws."""
    for name in ("rf_published_chan", "xgboost_published_chan"):
        cfgs = M.sample_entry(name, 32)
        assert len(cfgs) == len(M.CHANNEL_CHOICES_MATCHED) == 3
        base = M.SEARCHED[name]["base"]
        for c in cfgs:
            assert c["params"] == M.PUBLISHED_PARAMS[base]


# --------------------------------------------------- C-A3-3: the grouping is recorded truthfully

def test_gpboost_space_never_samples_group_columns():
    assert "group_columns" not in M.GPBOOST_SPACE_TRACK["params"]
    for c in M.sample_entry("gpboost", 32):
        assert "group_columns" not in c["params"]
        assert not any(k in str(c["params"]) for k in M.FORBIDDEN_COLUMNS)


# --------------------------------------------------- C-A3-1 / C-A3-2: read-across degenerates

def test_track_readacross_degenerates_to_global_knn():
    """§5.1.5's pre-registered behaviour, and proof it runs at all (it used to raise KeyError)."""
    from sentinel.models.chemtox import unseen_taxa as U
    from sentinel.models.chemtox.data import FeatureBuilder, FeatureSpec

    pool = _pool()
    y = pd.to_numeric(pool[M.TARGET], errors="coerce").to_numpy(float)
    cls, tr, va = next(t for t in U.validation_splits(pool) if t[0] == "Actinopterygii")
    train = pool.iloc[tr].reset_index(drop=True)
    val = pool.iloc[va].reset_index(drop=True).head(50)
    ytr = y[tr]
    ok = np.isfinite(ytr)
    train, ytr = train.loc[ok].reset_index(drop=True), ytr[ok]

    b = U.ChannelBuilder(FeatureBuilder(FeatureSpec(chemical="mw_logp", fingerprint="MACCS",
                                                    published=True)), ())
    Xtr, Xva = b.fit_transform(train), b.transform(val)
    cfg = M.sample_entry("readacross", 32)[0]
    model = M.build_model("readacross", cfg["params"], 0)
    model.fit(Xtr, ytr, train_rows=train, valid_rows=val, feature_names=b.names_, builder=b,
              sample_weight=None)
    pred = model.predict(Xva, rows=val)

    assert np.isfinite(pred).all()
    w = model.widening_
    assert w["species"] == 0 and w["group"] == 0, f"species/group level fired: {w}"
    assert w["global"] == len(val)


def test_track_readacross_pins_match_species_off():
    """Even a configuration that sampled `match_species=True` must run with it off (C-A3-2)."""
    model = M.build_model("readacross", {"match_species": True, "fingerprint": "MACCS"}, 0)
    assert model.inner.match_species is False


# --------------------------------------------------- blocked entries stay blocked

def test_only_the_multitaxa_aliases_are_blocked():
    for name in ("rf_multitaxa", "xgboost_multitaxa", "rf_multitaxa_chan",
                 "xgboost_multitaxa_chan"):
        assert name in M.BLOCKED
        assert M.BLOCKED_REASON[name]
    # unblocked by the lead's 2026-11-09 ruling, branch (1)
    assert "gpboost_published_chan" not in M.BLOCKED


def test_published_gpboost_values_are_verbatim_from_the_released_models():
    """Branch (1): the authors' released `_without_DEB` models state the complete set."""
    fish = M.PUBLISHED_GPBOOST["fish_form"]
    inv = M.PUBLISHED_GPBOOST["invertebrate_form"]
    for prm in (fish, inv):
        assert prm["num_boost_round"] == 1000
        assert prm["learning_rate"] == 0.01
        assert prm["min_data_in_leaf"] == 10
        assert prm["max_depth"] == -1
        assert prm["feature_fraction"] == 1.0 and prm["bagging_fraction"] == 1.0
        assert prm["objective"] == "regression"
    assert fish["num_leaves"] == 64 and fish["lambda_l2"] == 1
    assert inv["num_leaves"] == 32 and inv["lambda_l2"] == 10


def test_published_gpboost_entry_is_fixed_config_not_a_search():
    cfgs = M.sample_entry("gpboost_published_chan", 32)
    assert len(cfgs) == 6                      # 2 published forms x 3 non-empty channel settings
    assert {c["published_form"] for c in cfgs} == {"fish_form", "invertebrate_form"}
    for c in cfgs:
        assert c["channels"]                   # C-A3-5
        assert "group_columns" not in c["params"]
        assert c["params"] in list(M.PUBLISHED_GPBOOST.values())


def test_aliases_name_their_plain_counterpart():
    assert M.ALIASES == {"rf_multitaxa": "rf", "xgboost_multitaxa": "xgboost",
                         "rf_multitaxa_chan": "rf_chan",
                         "xgboost_multitaxa_chan": "xgboost_chan"}


# --------------------------------------------------- W-A3-1 / W-A3-3: chemprop routing

def test_chemprop_is_sampled_single_task():
    cfgs = M.sample_entry("chemprop", 32)
    assert len(cfgs) == 32
    assert all(c["params"]["multitask"] is False for c in cfgs)


def test_taxon_agnostic_is_enforced_for_chemprop():
    from sentinel.models.chemtox import unseen_taxa as U

    with pytest.raises(Exception):
        U.assert_taxon_agnostic("chemprop", {"multitask": True})


# --------------------------------------------------- inventory reconciles to one set of numbers

def test_configuration_inventory_is_internally_consistent():
    searched = {n: len(M.sample_entry(n, 32)) for n in M.SEARCHED}
    assert sum(searched.values()) == 396          # 390 + 6 published-GPBoost configurations
    assert len(M.SEARCHED) == 15
    assert len(M.SEARCHED) + 12 == 27             # 15 searched + 12 fixed entries


# ======================================================================= amendment A5 (2026-11-09)

def test_a5_folds_hold_out_the_class_only_and_do_not_block_chemicals():
    """A5.1: W-M4U-3's chemical blocking is gone, because plan §1.2 allows it at test time."""
    from sentinel.models.chemtox import unseen_taxa as U

    pool = _pool()
    for cls, tr, va in M.validation_splits_classonly(pool):
        trr, var = pool.iloc[tr], pool.iloc[va]
        # the held-out class is still fully removed
        assert (trr["class"].astype(str) == cls).sum() == 0
        # ... but its chemicals are NOT, which is what makes the fold mirror the test block
        shared = set(U._up(var["test_cas"]).dropna()) & set(U._up(trr["test_cas"]).dropna())
        assert shared, f"fold {cls} still blocks chemicals"


def test_a5_surrogate_availability_on_folds_matches_the_test_blocks():
    """The defect was 0.0% on every fold against 87.6-94.6% on the test blocks."""
    from sentinel.models.chemtox import unseen_taxa as U

    pool = _pool()
    for cls, tr, va in M.validation_splits_classonly(pool):
        av = float(U.surrogate_available(pool.iloc[va], pool.iloc[tr]).mean())
        assert av > 0.80, f"fold {cls}: surrogate availability {av:.3f} does not mirror test (~0.88-0.95)"


def _two_sided_self_inclusion_check(factory):
    """Perturb ONE training row's target and report (own_row_unchanged, witness_changed).

    Replaces a vacuous first version (A5 delta review A5.2-a): it shifted `M.TARGET`
    (`result_conc1_mean_mol_log`), but every surrogate aggregate reads `unseen_taxa.TARGET`
    (`log10_conc_mol_L`). `to_model_schema` keeps both columns, so that test perturbed a column no
    surrogate code reads and **could not fail** — it would have passed against the defective code.

    This version perturbs **both** columns, so the column choice cannot hide anything, and checks
    two sides:
      * the perturbed row's OWN features are unchanged  -> no self- or same-class inclusion;
      * a row of ANOTHER class sharing the chemical CHANGES -> the check has power.
    """
    from sentinel.models.chemtox import unseen_taxa as U

    pool = _pool()
    cls, tr, va = next(t for t in M.validation_splits_classonly(pool) if t[0] == "Malacostraca")
    train = pool.iloc[tr].reset_index(drop=True)
    y = pd.to_numeric(train[U.TARGET], errors="coerce")
    train = train.loc[np.isfinite(y)].reset_index(drop=True)

    cas = U._up(train["test_cas"])
    c = train["class"].astype("string")
    per = pd.DataFrame({"cas": cas, "cls": c}).dropna().groupby("cas")["cls"].nunique()
    probe = per[per >= 2].index[0]
    rows = np.flatnonzero((cas == probe).to_numpy())
    r = int(rows[0])
    s = int(next(i for i in rows if str(c.iloc[i]) != str(c.iloc[r])))

    X0 = factory().fit_transform(train)
    pert = train.copy()
    for col in (U.TARGET, M.TARGET):
        pert.loc[r, col] = float(pert.loc[r, col]) + 1000.0
    X1 = factory().fit_transform(pert)
    own_unchanged = bool(np.allclose(X0[r], X1[r], equal_nan=True))
    witness_changed = not bool(np.allclose(X0[s], X1[s], equal_nan=True))
    return own_unchanged, witness_changed


def test_a5_no_self_inclusion_in_the_surrogate_features():
    """A5.2: a training row's own target never reaches its own features (two-sided, with power)."""
    own_unchanged, witness_changed = _two_sided_self_inclusion_check(M.OutOfClassSurrogateFeatures)
    assert witness_changed, "check has no power: perturbation did not reach another class's row"
    assert own_unchanged, "a training row's own target reached its own surrogate features"


def test_a5_self_inclusion_check_has_power_against_the_pinned_path():
    """The same check MUST flag the defective pinned implementation, or it proves nothing.

    `unseen_taxa.SurrogateFeatures` is the pre-A5 path that produced 736/736 self-inclusion on the
    old folds. If this test ever passes the pinned path, the check above has lost its power.
    """
    from sentinel.models.chemtox import unseen_taxa as U

    own_unchanged, witness_changed = _two_sided_self_inclusion_check(U.SurrogateFeatures)
    assert witness_changed
    assert not own_unchanged, "check failed to detect self-inclusion in the pinned SurrogateFeatures"


def test_a5_perturbing_only_the_unread_column_is_vacuous():
    """Documents A5.2-a: perturbing `M.TARGET` alone leaves the pinned path bit-identical."""
    from sentinel.models.chemtox import unseen_taxa as U

    assert M.TARGET != U.TARGET
    pool = _pool()
    cls, tr, va = next(t for t in M.validation_splits_classonly(pool) if t[0] == "Malacostraca")
    train = pool.iloc[tr].reset_index(drop=True)
    train = train.loc[np.isfinite(pd.to_numeric(train[U.TARGET], errors="coerce"))].reset_index(drop=True)
    X0 = U.SurrogateFeatures().fit_transform(train)
    t = train.copy()
    t[M.TARGET] = t[M.TARGET].astype(float) + 100.0
    assert np.allclose(X0, U.SurrogateFeatures().fit_transform(t), equal_nan=True)


def test_a5_surrogate_features_are_not_constant_on_validation_rows():
    """The defect made all 7 features constant on validation (correlation with y exactly 0)."""
    pool = _pool()
    cls, tr, va = next(t for t in M.validation_splits_classonly(pool) if t[0] == "Malacostraca")
    train = pool.iloc[tr].reset_index(drop=True)
    val = pool.iloc[va].reset_index(drop=True)
    y = pd.to_numeric(train[M.TARGET], errors="coerce")
    train = train.loc[np.isfinite(y)].reset_index(drop=True)

    Xva = M.OutOfClassSurrogateFeatures().fit(train).transform(val)
    assert len(np.unique(Xva[:, 2])) > 1, "surr_pool_geomean is constant on validation rows"
    assert Xva[:, 4].mean() > 0.80, "surr_available is ~0 on validation rows"


def test_a5_track_channel_builder_uses_out_of_class_surrogate():
    from sentinel.models.chemtox.data import FeatureBuilder, FeatureSpec

    b = M.TrackChannelBuilder(FeatureBuilder(FeatureSpec(chemical="mw_logp")),
                              ("surrogate", "higher_rank"))
    assert isinstance(b.surrogate_, M.OutOfClassSurrogateFeatures)


def test_a5_residual_candidates_use_the_out_of_class_base():
    m = M.build_model("lightgbm_resid_ice", {"num_boost_round": 10}, 0)
    assert isinstance(m, M.ResidualOutOfClassModel)
    assert isinstance(m.surrogate, M.OutOfClassSurrogateFeatures)


# ============================================================ sharding under the 2 h cap (A5 re-run)

def _fake_shards(tmp_path, name, shards, *, drop=None, dup=None, drift=None, folds=None):
    """Write synthetic shard files for `name` exactly as run_entry would, with optional faults."""
    from sentinel.models.chemtox import unseen_taxa as U

    cfgs = M.sample_entry(name, 32)
    folds = folds or list(U.VALIDATION_CLASSES)
    for k in range(shards):
        rows = []
        for i in range(len(cfgs)):
            if i % shards != k or i == drop:
                continue
            c = cfgs[i]
            params = dict(c["params"])
            if i == drift:
                params = {**params, "__drifted__": 1}
            row = {"entry": name, "role": M.SEARCHED[name]["role"], "config_index": i,
                   "params": params, "channels": list(c["channels"]), "mean_rmse": 1.0 + i / 100,
                   "worst_rmse": 1.5, "per_fold": []}
            rows.append(row)
            if i == dup:
                rows.append(dict(row))
        summ = M.summarise(name, rows, folds, 0)
        (tmp_path / f"{name}.shard{k}of{shards}.json").write_text(json.dumps(summ, default=str))


def test_merge_accepts_full_coverage_and_recovers_the_single_run_best(tmp_path):
    _fake_shards(tmp_path, "rf", 3)
    s = M.merge_shards("rf", tmp_path, 32)
    assert s["n_configurations"] == 32
    assert [r["config_index"] for r in s["configs"]] == list(range(32))
    assert s["best"]["config_index"] == 0            # lowest synthetic RMSE
    assert len(s["merged_from"]) == 3


@pytest.mark.parametrize("fault", [{"drop": 7}, {"dup": 5}, {"drift": 11}])
def test_merge_refuses_missing_duplicated_or_drifted_configurations(tmp_path, fault):
    _fake_shards(tmp_path, "rf", 2, **fault)
    with pytest.raises(SystemExit):
        M.merge_shards("rf", tmp_path, 32)


def test_merge_refuses_shards_that_did_not_run_all_four_folds(tmp_path):
    _fake_shards(tmp_path, "rf", 2, folds=["Malacostraca"])
    with pytest.raises(SystemExit):
        M.merge_shards("rf", tmp_path, 32)


def test_merge_tolerates_only_the_documented_effective_param_keys(tmp_path):
    """group_columns / match_species are added by run_entry to the recorded params; not drift."""
    cfgs = M.sample_entry("gpboost", 32)
    rows = [{"entry": "gpboost", "role": "candidate", "config_index": i,
             "params": {**c["params"], "group_columns": ["rank_phylum"]},
             "channels": list(c["channels"]), "mean_rmse": 1.0, "worst_rmse": 1.0, "per_fold": []}
            for i, c in enumerate(cfgs)]
    from sentinel.models.chemtox import unseen_taxa as U
    (tmp_path / "gpboost.shard0of1.json").write_text(
        json.dumps(M.summarise("gpboost", rows, list(U.VALIDATION_CLASSES), 0), default=str))
    assert M.merge_shards("gpboost", tmp_path, 32)["n_configurations"] == 32


def test_registry_gate_refuses_before_any_compute(tmp_path):
    """The gate used to fire only after all compute had run; it must now refuse up front.

    `--out tmp_path` so a broken gate can never write into the real output directory.
    """
    with pytest.raises(SystemExit):
        M.main(["--entries", "rf", "--budget", "1", "--log-registry", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        M.main(["--entries", "rf", "--shard", "0", "--shards", "2", "--log-registry",
                "--out", str(tmp_path)])
    assert not list(tmp_path.iterdir()), "a refused run wrote output"


# ===================================== chemprop placeholder (lead ruling 2026-11-09, option (a))

def test_chemprop_placeholder_gives_two_species_slots_and_one_shared_index():
    """n_species == 2 and train/validation share ONE index, so species_dim>0 is an
    information-free learned bias, never a species channel."""
    from sentinel.models.chemtox import unseen_taxa as U
    from sentinel.models.chemtox.chemprop_model import ConditionEncoder

    pool = _pool()
    assert "tax_gs" not in pool.columns                       # T4 still holds on the pool
    cls, tr, va = next(t for t in M.validation_splits_classonly(pool) if t[0] == "Malacostraca")
    w = M.TrackChemprop({"multitask": False}, 0)
    train, val = w._prep(pool.iloc[tr]), w._prep(pool.iloc[va])
    enc = ConditionEncoder().fit(train)
    assert enc.n_species == 2
    it, iv = set(enc.species_index(train).tolist()), set(enc.species_index(val).tolist())
    assert it == iv and len(it) == 1


def test_chemprop_wrapper_never_writes_the_placeholder_onto_the_pool():
    pool = _pool()
    M.TrackChemprop({}, 0)._prep(pool)
    assert "tax_gs" not in pool.columns


def test_chemprop_entry_is_built_through_the_track_wrapper_and_keeps_its_sampled_space():
    cfgs = M.sample_entry("chemprop", 32)
    assert {c["params"]["species_dim"] for c in cfgs} <= {0, 8, 16, 32}   # space unchanged
    assert any(c["params"]["species_dim"] > 0 for c in cfgs)
    assert isinstance(M.build_model("chemprop", cfgs[0]["params"], 0), M.TrackChemprop)


# ==================================== fold-level jobs (chemprop under the 2 h cap) — 2026-11-09

def _old_aggregation(per_fold):
    """Verbatim copy of run_entry's pre-refactor arithmetic, to prove the refactor is identical."""
    scores = [f["rmse"] for f in per_fold]
    finite = [s for s in scores if np.isfinite(s)]
    return {"mean_rmse": float(np.mean(finite)) if finite else float("nan"),
            "worst_rmse": float(np.max(finite)) if finite else float("nan"),
            "worse_half": (float(np.mean(sorted(finite)[len(finite) // 2:])) if finite else float("nan")),
            "n_folds_ok": len(finite)}


def test_aggregate_folds_is_bit_identical_to_the_pre_refactor_arithmetic():
    """The CPU queue was running across this refactor, so identity must be exact, not approximate."""
    rng = np.random.default_rng(0)
    for trial in range(500):
        vals = list(rng.normal(1.4, 0.3, size=4))
        if trial % 5 == 0:
            vals[int(rng.integers(0, 4))] = float("nan")
        pf = [{"rmse": v} for v in vals]
        a, b = M.aggregate_folds(pf), _old_aggregation(pf)
        assert a.keys() == b.keys()
        for k in a:
            assert (np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k], (trial, k, a[k], b[k])


def _fake_fold_jobs(tmp_path, name, *, drop_fold=None, dup_fold=None, drift=None, missing_cfg=None):
    from sentinel.models.chemtox import unseen_taxa as U

    cfgs = M.sample_entry(name, 32)
    for i, c in enumerate(cfgs):
        if i == missing_cfg:
            continue
        for j, fold in enumerate(U.VALIDATION_CLASSES):
            if (i, fold) == drop_fold:
                continue
            params = {**c["params"], **({"__drifted__": 1} if i == drift else {})}
            pf = [{"fold": fold, "rmse": 1.0 + i / 100 + j / 1000}]
            row = {"entry": name, "role": "candidate", "config_index": i, "params": params,
                   "channels": list(c["channels"]), "per_fold": pf, **M.aggregate_folds(pf)}
            (tmp_path / f"{name}.cfg{i:02d}.fold-{fold}.json").write_text(
                json.dumps(M.summarise(name, [row], [fold], 0), default=str))
            if (i, fold) == dup_fold:
                (tmp_path / f"{name}.cfg{i:02d}.fold-{fold}.dup.json").write_text(
                    json.dumps(M.summarise(name, [row], [fold], 0), default=str))


def test_fold_merge_reassembles_each_configuration_exactly(tmp_path):
    from sentinel.models.chemtox import unseen_taxa as U

    _fake_fold_jobs(tmp_path, "chemprop")
    s = M.merge_fold_jobs("chemprop", tmp_path, 32)
    assert s["n_configurations"] == 32
    for r in s["configs"]:
        assert [f["fold"] for f in r["per_fold"]] == list(U.VALIDATION_CLASSES)
        assert r["mean_rmse"] == _old_aggregation(r["per_fold"])["mean_rmse"]
    assert s["best"]["config_index"] == 0


@pytest.mark.parametrize("fault", [{"drop_fold": (3, "Branchiopoda")},
                                   {"dup_fold": (5, "Malacostraca")},
                                   {"drift": 9}, {"missing_cfg": 14}])
def test_fold_merge_refuses_incomplete_duplicated_or_drifted_jobs(tmp_path, fault):
    from sentinel.models.chemtox import unseen_taxa as U

    if "drop_fold" in fault:
        fault = {"drop_fold": (fault["drop_fold"][0], fault["drop_fold"][1])}
    _fake_fold_jobs(tmp_path, "chemprop", **fault)
    with pytest.raises(SystemExit):
        M.merge_fold_jobs("chemprop", tmp_path, 32)


def test_fold_job_cannot_log_to_the_registry(tmp_path):
    """Regression guard: the first version of this refusal was dead code (see the gate comment)."""
    with pytest.raises(SystemExit):
        M.main(["--entries", "chemprop", "--config-index", "0", "--folds", "Malacostraca",
                "--log-registry", "--out", str(tmp_path)])
    assert not list(tmp_path.iterdir()), "a refused fold job wrote output"
