"""Tests for the M4 arch-v2 split builder, auxiliary loader and single-look guard.

Each test names the lead ruling or verifier finding it defends. Nothing here fits a model; the
heavier tests read the pinned split, which carries no target column.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from sentinel.models.chemtox import arch_v2 as A
from sentinel.models.chemtox import arch_v2_guard as G

SPLIT_BUILT = Path(A.SPLIT_PATH).is_file()
needs_split = pytest.mark.skipif(not SPLIT_BUILT, reason="the v2 split has not been built yet")

# Ruling N2 of docs/preregistration/chemtox_arch_v2_lead_rulings.md: "any read of
# ecotox_aquatic.parquet for T1/T6 work outside load_training_pool() and build_protected_labels()
# is a protocol breach that voids the block concerned." Addendum 01 rev 1 adds load_test_block,
# load_auxiliary_rows and load_h4_rows, making five sanctioned readers in the guard.
#
# `arch_v2.auxiliary_rows()` reads that file DIRECTLY, and arch_v2.py is byte-frozen by ruling B1,
# so these items may never run as written. Ruling G-10 removes the environment switch that used to
# un-skip them: they stay in the file as the record of what they asserted, and they SKIP ALWAYS.
# What they asserted is now asserted by the guard's own re-check on every real call (G-9) and is
# tested on synthetic frames in tests/test_chemtox_arch_v2_guard.py.
reads_ecotox = pytest.mark.skip(
    reason="ruling G-10: arch_v2.auxiliary_rows() reads ecotox_aquatic.parquet directly, which "
           "ruling N2 makes a protocol breach for T1/T6 work. There is no opt-in. The same "
           "properties are tested through arch_v2_guard.load_auxiliary_rows on synthetic frames "
           "(G-9) in tests/test_chemtox_arch_v2_guard.py")


# ----------------------------------------------------------------------------- L1  spent ADORE

def test_scored_adore_partitions_are_derived_not_typed():
    """L1: the partition list comes out of the registry, and it is the expected union of 8."""
    parts = A.scored_adore_partitions()
    assert len(parts) == 8, sorted(parts)
    assert ("a-CA2F-same", "split_occurrence") in parts, \
        "the adore_trivial_baselines partitions must be included: a trivial baseline reads the " \
        "same held-out labels as a candidate"
    assert ("t-F2F", "split_occurrence") in parts
    # the alias normalisation must collapse 'occurrence' and 'split_occurrence'
    assert not any(sp == "occurrence" for _, sp in parts)


def test_spent_adore_id_union_counts():
    """L1: 8 partitions, 28,265 result_ids -- the figures the verifier measured."""
    ids, rep = A.spent_adore_result_ids()
    assert rep["n_partitions"] == 8
    assert rep["n_result_ids"] == 28265 == len(ids)


def test_spent_adore_ids_artefact_matches_the_code():
    if not Path(A.SPENT_IDS_PATH).is_file():
        pytest.skip("spent-id artefact not written yet")
    rec = json.loads(Path(A.SPENT_IDS_PATH).read_text())
    ids, _ = A.spent_adore_result_ids()
    assert set(rec["result_ids"]) == ids
    assert rec["n_result_ids"] == len(ids)


@needs_split
def test_no_spent_row_is_anywhere_in_the_split():
    """L1: spent rows leave the universe ENTIRELY -- no block, not even the pool."""
    ids, _ = A.spent_adore_result_ids()
    split = pd.read_parquet(A.SPLIT_PATH, columns=["result_id", "block"])
    assert not split["result_id"].astype(str).isin(ids).any()
    pro = set(pd.read_parquet(A.PROSPECTIVE, columns=["result_id"])["result_id"]
              .dropna().astype(str))
    assert not split["result_id"].astype(str).isin(pro).any()


def test_unscored_partitions_are_reported_not_subtracted():
    """L1: unscored partitions need a lead ruling, so the inventory only reports them."""
    inv = A.adore_partition_inventory()
    assert inv["n_scored"] == 8
    assert inv["n_unscored"] > 0
    assert inv["n_scored"] + inv["n_unscored"] == inv["n_partitions_total"]


# --------------------------------------------------------------------------- L2  parent-key unit

def test_holdout_unit_merges_clusters_and_parent_families():
    """L2: a parent-key family shared across two clusters must collapse into one unit."""
    tab = pd.DataFrame({
        "test_cas": ["1", "2", "3", "4", "5"],
        "chem_smiles": ["a", "b", "c", "d", "e"],
        "parent_key": ["PK1", "PK1", "PK2", "PK3", None],
        "t1_cluster": [0, 1, 1, 2, 3],
    })
    out = A.holdout_units(tab)
    u = dict(zip(out["test_cas"], out["holdout_unit"]))
    assert u["1"] == u["2"] == u["3"], "cluster 0 and 1 are bridged by PK1, and 3 is in cluster 1"
    assert u["4"] != u["1"] and u["5"] != u["1"]
    assert out["holdout_unit"].nunique() == 3


def test_holdout_unit_invariant_is_asserted_not_assumed():
    """A null parent_key must never act as a join key that merges unrelated chemicals."""
    tab = pd.DataFrame({"test_cas": ["1", "2"], "chem_smiles": ["a", "b"],
                        "parent_key": [None, None], "t1_cluster": [0, 1]})
    out = A.holdout_units(tab)
    assert out["holdout_unit"].nunique() == 2


def test_holdout_units_rejects_duplicate_cas():
    tab = pd.DataFrame({"test_cas": ["1", "1"], "chem_smiles": ["a", "a"],
                        "parent_key": ["P", "P"], "t1_cluster": [0, 0]})
    with pytest.raises(A.ArchV2Error):
        A.holdout_units(tab)


@needs_split
def test_no_parent_family_or_cluster_spans_the_holdout():
    """L2, on the real artefacts: the property the hold-out unit exists to guarantee."""
    split = pd.read_parquet(A.SPLIT_PATH)
    tab = pd.read_csv(A.CLUSTER_PATH)
    held = set(split.loc[split["block"].isin(("T1_test", "T6b_test")), "holdout_unit"].astype(int))
    pool = set(split.loc[split["block"] == "train_pool", "holdout_unit"].astype(int))
    assert not held & pool
    for col in ("t1_cluster", "parent_key"):
        sub = tab.dropna(subset=[col])
        spans = sub.groupby(col)["holdout_unit"].nunique()
        assert (spans == 1).all(), f"{col} spans units: {spans[spans > 1].head().to_dict()}"
        in_held = sub.groupby(col)["holdout_unit"].first().isin(held)
        assert set(in_held.unique()) <= {True, False}


# ----------------------------------------------------------- duration floor, bands and the window

def test_band_edges_and_window_agree_on_every_row():
    """DURATION-FLOOR-AND-BANDS: the window band IS the window, so the two cannot drift."""
    d = pd.Series([0.2, 0.3, 1.0, 23.9, 24.0, 48.0, 96.0, 96.1, 168.0, 720.0, 2160.0, 26298.0])
    b = A.assign_bands(d)
    assert b.tolist() == [None, "lt24", "lt24", "lt24", "w24_96", "w24_96", "w24_96",
                          "gt96_168", "gt96_168", "gt168_720", "gt720_2160", "gt2160"] or \
        list(b.fillna("NA")) == ["NA", "lt24", "lt24", "lt24", "w24_96", "w24_96", "w24_96",
                                 "gt96_168", "gt96_168", "gt168_720", "gt720_2160", "gt2160"]
    in_window = (b == A.WINDOW_BAND).fillna(False)
    assert (in_window.to_numpy() == d.between(*A.DURATION_WINDOW).to_numpy()).all()


def test_exactly_24h_is_in_the_window_not_the_short_tail():
    """The reason the literal half-open wording was adjusted: 24 h is a standard acute duration."""
    assert A.assign_bands(pd.Series([24.0]))[0] == A.WINDOW_BAND


def test_duration_floor_drops_nonpositive_durations():
    assert A.assign_bands(pd.Series([0.0, -1.0, A.MIN_DURATION_H])).isna().all()


@needs_split
def test_split_blocks_respect_the_window():
    split = pd.read_parquet(A.SPLIT_PATH)
    pool = split[split["block"].isin(("train_pool", "T1_test"))]
    assert (pool["duration_band"] == A.WINDOW_BAND).all()
    out = split[split["block"].isin(("T6a_test", "T6b_test"))]
    assert (out["duration_band"] != A.WINDOW_BAND).all()
    assert (split["duration_h"] > A.MIN_DURATION_H).all()


@needs_split
def test_blocks_are_row_disjoint_and_cover_the_universe():
    split = pd.read_parquet(A.SPLIT_PATH)
    assert not split["result_id"].duplicated().any()
    assert set(split["block"].unique()) == set(A.BLOCKS)


@needs_split
def test_split_file_carries_no_target(tmp_path):
    """L6: the sealed split is structure only."""
    cols = pd.read_parquet(A.SPLIT_PATH).columns
    assert set(cols) == set(A.SPLIT_COLUMNS)
    assert not [c for c in cols if "conc" in c or "log10" in c]


# -------------------------------------------------------------------------------- L4  draw rule

def test_draw_is_a_share_of_units_not_of_rows():
    units = pd.Series(list(range(100)) * 3).to_numpy()
    held = A._draw_units(units, 0.30, 7)
    assert len(held) == 30


def test_draw_is_deterministic_and_seed_sensitive():
    units = pd.Series(range(500)).to_numpy()
    assert A._draw_units(units, 0.2, 1) == A._draw_units(units, 0.2, 1)
    assert A._draw_units(units, 0.2, 1) != A._draw_units(units, 0.2, 2)


def test_draw_rejects_degenerate_shares():
    units = pd.Series(range(10)).to_numpy()
    for bad in (0.0, 1.0, 0.01):
        with pytest.raises(A.ArchV2Error):
            A._draw_units(units, bad, 0)


# --------------------------------------------------------------------------- L7  AUX-LEAK filters

@needs_split
@reads_ecotox
@pytest.mark.parametrize("target,track", [("H2", "T1"), ("H2", "T6"), ("H3", "T1"), ("H3", "T6")])
def test_auxiliary_rows_exclude_held_out_chemistry_and_sealed_cells(target, track):
    split = pd.read_parquet(A.SPLIT_PATH)
    tab = pd.read_csv(A.CLUSTER_PATH)
    aux, rep = A.auxiliary_rows(target, split, track=track, cluster_table=tab)
    held = set(split.loc[split["block"].isin(("T1_test", "T6b_test")), "holdout_unit"].astype(int))
    cas_unit = (tab.assign(_k=tab["test_cas"].astype(str)).drop_duplicates("_k")
                .set_index("_k")["holdout_unit"].astype(int))
    assert not aux["test_cas"].astype(str).map(cas_unit).isin(held).any()
    assert rep["admissible_rows"] < rep["universe_rows"], "the filters must actually remove rows"
    assert rep["chemistry_mapped_fraction"] > 0.5


@needs_split
@reads_ecotox
def test_t6_auxiliary_rows_carry_no_long_duration_label():
    """The POINT ruling: without this, 76k chronic rows up to 26,298 h reach a long-duration claim."""
    split = pd.read_parquet(A.SPLIT_PATH)
    tab = pd.read_csv(A.CLUSTER_PATH)
    aux, rep = A.auxiliary_rows("H3", split, track="T6", cluster_table=tab)
    assert rep["max_duration_h"] == A.DURATION_WINDOW[1]
    assert aux["duration_h"].between(*A.DURATION_WINDOW).all()
    assert rep["removed_outside_24_96h_for_T6"] > 0


@needs_split
@reads_ecotox
def test_t1_and_t6_auxiliary_counts_differ_and_both_are_below_the_universe():
    """MISSED item: the unfiltered §10.1 figures overstate the available signal."""
    split = pd.read_parquet(A.SPLIT_PATH)
    tab = pd.read_csv(A.CLUSTER_PATH)
    _, r1 = A.auxiliary_rows("H3", split, track="T1", cluster_table=tab)
    _, r6 = A.auxiliary_rows("H3", split, track="T6", cluster_table=tab)
    assert r1["universe_rows"] == r6["universe_rows"] == 225081
    assert r6["admissible_rows"] < r1["admissible_rows"] < r1["universe_rows"]


def test_auxiliary_rows_rejects_unknown_target_or_track():
    with pytest.raises(A.ArchV2Error):
        A.auxiliary_rows("H9", pd.DataFrame(), track="T1")
    with pytest.raises(A.ArchV2Error):
        A.auxiliary_rows("H3", pd.DataFrame(), track="T9")


@needs_split
@reads_ecotox
def test_degenerate_cas_join_raises_instead_of_silently_passing():
    """The dtype round-trip that turned the unit map into a no-op must now be loud."""
    split = pd.read_parquet(A.SPLIT_PATH)
    bad = pd.read_csv(A.CLUSTER_PATH).copy()
    bad["test_cas"] = ["ZZZ" + str(i) for i in range(len(bad))]   # nothing will map
    with pytest.raises(A.ArchV2Error, match="degenerate"):
        A.auxiliary_rows("H3", split, track="T1", cluster_table=bad)



# ------------------------------------------------------------------------------- L10  the guard
#
# Ruling B1 (i)-(v) and clarification C-1 are covered by the dedicated suite
# `tests/test_chemtox_arch_v2_guard.py`, which builds a complete synthetic guard world in a
# temporary directory and reads no universe row. What stays here are the properties that are about
# the guard's PLACE in the project rather than its behaviour under a look.


def test_guard_has_its_own_pins_experiment_and_blocks():
    assert G.EXPERIMENT == "chemtox_arch_v2_test" != "chemtox_unseen_taxa_test"
    assert G.PINS.endswith("chemtox_arch_v2_pins.json")
    assert G.BLOCKS == ("T1_test", "T6a_test", "T6b_test")
    assert G.FREEZE_RECORD.startswith("docs/preregistration/")
    assert G.PREREG_EXPERIMENT == "chemtox/prereg_snapshot"


def test_guard_does_not_import_or_touch_frozen_track_code():
    body = Path(G.__file__).read_text().split('"""', 2)[2]
    assert "unseen_taxa" not in body


def test_the_stamp_is_partial_so_nothing_is_unlocked():
    """The lead did not write the pins file, so verify_freeze() must keep raising."""
    assert not G.pins_path().exists()
    assert not G.freeze_record_path().exists()
    with pytest.raises(G.ProtectedBlockError, match="not been stamped"):
        G.verify_freeze()
    with pytest.raises(G.ProtectedBlockError, match="not been stamped"):
        G.load_training_pool()


def test_no_look_artefact_exists_for_any_block():
    for b in G.BLOCKS:
        st = G.look_state(b)
        assert st["registry_rows"] == 0
        assert st["freeze_file"] is False
        assert not any(st["markers"].values()), st
    assert not Path(G.LABELS_PATH).exists()


@needs_split
def test_protected_labels_do_not_exist_before_any_freeze():
    """L6: the sealed targets are not stored anywhere until the lead unlocks a block."""
    assert not Path(G.LABELS_PATH).exists(), (
        f"{G.LABELS_PATH} exists but no block has been authorised")


def test_registry_is_clean_for_every_block():
    for b in G.BLOCKS:
        assert G.registry_rows(b) == 0


def test_fixed_modules_are_at_their_stamped_hashes():
    """Ruling B1 / S3: arch_v2.py and arch_v2_power.py must stay byte-identical."""
    from sentinel.models.chemtox import arch_v2_power as P
    expected = {
        "sentinel/models/chemtox/arch_v2.py":
            "39ecc9cc39fb1c945a843ffa8e99d38eb73c3dce378d24d8a4566550e64fe980",
        "sentinel/models/chemtox/arch_v2_power.py":
            "0841903004361e52d0010b8be8e2c38745195e8d0629b0dd7e7c0c01243d4f7a",
    }
    for rel, want in expected.items():
        assert G.file_sha(Path(G.__file__).resolve().parents[3] / rel) == want, rel
    assert A.__file__ and P.__file__


def test_stamped_artefacts_are_at_their_stamped_hashes():
    """Ruling 1.1: the artefacts the stamp fixes."""
    root = Path(G.__file__).resolve().parents[3]
    assert G.file_sha(root / "docs/preregistration/chemtox_arch_v2_amendment.md") == \
        "fb52cd4b072fa19c8c1e19b9fe1374a0e8abb71d9492eff764f2034cb26d293b"
    assert G.file_sha(root / "docs/preregistration/chemtox_arch_v2_lead_rulings.md") == \
        "553be413bef252e5900023dc1dc983dd8f228a5825194ca70505cf6649c7f060"
    assert G.file_sha(A.SPLIT_PATH) == \
        "ab71aace46260980a6f011a561c2280fe0025226978508daa57f96493c5323a3"
    assert G.file_sha(root / "docs/preregistration/chemtox_t1_clusters_v2.csv") == \
        "a906aac1f9dd97577ff473981a1fce5aae1a509e3fb2dcb42f8726dea2d821f5"
    assert G.file_sha(root / "docs/preregistration/chemtox_arch_v2_spent_adore_ids_v2.json") == \
        "3fa94b68ad96c9018737886430ae6d1efd3b1c52dc33b2d98b4857a87f5bd1d8"
