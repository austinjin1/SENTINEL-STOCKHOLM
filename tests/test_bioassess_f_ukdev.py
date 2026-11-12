# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-F UK development design (protocol §9.4): fold invariants, fit guard, featurizer rules.

The design tests read the stored design of allowed UK rows (``scripts/eval/bioassess_f_ukdev.py design``) and are
skipped when it is absent. No French file is read; no model is fitted.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.models.bioassess import fdev as FV
from sentinel.models.bioassess import ftrack as F

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (ROOT / "sentinel/models/bioassess/fdev.py", ROOT / "scripts/eval/bioassess_f_ukdev.py")


def test_sources_keep_the_m5f_separation():
    for f in SOURCES:
        s = f.read_text()
        for banned in ("load_pretrained_encoder", "load_uk(", "keys.env", "torch.load(", "locked_dir("):
            assert banned not in s, (f.name, banned)
    runner = SOURCES[1].read_text()
    assert "assert_fit_ids(" in runner and "assert_stamped()" in runner


def test_reference_set_and_members_are_defined():
    known = set(FV.SIMPLE) | set(FV.TRANSFER) | set(FV.INDEX_MODELS) | set(FV.ML_SPECS)
    assert set(FV.UK_REFERENCES) <= known and set(FV.MEMBERS_B2) <= set(FV.ML_SPECS)
    assert not set(FV.UK_REFERENCES) & set(FV.MEMBERS_B2)
    for m in FV.MEMBERS_B2:                                    # ruling B2: no coordinates, no count descriptors
        kw = FV.ML_SPECS[m][1]
        assert kw["numeric"] == F.CONTEXT_COLS_NO_COORDS and kw["descriptors"] is False
    assert FV.fold_tag("A - B") == "ukdev:A - B"


@pytest.fixture(scope="module")
def dev():
    f = FV.UKDEV_DIR / "design.pkl"
    if not f.exists():
        pytest.skip("UK development design not built")
    with open(f, "rb") as fh:
        D, role = pickle.load(fh)
    return FV.UKDev(D, role)


def _km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def test_fold_invariants_and_guard(dev):
    S = dev.D.samples
    assert S["source"].isin(("uk", "ukd")).all()             # UK invertebrate rows and UK diatom-only rows
    for g in (dev.areas[0], dev.areas[len(dev.areas) // 2], dev.areas[-1]):
        tr, te, ctx = dev.training_ids(g), dev.test_ids(g), dev.context_ids(g)
        assert len(tr) and len(te) and not len(tr.intersection(te)) and not len(tr.intersection(ctx))
        assert set(te) <= set(ctx)
        assert pd.to_datetime(S.loc[tr, "date"]).max() <= pd.Timestamp(FV.UKDEV_TRAIN_TO)
        d = pd.to_datetime(S.loc[te, "date"])
        assert d.min() >= pd.Timestamp(FV.UKDEV_TEST_FROM) and d.max() <= pd.Timestamp(FV.UKDEV_TEST_TO)
        assert (S.loc[te, "region"] == g).all() and not (S.loc[tr, "region"] == g).any()
        assert not set(S.loc[tr, "location_id"]) & set(S.loc[te, "location_id"])
        st = S.loc[te, ["lat", "lon"]].drop_duplicates().to_numpy(float)
        box = S.loc[tr, ["lat", "lon"]].drop_duplicates().to_numpy(float)
        near = box[(np.abs(box[:, 0][:, None] - st[:, 0][None]) < 0.02).any(axis=1)]
        if len(near):
            dmin = min(_km(a, b, st[:, 0], st[:, 1]).min() for a, b in near)
            assert dmin > 1.0
        dev.assert_fit_ids(tr, g)
        with pytest.raises(F.FitGuardError):
            dev.assert_fit_ids(te[:5], g)                       # pseudo-test rows
        other = S.index[(S["region"] == g) & (pd.to_datetime(S["date"]) <= pd.Timestamp(FV.UKDEV_TRAIN_TO))]
        if len(other):
            with pytest.raises(F.FitGuardError):
                dev.assert_fit_ids(other[:5], g)                # rows of the held-out area


def test_featurizer_rules(dev):
    TB = FV.TaxonBlocks(dev.D)
    g = dev.areas[0]
    tr, te = dev.training_ids(g)[:20000], dev.test_ids(g)[:50]
    kind, kw = FV.ML_SPECS["b2_tabpfn_zs"]
    fz = FV.FFeaturizer(**kw).fit(TB, tr)
    X = fz.transform(TB, te)
    assert X.shape[0] == len(te) and X.shape[1] <= FV.TABPFN_MAX_COLS == 500
    assert fz.transform(TB, tr[:30]).shape[1] == X.shape[1]
    kind, kw = FV.ML_SPECS["rf__nocoords"]
    fr = FV.FFeaturizer(**kw).fit(TB, tr)
    Xr = fr.transform(TB, te)
    assert np.isfinite(Xr).all()                              # RF inputs: no missing values
    idx = TB.indices(te)
    assert idx.shape == (len(te), len(FV.INDEX_COLS))
    ok = np.isfinite(idx).all(axis=1)
    assert ok.any() and (idx[ok, 1] <= idx[ok, 0]).all() and ((idx[ok, 2] >= 0) & (idx[ok, 2] <= 1)).all()
