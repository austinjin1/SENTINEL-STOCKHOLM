# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for the M5-F machinery: rules and guards (ftrack), designs (fdata), BioSet-MA, TabPFN hybrid.

Synthetic toy data, plus metadata and shape checks on allowed UK pool rows where the tables are on this machine.
No test computes or prints a validation metric of a new arm (protocol §9.4, "before the stamp")."""
from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
F_MODULES = ("ftrack.py", "fdata.py", "bioset_ma.py", "pfn_hybrid.py", "fevents.py")
LEVELS = ("phylum", "class", "order", "family", "genus", "name")


# ------------------------------------------------------------------------------------------------ toy design
def _toy(n_inv=160, n_dia=40, seed=0):
    """Synthetic multi-assemblage design: UK rows (allowed), UK diatom-only rows, French rows of folds G1 / G2
    dated 2015 (training period) or 2020 (development pseudo-test period)."""
    from sentinel.models.bioassess import fdata as FD

    rng = np.random.default_rng(seed)
    inv = [(f"Ip{k % 2}", f"Ic{k % 3}", f"Io{k % 5}", f"If{k % 8}", f"Ig{k}", f"Ig{k} sp") for k in range(30)]
    dia = [("Dp", "Dc", f"Do{k % 3}", f"Df{k % 6}", f"Dg{k}", f"Dg{k} sp") for k in range(20)]
    centre = {"uk": (52.0, -1.5), "fr": (46.5, 2.5)}
    rows, ti, td = [], [], []

    def tokens(key, pool, lo, hi, out, cmax):
        for k in rng.choice(len(pool), size=int(rng.integers(lo, hi)), replace=False):
            c = float(rng.integers(1, cmax))
            out.append({"key": key, **dict(zip(LEVELS, pool[k])), "mixed_otu": pool[k][5], "count": c,
                        "log_class": float(np.clip(np.floor(np.log10(c)) + 1, 1, 5))})

    for i in range(n_inv):
        src = "uk" if i % 2 == 0 else "fr"
        late = src == "fr" and i % 4 == 3
        date = pd.Timestamp("2020-05-01" if late else "2015-05-01") + pd.Timedelta(days=int(i))
        key = f"uk:{i}" if src == "uk" else f"fr:S{i}:{date.date().isoformat()}:13"
        tokens(key, inv, 4, 12, ti, 200)
        dkey = None
        if i % 3 == 0:
            dkey = f"{src}d:{i}"
            tokens(dkey, dia, 4, 10, td, 80)
        rows.append({"key": key, "source": src, "station": f"S{i}", "date": date,
                     "lat": centre[src][0] + rng.uniform(-2, 2), "lon": centre[src][1] + rng.uniform(-2, 2),
                     "altitude_m": np.nan if i % 11 == 0 else rng.uniform(0, 400), "region": f"{src}R{i % 3}",
                     "location_id": f"L{i // 2 * 2 + (i % 2)}" if i % 10 else f"L{i}", "has_inv": True,
                     "inv_key": key, "diatom_key": dkey,
                     "diatom_offset_days": float(rng.integers(-60, 60)) if dkey else np.nan,
                     "protocol_invertebrate": f"{src}:kick", "protocol_diatom": f"{src}:LM" if dkey else None,
                     "fold": "UK" if src == "uk" else ("G1" if i % 8 in (1, 3) else "G2"), "lock_role": None,
                     "uk_allowed": src == "uk"})
    for j in range(n_dia):
        key = f"ukd:x{j}"
        tokens(key, dia, 4, 10, td, 80)
        rows.append({"key": key, "source": "ukd", "station": f"D{j}", "date": pd.Timestamp("2014-04-01"),
                     "lat": 52.0 + rng.uniform(-2, 2), "lon": -1.5 + rng.uniform(-2, 2), "altitude_m": 50.0,
                     "region": f"ukR{j % 3}", "location_id": f"LD{j}", "has_inv": False, "inv_key": None,
                     "diatom_key": key, "diatom_offset_days": 0.0, "protocol_invertebrate": None,
                     "protocol_diatom": "uk:LM", "fold": "UK", "lock_role": None, "uk_allowed": True})
    S = pd.DataFrame(rows).set_index("key")
    S["doy_sin"], S["doy_cos"] = FD.doy_features(S["date"])
    Y = pd.DataFrame(rng.normal(0.0, 1.0, (len(S), len(FD.Y_COLUMNS))), index=S.index, columns=list(FD.Y_COLUMNS))
    Y = Y.mask(rng.random(Y.shape) < 0.3)
    fr = (S["source"] == "fr").to_numpy()
    for c in Y.columns:
        if c.startswith(("log10_ton", "log10_alk")):
            Y.loc[fr, c] = np.nan
        if c.startswith("log10_nkj"):
            Y.loc[~fr, c] = np.nan
    return FD.MADesign(S, {"invertebrate": pd.DataFrame(ti), "diatom": pd.DataFrame(td)}, Y, {})


def _small(name="ma_full", **over):
    from sentinel.models.bioassess import bioset_ma as MA

    base = {"d_model": 32, "trunk": 48, "max_epochs": 3, "patience": 3, "batch_size": 64, "prevalence_min_pool": 5}
    base.update(over)
    return MA.recipe(name, **base)


def _fit_toy(D, cfg, seed=0, g="G1"):
    from sentinel.models.bioassess import bioset_ma as MA
    from sentinel.models.bioassess import fdata as FD
    from sentinel.models.bioassess import ftrack as F

    S = D.samples
    pool = MA.usable_ids(S, F.training_ids(S, "F-A", g, "dev"), cfg["modalities"])
    F.assert_fit_ids(pool, S, "F-A", g, "dev")
    fit_ids, val_ids = MA.inner_split(S, pool, seed=0)
    feat = MA.MAFeaturizer(cfg).fit(D, pool)
    tr, va = feat.transform(D, fit_ids), feat.transform(D, val_ids)
    cols = list(FD.Y_COLUMNS)
    fitted = MA.fit_ma(cfg, feat, tr, D.Y.loc[fit_ids, cols].to_numpy(float), va,
                       D.Y.loc[val_ids, cols].to_numpy(float), columns=cols, primary=FD.PRIMARY["F-A"], seed=seed)
    return fitted, feat, pool, fit_ids, val_ids


# ------------------------------------------------------------------------------------------------ ftrack: folds
def test_fold_table_holds_each_department_once():
    from sentinel.models.bioassess import ftrack as F

    F.validate_folds()
    assert len(F.DEPT_TO_FOLD) == 96 and set(F.FOLDS) == set(F.FOLD_NAMES) == set(F.MERGE_PARTNER)
    assert all(F.MERGE_PARTNER[F.MERGE_PARTNER[g]] == g and F.MERGE_PARTNER[g] != g for g in F.FOLDS)
    assert len(F.CONFIGS) == F.CONFIG_CAP == 8 and F.HOLM_FAMILY == len(F.TRACKS) * 6 * 2


@pytest.mark.parametrize("dept, commune, expect", [
    ("1", None, "01"), (6.0, None, "06"), ("20", None, "2A"), ("2b", None, "2B"), ("971", None, None),
    (None, "75056", "75"), (np.nan, "2A004", "2A"), (None, None, None), ("99", None, None), (None, "97105", None)])
def test_department_code(dept, commune, expect):
    from sentinel.models.bioassess import ftrack as F

    assert F.department_code(dept, commune) == expect
    assert F.fold_of(dept, commune) == (None if expect is None else F.DEPT_TO_FOLD[expect])


# ------------------------------------------------------------------------------------------------ ftrack: lock
def _ops():
    return pd.DataFrame({"station": ["05001", "05001", "05002", "05003", "05004", "05004"],
                         "date": ["2021-12-31", "2022-06-01", "2023-01-01", "2025-12-31", "2026-01-01", "2026-01-01"],
                         "support": ["13", "13", "13", "10", "13", "13"], "op_id": ["a", "b", "c", "d", "e", "f"]})


def test_lock_roles_and_keys():
    from sentinel.models.bioassess import ftrack as F

    assert [F.lock_role(d) for d in ("2021-12-31", "2022-01-01", "2022-12-31", "2023-01-01", "2025-12-31",
                                     "2026-01-01")] == [None, "buffer", "buffer", "test", "test", "future"]
    assert F.lock_key(" 05001 ", dt.datetime(2023, 5, 4, 10), 13) == "FR:05001:2023-05-04:13"
    assert F.row_key("05001", "2023-05-04", "13") == "fr:05001:2023-05-04:13"
    doc = F.build_lock(_ops())
    assert doc["n_keys"] == 4 and "FR:05001:2021-12-31:13" not in doc["keys"]          # unlocked rows are not listed
    assert doc["counts"]["role"] == {"buffer": 1, "future": 1, "test": 2}
    assert doc["keys"]["FR:05004:2026-01-01:13"]["op_ids"] == ["e", "f"]              # replicates share one key


def test_lock_file_is_written_once_read_only_and_hash_checked(tmp_path):
    from sentinel.models.bioassess import ftrack as F

    out = tmp_path / "lock.json"
    assert not F.chemistry_request_allowed(out)                                       # gate G2: lock first
    path, sha = F.write_lock(_ops(), out)
    assert len(sha) == 64 and not os.access(path, os.W_OK) and F.chemistry_request_allowed(out)
    with pytest.raises(F.FitGuardError):
        F.write_lock(_ops(), out)
    roles = F.read_lock(out)
    assert roles["fr:05002:2023-01-01:13"] == "test" and roles["fr:05001:2022-06-01:13"] == "buffer"
    bad = tmp_path / "tampered.json"
    bad.write_text(out.read_text().replace("FR:05002:2023-01-01:13", "FR:05009:2023-01-01:13"))
    with pytest.raises(F.FitGuardError):
        F.read_lock(bad)


def test_window_counts_take_no_result_values():
    from sentinel.models.bioassess import ftrack as F

    ops = pd.DataFrame({"key": ["k1", "k2"], "station": ["A", "B"], "date": ["2023-06-15", "2023-06-15"]})
    obs = pd.DataFrame({"station": ["A", "A", "A", "B"], "date": ["2023-06-01", "2023-01-10", "2023-07-20", "2020-01-01"],
                        "parameter": ["1302", "1302", "1302", "1302"]})
    c = F.window_counts(ops, obs).set_index(["key", "window"])["n_obs"].to_dict()
    assert c == {("k1", "pm30d"): 1, ("k1", "prior365d"): 2}
    with pytest.raises(F.FitGuardError):
        F.window_counts(ops, obs.assign(result=1.0))


# ------------------------------------------------------------------------------------------------ ftrack: pools
def _split():
    rows = {
        "fr:a": ("fr", "2015-06-01", 48.00, 2.00, "G1", None, False, True),     # G1 training-period row
        "fr:b": ("fr", "2020-06-01", 48.00, 2.00, "G1", None, False, True),     # G1 dev pseudo-test
        "fr:c": ("fr", "2015-06-01", 48.004, 2.00, "G2", None, False, True),    # G2, 0.45 km from fr:b's station
        "fr:d": ("fr", "2015-06-01", 47.00, 3.00, "G2", None, False, True),     # G2 training row
        "fr:e": ("fr", "2018-06-01", 47.10, 3.00, "G2", None, False, True),     # dev buffer year
        "fr:f": ("fr", "2021-06-01", 47.20, 3.00, "G2", None, False, True),     # trainable in the test design only
        "fr:g": ("fr", "2022-06-01", 47.30, 3.00, "G2", "buffer", False, True),
        "fr:h": ("fr", "2023-06-01", 48.50, 2.50, "G1", "test", False, True),   # reserved test row
        "fr:i": ("fr", "2024-06-01", 47.40, 3.00, "G2", "test", False, True),
        "fr:j": ("fr", "2020-06-01", 48.20, 2.20, "G1", None, False, False),    # diatom-only: not a scored unit
        "fr:k": ("fr", "2015-06-01", 16.0, -61.5, None, None, False, True),     # overseas: no fold
        "fr:l": ("fr", "2015-06-01", 48.5045, 2.50, "G2", None, False, True),   # 0.5 km from the G1 test station
        "uk:1": ("uk", "2015-06-01", 52.0, -1.0, "UK", None, True, True),
        "uk:2": ("uk", "2022-06-01", 52.1, -1.0, "UK", None, False, True),      # after the allowed period
        "ukd:3": ("ukd", "2015-06-01", 52.2, -1.0, "UK", None, True, False),
    }
    S = pd.DataFrame.from_dict(rows, orient="index", columns=["source", "date", "lat", "lon", "fold", "lock_role",
                                                                "uk_allowed", "has_inv"])
    S["date"] = pd.to_datetime(S["date"])
    return S


def test_training_and_test_pools():
    from sentinel.models.bioassess import ftrack as F

    S = _split()
    assert set(F.test_ids(S, "G1", "dev")) == {"fr:b"}
    assert set(F.test_ids(S, "G1", "test")) == {"fr:h"}
    assert set(F.test_ids(S, "G2", "test")) == {"fr:i"}
    uk = {"uk:1", "ukd:3"}
    assert set(F.training_ids(S, "F-A", "G1", "dev")) == {"fr:d", "fr:l"} | uk       # fr:c is inside the 1-km buffer
    assert set(F.training_ids(S, "F-T", "G1", "test")) == {"fr:c", "fr:d", "fr:e", "fr:f"} | uk   # fr:l within 1 km
    assert set(F.training_ids(S, "F-A", "G2", "dev")) == {"fr:a"} | uk
    assert set(F.training_ids(S, "F-Z", None, "test")) == uk                         # zero-shot: no French row
    for design in F.DESIGNS:
        for g in ("G1", "G2"):
            tr = set(F.training_ids(S, "F-A", g, design))
            assert not tr & set(F.test_ids(S, g, design))
            assert not tr & set(S.index[S["lock_role"].notna()])                     # locked rows are never trainable
            assert "uk:2" not in tr and "fr:k" not in tr


def test_fit_guard_refuses_protected_rows():
    from sentinel.models.bioassess import ftrack as F

    S = _split()
    F.assert_fit_ids(["fr:d", "uk:1"], S, "F-A", "G1", "dev")
    for bad in ("fr:b", "fr:c", "fr:a", "fr:e", "fr:g", "fr:h", "uk:2", "not:there"):
        with pytest.raises(F.FitGuardError):
            F.assert_fit_ids(["fr:d", bad], S, "F-A", "G1", "dev")
    with pytest.raises(F.FitGuardError):
        F.assert_fit_ids(["fr:d"], S, "F-Z", None, "dev")
    with pytest.raises(ValueError):
        F.training_ids(S.drop(columns=["lock_role"]), "F-A", "G1", "dev")


def test_checkpoint_guard(tmp_path):
    import torch

    from sentinel.models.bioassess import ftrack as F

    S = _split()
    ck = tmp_path / "m.pt"
    torch.save({"state": {}}, ck)
    with pytest.raises(F.FitGuardError):                               # no M5-F record (an M5 / M5-X checkpoint)
        F.load_checkpoint(ck, S, "F-A", "G1", "dev")
    F.write_train_ids_sidecar(ck, ["fr:d", "uk:1"], track="F-A", g="G1", design="dev")
    obj = F.load_checkpoint(ck, S, "F-A", "G1", "dev")
    F.require_guarded(obj, "F-A", "G1", "dev")
    with pytest.raises(F.FitGuardError):
        F.require_guarded(obj, "F-A", "G2", "dev")
    with pytest.raises(F.FitGuardError):
        F.require_guarded({"state": {}}, "F-A", "G1", "dev")
    for track, g, design in (("F-A", "G2", "dev"), ("F-Z", "G1", "dev"), ("F-A", "G1", "test")):
        with pytest.raises(F.FitGuardError):                           # trained for another fold / track / design
            F.load_checkpoint(ck, S, track, g, design)
    F.write_train_ids_sidecar(ck, ["fr:d", "fr:b"], track="F-A", g="G1", design="dev")
    with pytest.raises(F.FitGuardError):                               # trained on a pseudo-test row
        F.load_checkpoint(ck, S, "F-A", "G1", "dev")


# ------------------------------------------------------------------------------------------------ static rules
def _f_sources() -> dict:
    files = [REPO / "sentinel/models/bioassess" / f for f in F_MODULES]
    files += sorted(REPO.glob("scripts/*/bioassess_f_*.py")) + sorted(REPO.glob("sentinel/**/bioassess_f*.py"))
    return {str(f.relative_to(REPO)): f.read_text() for f in files if f.is_file()}


def test_static_single_checkpoint_loader_and_no_locked_store_access():
    src = _f_sources()
    assert len(src) >= len(F_MODULES)
    loads = {k: len(re.findall(r"torch\.load\(", v)) for k, v in src.items()}
    assert loads.pop("sentinel/models/bioassess/ftrack.py") == 1        # ftrack.load_checkpoint only
    assert not any(loads.values()), loads
    for k, v in src.items():
        if not k.endswith("ftrack.py"):
            assert not re.search(r"naiades[\"'/, ]+\s*[\"']?locked", v), k   # only ftrack names the locked store
            assert "locked_dir(" not in v, k
        assert "load_uk(" not in v, k                                   # the M5 loader reads every pool row's labels
        assert "load_pretrained_encoder" not in v, k                    # no M5 / M5-X checkpoint reuse
        assert "keys.env" not in v, k


def test_static_scripts_pass_fits_through_the_guard():
    for k, v in _f_sources().items():
        if k.startswith("scripts/") and re.search(r"\bfit_ma(_chunk)?\(|hybrid_predict\(|\.fit\(", v):
            assert "assert_fit_ids(" in v, k


def test_m5f_modules_are_outside_the_m5x_scoring_closure():
    from sentinel.models.bioassess import xtestlook as XT

    closure = set(XT.code_closure())
    assert not {f"sentinel/models/bioassess/{f}" for f in F_MODULES} & closure


# ------------------------------------------------------------------------------------------------ fdata
def test_attach_nearest_same_station_within_90_days():
    from sentinel.models.bioassess import fdata as FD

    inv = pd.DataFrame({"station": ["A", "A", "B", "C"], "date": ["2015-06-01", "2016-06-01", "2015-06-01", "2015-06-01"]})
    dia = pd.DataFrame({"station": ["A", "A", "A", "B", "D"],
                        "date": ["2015-05-20", "2015-08-01", "2016-10-01", "2015-08-30", "2015-06-01"],
                        "key": ["d1", "d2", "d3", "d4", "d5"]})
    out = FD.attach_nearest(inv, dia)
    assert list(out["diatom_key"]) == ["d1", None, "d4", None]          # 122 days is too far; no diatoms at C
    assert out["diatom_offset_days"].iloc[0] == -12 and out["diatom_offset_days"].iloc[2] == 90
    assert np.isnan(out["diatom_offset_days"].iloc[1])
    assert FD.attach_nearest(inv, dia.iloc[:0])["diatom_key"].isna().all()


def test_target_tables():
    from sentinel.models.bioassess import fdata as FD
    from sentinel.models.bioassess import ftrack as F

    assert set(FD.PRIMARY) == set(F.TRACKS) and all(len(v) == 6 for v in FD.PRIMARY.values())
    assert all(c in FD.Y_COLUMNS for v in FD.PRIMARY.values() for c in v)
    assert all(c.endswith("__pm30d") for c in FD.PRIMARY["F-A"]) and all(c.endswith("__p365") for c in FD.PRIMARY["F-T"])
    assert len(FD.Y_COLUMNS) == 2 * len(FD.TARGETS) == len(set(FD.Y_COLUMNS))
    np.testing.assert_allclose(FD.transform([0.0, 0.5, 10.0], "log10", 0.01), [-2.0, np.log10(0.5), 1.0])
    np.testing.assert_allclose(FD.transform([7.2], None, None), [7.2])
    s, c = FD.doy_features(pd.to_datetime(["2020-01-01", "2020-07-01"]))
    np.testing.assert_allclose(s ** 2 + c ** 2, 1.0)


def test_toy_design_restrict():
    from sentinel.models.bioassess import fdata as FD

    D = _toy()
    ids = D.samples.index[:10]
    R = FD.restrict(D, ids)
    assert list(R.samples.index) == list(ids) and list(R.Y.index) == list(ids)
    assert set(R.tokens["invertebrate"]["key"]) == set(D.samples.loc[ids, "inv_key"].dropna())
    assert set(R.tokens["diatom"]["key"]) == set(D.samples.loc[ids, "diatom_key"].dropna())


# ------------------------------------------------------------------------------------------------ pool prevalence
def test_pool_prevalence_leaves_the_sample_out_and_flags_small_pools():
    from sentinel.models.bioassess import bioset_ma as MA

    n = 12
    keys = np.array([f"k{i}" for i in range(n)], object)
    lat, lon = np.full(n, 50.01), np.full(n, 1.01)
    name = np.zeros((n, 2), np.int32)
    name[:, 0] = 3
    name[:6, 1] = 2                                                     # name 2 in six samples, name 3 in all twelve
    mask = name > 0
    P = MA.PoolPrevalence.fit(keys, lat, lon, name, mask, 6, cell_km=50, radius_km=200, min_pool=10)
    prev, miss = P.lookup(keys[:1], lat[:1], lon[:1], name[:1], mask[:1])
    np.testing.assert_allclose(prev[0], [11 / 11, 5 / 11], atol=1e-6)   # own presence removed
    assert not miss.any()
    q = np.array([[2, 4, 0]], np.int32)                                 # a new sample: seen, absent-from-pool, UNK
    qm = np.array([[True, True, True]])
    prev, miss = P.lookup(np.array(["new"], object), lat[:1], lon[:1], q, qm)
    np.testing.assert_allclose(prev[0], [6 / 12, 0.0, 0.0], atol=1e-6)
    assert miss[0].tolist() == [False, False, True]
    prev, miss = P.lookup(np.array(["new"], object), np.array([50.9]), np.array([1.0]), q, qm)   # ~100 km away
    np.testing.assert_allclose(prev[0, 0], 0.5, atol=1e-6)
    prev, miss = P.lookup(np.array(["far"], object), np.array([40.0]), np.array([1.0]), q, qm)   # > 1000 km away
    assert miss.all() and not prev.any()
    P9 = MA.PoolPrevalence.fit(keys[:10], lat[:10], lon[:10], name[:10], mask[:10], 6, min_pool=10)
    _, miss = P9.lookup(keys[:1], lat[:1], lon[:1], name[:1], mask[:1])
    assert miss.all()                                                   # 9 other samples < 10
    _, miss = P9.lookup(np.array(["new"], object), lat[:1], lon[:1], name[:1], np.array([[True, False]]))
    assert miss.tolist() == [[False, False]]                            # padding is never flagged


# ------------------------------------------------------------------------------------------------ featurizer
def test_featurizer_shapes_caps_and_unknowns():
    from sentinel.models.bioassess import bioset_ma as MA

    D = _toy()
    S = D.samples
    cfg = _small(token_cap={"invertebrate": 6, "diatom": 5})
    uk = S.index[S["source"].isin(["uk", "ukd"])]
    feat = MA.MAFeaturizer(cfg).fit(D, uk)
    assert feat.source_map == {"uk": 1, "ukd": 2} and feat.proto_map["invertebrate"] == {"uk:kick": 1}
    d = feat.transform(D, S.index)
    N = len(S)
    inv, dia = d.mod["invertebrate"], d.mod["diatom"]
    assert inv.lev.shape[1:] == (6, 6) and dia.lev.shape[1:] == (5, 6) and inv.lev.dtype == np.int32
    assert inv.feat.shape[2] == MA.N_TOKEN_FEAT and inv.mask.sum(1).max() <= 6 and inv.mask.sum(1).min() >= 1
    assert inv.mask[:, 0].all() and (np.diff(inv.mask.astype(int), axis=1) <= 0).all()      # packed to the left
    assert (inv.feat[..., 0][:, :-1] >= inv.feat[..., 0][:, 1:] - 1e-6)[inv.mask[:, 1:]].all()   # most abundant first
    assert d.ctx.shape == (N, 2 * len(cfg["context_cols"])) and np.isfinite(d.ctx).all()
    k = list(cfg["context_cols"]).index("altitude_m")
    np.testing.assert_array_equal(d.ctx[:, len(cfg["context_cols"]) + k] == 1.0, S["altitude_m"].isna().to_numpy())
    assert (inv.index >= 0).tolist() == S["has_inv"].tolist()
    assert (dia.index >= 0).tolist() == S["diatom_key"].notna().tolist()
    fr = (S["source"] == "fr").to_numpy()
    assert (d.src[fr] == 0).all() and (d.src[~fr] > 0).all()            # unseen source -> unknown token
    assert (inv.proto[fr] == 0).all() and (inv.proto[(S["source"] == "uk").to_numpy()] == 1).all()
    assert (dia.offset[dia.index < 0] == 0).all() and np.isfinite(dia.offset).all()
    assert set(np.unique(inv.feat[..., MA.I_MISS])) <= {0.0, 1.0}
    assert ((inv.feat[..., MA.I_PREV] >= 0) & (inv.feat[..., MA.I_PREV] <= 1)).all()
    assert not inv.feat[~inv.mask].any()
    with pytest.raises(ValueError):
        feat.transform(D, list(S.index[:3]) + [S.index[0]])


def test_featurizer_rare_names_become_unknown_and_inv_only_drops_diatom_rows():
    from sentinel.models.bioassess import bioset_ma as MA

    D = _toy()
    S = D.samples
    rare = {"key": S.index[0], "phylum": "Ip0", "class": "Ic0", "order": "Io0", "family": "If0", "genus": "Rare",
            "name": "Rare one", "mixed_otu": "Rare one", "count": 1e6, "log_class": 5.0}
    D.tokens["invertebrate"] = pd.concat([D.tokens["invertebrate"], pd.DataFrame([rare])], ignore_index=True)
    cfg = _small("ma_inv")
    ids = MA.usable_ids(S, S.index, cfg["modalities"])
    assert len(ids) == int(S["has_inv"].sum())
    feat = MA.MAFeaturizer(cfg).fit(D, ids)
    assert set(feat.prevalence) == {"invertebrate"}
    d = feat.transform(D, ids[:5])
    assert set(d.mod) == {"invertebrate"}
    u = d.mod["invertebrate"].index[0]                                  # token arrays are per unique sample key
    tok0 = d.mod["invertebrate"].lev[u, 0]                              # the rare taxon is the most abundant token
    assert tok0[MA.NAME] == 0 and tok0[4] == 0 and tok0[3] > 0          # name and genus unknown, family kept
    assert d.mod["invertebrate"].feat[u, 0, MA.I_MISS] == 1.0
    with pytest.raises(ValueError):
        feat.transform(D, S.index[~S["has_inv"].to_numpy()][:2])


# ------------------------------------------------------------------------------------------------ model
def _model_and_data(cfg, n=24):
    import torch

    from sentinel.models.bioassess import bioset_ma as MA
    from sentinel.models.bioassess.bioset import LEVELS as L

    torch.manual_seed(0)
    D = _toy()
    ids = D.samples.index[np.r_[0:n - 4, len(D.samples) - 4:len(D.samples)]]      # last four rows are diatom-only
    feat = MA.MAFeaturizer(cfg).fit(D, D.samples.index)
    data = feat.transform(D, MA.usable_ids(D.samples, ids, cfg["modalities"]))
    model = MA.build_ma(cfg, {lv: feat.vocab.size(lv) for lv in L}, feat.n_ctx, len(feat.source_map),
                        {m: len(v) for m, v in feat.proto_map.items()}, 5)
    return model.eval(), data, feat


def test_model_shapes_and_rows_without_a_modality():
    import torch

    from sentinel.models.bioassess import bioset_ma as MA

    cfg = _small("ma_full_rc")
    model, data, _ = _model_and_data(cfg)
    n = len(data)
    b = MA.make_batch(data, np.arange(n), "cpu", group=np.zeros(n, np.int64))
    with torch.no_grad():
        loc, ls, t = model(b)
        vec, ok = model.community(b)
    assert loc.shape == ls.shape == (n, 5) and t.shape == (n, cfg["trunk"])
    assert torch.isfinite(loc).all() and torch.isfinite(ls).all() and torch.isfinite(t).all()
    assert ok[:, 0].tolist() == (data.mod["invertebrate"].index >= 0).tolist()
    assert ok[:, 1].tolist() == (data.mod["diatom"].index >= 0).tolist()
    assert not ok.all() and ok.any(1).all() and not vec[~ok].any()
    assert torch.allclose(ls, torch.zeros_like(ls))                    # log-scale head starts at 0 (unit scale)


def test_set_encoder_is_permutation_and_padding_invariant():
    import torch

    from sentinel.models.bioassess import bioset_ma as MA

    model, data, _ = _model_and_data(_small("ma_full"))
    b = MA.make_batch(data, np.arange(len(data)), "cpu")
    with torch.no_grad():
        ref = model(b)[0]
        lev, feat, mask, rows, mf = b["mod"]["invertebrate"]
        perm = torch.randperm(lev.shape[1], generator=torch.Generator().manual_seed(1))
        b["mod"]["invertebrate"] = (lev[:, perm], feat[:, perm], mask[:, perm], rows, mf)
        out = model(b)[0]
        pad = lambda x: torch.cat([x, torch.zeros_like(x[:, :3])], 1)   # noqa: E731
        b["mod"]["invertebrate"] = (pad(lev), pad(feat), pad(mask), rows, mf)
        out2 = model(b)[0]
    assert torch.allclose(ref, out, atol=1e-5) and torch.allclose(ref, out2, atol=1e-5)


def test_training_augmentations_keep_one_token_and_one_modality():
    from sentinel.models.bioassess import bioset_ma as MA

    cfg = _small("ma_full", modality_dropout=1.0, token_dropout=0.99, source_dropout=1.0)
    _, data, _ = _model_and_data(cfg)
    n = len(data)
    b = MA.make_batch(data, np.arange(n), "cpu", cfg=cfg, rng=np.random.default_rng(0))
    rows = {m: set(b["mod"][m][3].tolist()) for m in b["mod"]}
    had = {m: set(np.flatnonzero(data.mod[m].index >= 0).tolist()) for m in data.mod}
    both = had["invertebrate"] & had["diatom"]
    assert both and not (rows["invertebrate"] & rows["diatom"])         # rows with both keep exactly one
    assert rows["invertebrate"] | rows["diatom"] == set(range(n))       # nobody loses every modality
    for m in b["mod"]:
        lev, feat, mask, r, mf = b["mod"][m]
        assert mask.any(1).all()                                        # token dropout keeps >= 1 token
        assert (feat[..., MA.I_PREV] == 0).all() and (feat[..., MA.I_MISS][mask] == 1).all()
        assert (b["proto"][m] == 0).all()
    assert (b["src"] == 0).all() and (b["src_true"] > 0).all()
    plain = MA.make_batch(data, np.arange(n), "cpu")                    # no rng: no augmentation
    assert {m: set(plain["mod"][m][3].tolist()) for m in plain["mod"]} == had and (plain["src"] > 0).all()


def test_fz_candidates_take_no_coordinates_and_fz_comparators_name_their_input_set():
    """Lead ruling (a): no latitude or longitude in an F-Z candidate; the two comparator input sets are both named."""
    from sentinel.models.bioassess import bioset_ma as MA
    from sentinel.models.bioassess import ftrack as F
    from sentinel.models.bioassess import pfn_hybrid as PH

    full, bare = ("doy_sin", "doy_cos", "lat", "lon", "altitude_m"), ("doy_sin", "doy_cos", "altitude_m")
    assert F.CONTEXT_COLS == MA.CONTEXT_COLS == full and F.CONTEXT_COLS_NO_COORDS == MA.CONTEXT_COLS_NO_COORDS == bare
    for track in ("F-A", "F-T"):
        assert F.context_cols(track) == F.context_cols(track, "comparator") == full
        with pytest.raises(ValueError):
            F.context_cols(track, "candidate", "nocoords")
    assert F.context_cols("F-Z") == F.context_cols("F-Z", "candidate", "nocoords") == bare
    with pytest.raises(F.FitGuardError):
        F.context_cols("F-Z", "candidate", "coords")
    with pytest.raises(ValueError):
        F.context_cols("F-Z", "comparator")                             # no default: the set is named
    assert F.context_cols("F-Z", "comparator", "coords") == full
    assert F.context_cols("F-Z", "comparator", "nocoords") == bare
    assert F.fz_comparator_names("rf_community") == ("rf_community__coords", "rf_community__nocoords")
    zs = MA.recipe("ma_zs")
    assert tuple(zs["context_cols"]) == bare and zs["region_context"]
    assert all(tuple(MA.recipe(r)["context_cols"]) == full for r in MA.RECIPES if r != "ma_zs")
    assert tuple(PH.HYBRID["context_cols"]) == full and tuple(PH.HYBRID["context_cols_fz"]) == bare
    D = _toy()
    f_zs = MA.MAFeaturizer(_small("ma_zs")).fit(D, D.samples.index)
    f_fa = MA.MAFeaturizer(_small("ma_full")).fit(D, D.samples.index)
    assert f_zs.n_ctx == 2 * len(bare) and f_fa.n_ctx == 2 * len(full)
    moved = D.samples.copy()
    moved[["lat", "lon"]] = moved[["lat", "lon"]] + 7.0                 # coordinates do not reach the F-Z inputs
    D2 = type(D)(moved, D.tokens, D.Y, {})
    ids = D.samples.index[:20]
    np.testing.assert_array_equal(f_zs.transform(D, ids).ctx, f_zs.transform(D2, ids).ctx)
    assert not np.array_equal(f_fa.transform(D, ids).ctx, f_fa.transform(D2, ids).ctx)


def test_mirror_rule_sample_is_order_free_and_uses_the_registered_seed():
    """Lead ruling (m): keys sorted in byte order, ``default_rng([20261110, k, s])``, at most 256 rows."""
    from sentinel.models.bioassess import ftrack as F

    assert F.MIRROR_SEED == 20261110 and F.REGION_CONTEXT_MAX == 256
    keys = np.array([f"fr:S{i}:2021-05-0{i % 9 + 1}:13" for i in range(600)], object)
    a = F.mirror_pick(keys, 3, 1)
    assert len(a) == 256 and len(set(a.tolist())) == 256
    order = np.asarray(sorted(range(600), key=lambda i: keys[i].encode("utf-8")))
    expect = order[np.sort(np.random.default_rng([20261110, 3, 1]).choice(600, 256, replace=False))]
    np.testing.assert_array_equal(a, expect)
    perm = np.random.default_rng(5).permutation(600)
    b = F.mirror_pick(keys[perm], 3, 1)
    assert set(keys[perm][b]) == set(keys[a])                           # the row order of the caller does not matter
    np.testing.assert_array_equal(F.mirror_pick(keys, 3, 1), a)
    assert set(F.mirror_pick(keys, 4, 1).tolist()) != set(a.tolist())   # another unit
    assert set(F.mirror_pick(keys, 3, 2).tolist()) != set(a.tolist())   # another model seed
    few = np.array(["b", "a", "\u00e9", "Z", "a0"], object)
    assert list(few[F.mirror_pick(few, 0, 0)]) == ["Z", "a", "a0", "b", "\u00e9"]     # all rows, byte order
    assert len(F.mirror_pick(keys[:256], 9, 9)) == 256 and len(F.mirror_pick(keys[:257], 9, 9)) == 256
    assert len(F.mirror_pick(keys, 0, 0, cap=8)) == 8
    with pytest.raises(ValueError):
        F.mirror_pick(np.array(["a", "a"], object), 0, 0)


def test_batches_are_assembled_on_the_device_with_the_stored_values():
    """Lead ruling (o): batches are gathered from tensors held on the device; the stored tensors are never edited."""
    import torch

    from sentinel.models.bioassess import bioset_ma as MA

    cfg = _small("ma_full")
    _, data, _ = _model_and_data(cfg)
    dev = data.on("cpu")
    assert set(dev) == set(data.mod) | {"ctx"} and data.on("cpu") is dev
    rows = np.array([5, 0, 3, len(data) - 1, 7, len(data) - 2])
    b = MA.make_batch(data, rows, "cpu")
    np.testing.assert_array_equal(b["ctx"].numpy(), data.ctx[rows])
    np.testing.assert_array_equal(b["alt"].numpy(), data.alt[rows].astype(np.float32))
    assert np.isnan(data.alt).any() and np.isfinite(data.alt).any()
    seen = set()
    for m, (lev, feat, mask, loc, _) in b["mod"].items():
        md = data.mod[m]
        u = md.index[rows[loc.numpy()]]
        T = mask.shape[1]
        assert (u >= 0).all() and T == max(1, int(md.mask[u].sum(1).max()))
        np.testing.assert_array_equal(lev.numpy(), md.lev[u, :T])
        np.testing.assert_array_equal(feat.numpy(), md.feat[u, :T])
        np.testing.assert_array_equal(mask.numpy(), md.mask[u, :T])
        seen |= set(loc.tolist())
    assert seen == set(range(len(rows)))
    before = {m: (dev[m][0].clone(), dev[m][1].clone()) for m in data.mod}
    aug = _small("ma_full", source_dropout=1.0, token_dropout=0.5, modality_dropout=0.5)
    MA.make_batch(data, np.arange(len(data)), "cpu", cfg=aug, rng=np.random.default_rng(0))
    for m in data.mod:
        assert torch.equal(before[m][0], dev[m][0]) and torch.equal(before[m][1], dev[m][1])
        np.testing.assert_array_equal(dev[m][1].numpy(), data.mod[m].feat)


def test_region_token_excludes_self_and_other_groups():
    import torch

    from sentinel.models.bioassess import bioset_ma as MA

    model, _, _ = _model_and_data(_small("ma_full_rc"))
    R = model.region
    group = torch.tensor([0, 0, 0, 0, 1, 1, 1, -1, 2])
    A = R.allowed(group)
    assert not A.diagonal().any() and A[0, 1] and not A[0, 4] and not A[7].any() and not A[:, 7].any()
    assert not A[8].any()                                               # a group of one has no context
    h = torch.randn(9, 32, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        r0, has = R(h, group)
        gen = torch.Generator().manual_seed(1)
        h2 = h.clone()
        h2[4] = torch.randn(32, generator=gen)                          # (a uniform shift would vanish in the LayerNorm)
        r1, _ = R(h2, group)
        key_ok = torch.tensor([False, False, True, True, True, True, True, True, True])
        h3 = h.clone()
        h3[0] = torch.randn(32, generator=gen)
        ra, _ = R(h, group, key_ok)
        rb, _ = R(h3, group, key_ok)
    assert has.tolist() == [True] * 7 + [False, False] and not r0[7].any() and not r0[8].any()
    assert torch.allclose(r0[:4], r1[:4]) and not torch.allclose(r0[5], r1[5])    # other groups do not leak
    assert torch.allclose(ra[1:4], rb[1:4])                             # a row that is not a context key is unseen
    assert not torch.allclose(r0[1], R(h3, group)[0][1])                # ... and seen when it is one
    assert torch.isfinite(r0).all()


def test_student_t_nll_matches_scipy_and_masks_missing():
    import torch
    from scipy import stats

    from sentinel.models.bioassess import bioset_ma as MA

    rng = np.random.default_rng(0)
    loc, ls, y = rng.normal(size=(50, 3)), rng.normal(0, 0.5, (50, 3)), rng.normal(size=(50, 3))
    y[::7, 1] = np.nan
    nll, ok = MA.student_t_nll(torch.tensor(loc), torch.tensor(ls), torch.tensor(y), 4.0)
    ref = -stats.t.logpdf(y, df=4.0, loc=loc, scale=np.exp(ls))
    np.testing.assert_allclose(nll.numpy()[np.isfinite(y)], ref[np.isfinite(y)], rtol=1e-9, atol=1e-9)
    assert (nll.numpy()[~np.isfinite(y)] == 0).all() and ok.numpy().tolist() == np.isfinite(y).tolist()
    np.testing.assert_allclose(float(MA.pooled_nll(nll, ok)), np.nanmean(ref), rtol=1e-9)
    np.testing.assert_allclose(MA.column_balanced_nll(nll, ok, [0, 1]), np.nanmean(ref[:, :2], 0).mean(), rtol=1e-9)
    empty = torch.full((4, 2), float("nan"))
    n2, ok2 = MA.student_t_nll(torch.zeros(4, 2), torch.zeros(4, 2), empty)
    assert float(MA.pooled_nll(n2, ok2)) == 0.0 and np.isnan(MA.column_balanced_nll(n2, ok2, [0, 1]))


def test_oxygen_solubility_and_coherence_penalty():
    import torch

    from sentinel.models.bioassess import bioset_ma as MA
    from sentinel.models.bioassess import fdata as FD

    np.testing.assert_allclose(MA.o2_solubility_mg_l(np.array([0.0, 10.0, 20.0, 30.0])), [14.62, 11.29, 9.09, 7.56],
                               atol=0.01)
    cols = list(FD.Y_COLUMNS)
    cons = MA.default_constraints(cols)
    assert len(cons) == 8 and {c["window"] for c in cons} == {"pm30d", "p365"}
    ix = {c: i for i, c in enumerate(cols)}
    P = torch.zeros(6, len(cols), dtype=torch.float64)
    for w in ("pm30d", "p365"):
        P[:, ix[f"log10_tp_mg_l__{w}"]] = 0.1
        P[:, ix[f"log10_ton_mg_l__{w}"]] = 0.1
        P[:, ix[f"log10_nkj_mg_l__{w}"]] = 0.1
        P[:, ix[f"temp_c__{w}"]] = 20.0
        P[:, ix[f"do_sat_pct__{w}"]] = 90.0
        P[:, ix[f"log10_do_mg_l__{w}"]] = float(np.log10(0.9 * MA.o2_solubility_mg_l(np.array(20.0))))
    src = torch.tensor([1, 1, 1, 2, 2, 2])
    smap = {"uk": 1, "fr": 2}
    assert float(MA.coherence_penalty(P, cons, src, smap)) == 0.0
    Q = P.clone()
    Q[:3, ix["log10_po4p_mg_l__pm30d"]] = 0.4                           # PO4-P above total P in half the rows
    np.testing.assert_allclose(float(MA.coherence_penalty(Q, cons, src, smap)), 0.15, atol=1e-9)
    Q = P.clone()
    Q[:, ix["log10_no3n_mg_l__p365"]] = 0.3                             # nitrate above TON: counted on UK rows only
    Q[3:, ix["log10_ton_mg_l__p365"]] = 5.0
    np.testing.assert_allclose(float(MA.coherence_penalty(Q, cons, src, smap)), 0.2, atol=1e-9)
    Q = P.clone()
    Q[:, ix["log10_do_mg_l__pm30d"]] += 0.05                            # inside the tolerance
    assert float(MA.coherence_penalty(Q, cons, src, smap)) == 0.0
    Q[:, ix["log10_do_mg_l__pm30d"]] += 0.11
    np.testing.assert_allclose(float(MA.coherence_penalty(Q, cons, src, smap)), 0.1, atol=1e-9)
    assert MA.default_constraints(["ph__pm30d", "log10_tp_mg_l__pm30d"]) == []


def test_do_coherence_uses_the_altitude_pressure_and_the_shared_solubility_routine():
    """Lead ruling (e): pressure factor (1 - 2.25577e-5 h)^5.25588; missing altitude -> 1 atm; the solubility is
    ``hydromet.physics.do_sat_mgl`` itself; its constants are checked against Benson & Krause (1984)."""
    import torch

    from sentinel.models.bioassess import bioset_ma as MA
    from sentinel.models.bioassess import fdata as FD
    from sentinel.models.hydromet import physics as PHY

    h = np.array([0.0, 150.0, 1000.0, 2500.0, np.nan, -500.0, 9000.0])
    p = MA.altitude_pressure_atm(h)
    np.testing.assert_allclose(p, PHY.pressure_atm_from_elevation(h), rtol=0, atol=0)
    np.testing.assert_allclose(p[:4], (1.0 - 2.25577e-5 * h[:4]) ** 5.25588, rtol=1e-12)
    assert p[0] == 1.0 and p[4] == 1.0 and 0.88 < p[2] < 0.89          # 1000 m: about 0.887 atm
    np.testing.assert_allclose(MA.altitude_pressure_atm(torch.as_tensor(h)).numpy(), p, rtol=1e-12)
    t = np.array([0.0, 5.0, 10.0, 20.0, 30.0, 35.0])
    for pr in (1.0, 0.887):
        np.testing.assert_array_equal(MA.o2_solubility_mg_l(t, pr), PHY.do_sat_mgl(t, pr))
        np.testing.assert_allclose(MA.o2_solubility_mg_l(torch.as_tensor(t), pr).numpy(), PHY.do_sat_mgl(t, pr),
                                   rtol=1e-9)
    # Benson & Krause (1984), fresh water, 1 atm, written out here from the source in its two forms.
    # mg/L form: ln C* = -139.34411 + 1.575701e5/T - 6.642308e7/T^2 + 1.243800e10/T^3 - 8.621949e11/T^4
    T = t + 273.15
    mg_l = np.exp(-139.34411 + 1.575701e5 / T - 6.642308e7 / T ** 2 + 1.243800e10 / T ** 3 - 8.621949e11 / T ** 4)
    np.testing.assert_allclose(MA.o2_solubility_mg_l(t), mg_l, rtol=1e-12)
    # umol/kg form: ln C* = -135.29996 + 1.572288e5/T - 6.637149e7/T^2 + 1.243678e10/T^3 - 8.621061e11/T^4,
    # times the molar mass of O2 and the density of pure water (kg/L)
    umol_kg = np.exp(-135.29996 + 1.572288e5 / T - 6.637149e7 / T ** 2 + 1.243678e10 / T ** 3 - 8.621061e11 / T ** 4)
    rho = 1.0 - (t + 288.9414) / (508929.2 * (t + 68.12963)) * (t - 3.9863) ** 2
    np.testing.assert_allclose(MA.o2_solubility_mg_l(t), umol_kg * 31.9988e-3 * rho, rtol=5e-4)
    # pressure correction of the same source: vapour pressure and theta
    pwv = np.exp(11.8571 - 3840.70 / T - 216961.0 / T ** 2)
    theta = 0.000975 - 1.426e-5 * t + 6.436e-8 * t ** 2
    np.testing.assert_allclose(MA.o2_solubility_mg_l(t, 0.887),
                               mg_l * 0.887 * (1 - pwv / 0.887) * (1 - theta * 0.887) / ((1 - pwv) * (1 - theta)),
                               rtol=1e-12)
    # table values 14.62 / 11.29 / 9.09 / 7.56 mg/L at 0 / 10 / 20 / 30 C
    np.testing.assert_allclose(MA.o2_solubility_mg_l(t[[0, 2, 3, 4]]), [14.62, 11.29, 9.09, 7.56], atol=0.01)
    ratio = MA.o2_solubility_mg_l(t, 0.887) / MA.o2_solubility_mg_l(t)   # a little below P (vapour pressure)
    assert (ratio < 0.887).all() and (ratio > 0.878).all() and (np.diff(ratio) < 0).all()

    cols = list(FD.Y_COLUMNS)
    cons = [c for c in MA.default_constraints(cols) if c["kind"] == "do"]
    assert len(cons) == 2 and all(c["tol"] == 0.06 for c in cons)
    ix = {c: i for i, c in enumerate(cols)}
    alt = torch.tensor([0.0, 2000.0, 2000.0, float("nan")], dtype=torch.float64)
    press = MA.altitude_pressure_atm(alt)
    P = torch.zeros(4, len(cols), dtype=torch.float64)
    four = torch.ones(4, dtype=torch.float64)
    for w in ("pm30d", "p365"):
        P[:, ix[f"temp_c__{w}"]] = 12.0
        P[:, ix[f"do_sat_pct__{w}"]] = 100.0
        P[:, ix[f"log10_do_mg_l__{w}"]] = torch.log10(MA.o2_solubility_mg_l(12.0 * four, press))
    assert float(MA.coherence_penalty(P, cons, alt=alt)) == 0.0         # coherent at each row's own pressure
    # without the pressure factor the two rows at 2,000 m are off by log10(1 / 0.785) = 0.105 > 0.06
    miss = float(MA.coherence_penalty(P, cons))
    t12 = torch.full((1,), 12.0, dtype=torch.float64)
    gap = float(torch.log10(MA.o2_solubility_mg_l(t12) / MA.o2_solubility_mg_l(t12, press[1]))) - 0.06
    assert gap > 0.04
    np.testing.assert_allclose(miss, 2 * (2 * gap / 4), rtol=1e-9)      # two windows x two of four rows
    nan_alt = torch.full((4,), float("nan"), dtype=torch.float64)
    np.testing.assert_allclose(float(MA.coherence_penalty(P, cons, alt=nan_alt)), miss, rtol=1e-12)
    # clamps of the ruling: temperature 0-35 C, saturation 1-300 %
    Q = P.clone()
    Q[:, ix["temp_c__pm30d"]], Q[:, ix["do_sat_pct__pm30d"]] = 60.0, 900.0
    Q[:, ix["log10_do_mg_l__pm30d"]] = torch.log10(3.0 * MA.o2_solubility_mg_l(35.0 * four, press))
    assert float(MA.coherence_penalty(Q, cons, alt=alt)) < 1e-9


def test_fit_counts_training_rows_without_altitude_for_the_do_term():
    D = _toy()
    fitted, feat, pool, fit_ids, _ = _fit_toy(D, _small("ma_full_coh", max_epochs=1))
    rec = fitted.info["do_coherence"]
    n_miss = int(D.samples.loc[fit_ids, "altitude_m"].isna().sum())
    assert n_miss > 0 and rec["n_altitude_missing"] == n_miss and rec["n_train_rows"] == len(fit_ids)
    np.testing.assert_array_equal(np.isnan(feat.transform(D, fit_ids).alt),
                                  D.samples.loc[fit_ids, "altitude_m"].isna().to_numpy())
    plain = _fit_toy(D, _small("ma_full", max_epochs=1))[0]
    assert "do_coherence" not in plain.info                             # no coherence term: nothing to count


def test_nitrogen_and_phosphorus_constraints_compare_like_units():
    """Lead ruling (l): each >= constraint compares two labels in the same unit (mg N/L or mg P/L)."""
    import torch

    from sentinel.models.bioassess import bioset_ma as MA
    from sentinel.models.bioassess import fdata as FD

    N, Ox, H, Pm = 14.0067, 15.9994, 1.00794, 30.973762                 # standard atomic weights
    exact = {"no3_to_n": N / (N + 3 * Ox), "nh4_to_n": N / (N + 4 * H), "po4_to_p": Pm / (Pm + 4 * Ox),
             "no2_to_n": N / (N + 2 * Ox)}
    assert set(FD.FR_FACTORS) == set(exact)
    for k, v in exact.items():
        assert abs(FD.FR_FACTORS[k] - v) < 5e-5, k
    element = {"log10_tp_mg_l": "P", "log10_po4p_mg_l": "P", "log10_ton_mg_l": "N", "log10_no3n_mg_l": "N",
               "log10_nkj_mg_l": "N", "log10_nh4n_mg_l": "N"}
    uk_name = {"log10_tp_mg_l": "total_p", "log10_po4p_mg_l": "orthophosphate_p", "log10_ton_mg_l": "ton_n",
               "log10_no3n_mg_l": "nitrate_n", "log10_nh4n_mg_l": "ammoniacal_n", "log10_nkj_mg_l": None}
    for t, name in uk_name.items():
        assert FD.TARGETS[t][0] == name and FD.TARGETS[t][1] == "log10"
        assert name is None or name.endswith("_" + element[t].lower())   # UK determinands are "as N" / "as P"
    cols = list(FD.Y_COLUMNS)
    cons = [c for c in MA.default_constraints(cols) if c["kind"] == "ge"]
    pairs = {(cols[c["hi"]].split("__")[0], cols[c["lo"]].split("__")[0], c["sources"]) for c in cons}
    assert pairs == {("log10_tp_mg_l", "log10_po4p_mg_l", None), ("log10_ton_mg_l", "log10_no3n_mg_l", ("uk", "ukd")),
                     ("log10_nkj_mg_l", "log10_nh4n_mg_l", ("fr",))}
    for hi, lo, _ in pairs:
        assert element[hi] == element[lo]
    for c in cons:
        assert cols[c["hi"]].split("__")[1] == cols[c["lo"]].split("__")[1] == c["window"]
    # a water with total P 0.2 mg P/L and orthophosphate 0.5 mg PO4/L (= 0.163 mg P/L) is coherent in mg P/L;
    # the same numbers compared without the molar factor would be flagged
    ix = {c: i for i, c in enumerate(cols)}
    tp = [c for c in cons if cols[c["hi"]] == "log10_tp_mg_l__pm30d"]
    P = torch.zeros(3, len(cols), dtype=torch.float64)
    P[:, ix["log10_tp_mg_l__pm30d"]] = float(np.log10(0.2))
    P[:, ix["log10_po4p_mg_l__pm30d"]] = float(np.log10(0.5 * FD.FR_FACTORS["po4_to_p"]))
    assert float(MA.coherence_penalty(P, tp)) == 0.0
    P[:, ix["log10_po4p_mg_l__pm30d"]] = float(np.log10(0.5))
    np.testing.assert_allclose(float(MA.coherence_penalty(P, tp)), np.log10(0.5 / 0.2), rtol=1e-9)
    # French class limits in the label units (protocol section 7): 50 mg NO3/L, 0.5 mg PO4/L, 0.5 mg NH4/L
    np.testing.assert_allclose([50 * FD.FR_FACTORS["no3_to_n"], 0.5 * FD.FR_FACTORS["po4_to_p"],
                                0.5 * FD.FR_FACTORS["nh4_to_n"]], [11.295, 0.16305, 0.38825], rtol=1e-12)


def test_loss_gradients_reach_every_part_with_region_context_and_coherence():
    import torch

    from sentinel.models.bioassess import bioset_ma as MA

    cfg = _small("ma_full_rc", coherence_weight=0.1)
    model, data, feat = _model_and_data(cfg)
    model.train()
    n = len(data)
    b = MA.make_batch(data, np.arange(n), "cpu", cfg=cfg, rng=np.random.default_rng(0),
                      group=np.arange(n) % 2)
    y = torch.randn(n, 5)
    y[::3, 0] = float("nan")
    y[:, 4] = float("nan")                                              # a target without any label
    loc, ls, _ = model(b)
    nll, ok = MA.student_t_nll(loc, ls.clamp(*MA.LOG_SCALE_RANGE), y)
    cons = [{"kind": "ge", "hi": 0, "lo": 1, "sources": None}, {"kind": "do", "do": 2, "sat": 3, "temp": 4,
                                                                 "tol": 0.06, "sources": None}]
    loss = MA.pooled_nll(nll, ok) + 0.1 * MA.coherence_penalty(loc, cons, b["src_true"], feat.source_map)
    loss.backward()
    assert torch.isfinite(loss)
    g = {k: p.grad for k, p in model.named_parameters()}
    for part in ("encoder.emb.name.weight", "encoder.emb.lineage.family.weight", "encoder.emb.feat.weight",
                 "encoder.blocks.0.att.in_proj_weight", "encoder.pma.in_proj_weight", "mod_proj.diatom.weight",
                 "ctx.weight", "src.weight", "region.att.in_proj_weight", "fusion.att.in_proj_weight",
                 "trunk.0.weight", "loc.weight", "log_scale.weight", "src_scale.weight"):
        assert g[part] is not None and torch.isfinite(g[part]).all() and float(g[part].abs().sum()) > 0, part


def test_epoch_batches_use_each_row_once():
    from sentinel.models.bioassess import bioset_ma as MA

    rng = np.random.default_rng(0)
    n = 1000
    plain = list(MA.epoch_batches(n, 256, rng))
    assert sorted(np.concatenate([r for r, _ in plain]).tolist()) == list(range(n)) and all(g is None for _, g in plain)
    groups = rng.choice(7, n, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02])
    groups[0] = 99                                                      # a pseudo-region with a single row
    ep = list(MA.epoch_batches(n, 256, rng, groups, 64))
    assert sorted(np.concatenate([r for r, _ in ep]).tolist()) == list(range(n))
    for rows, grp in ep:
        assert len(rows) == len(grp) <= 256
        for k in np.unique(grp):
            assert (grp == k).sum() <= 64 and len(set(groups[rows[grp == k]])) == 1
    assert len(ep) < n / 64                                             # chunks are packed into batches


def test_pseudo_regions_and_cells():
    from sentinel.models.bioassess import bioset_ma as MA

    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(42, 51, 500), rng.uniform(-4, 8, 500)
    a, b = MA.pseudo_regions(lat, lon, np.random.default_rng(1)), MA.pseudo_regions(lat, lon, np.random.default_rng(2))
    assert a.shape == (500,) and a.min() == 0 and not np.array_equal(a, b)
    c = MA.cell_codes(lat, lon, 50.0)
    _, _, clat, clon = MA.grid_cells(lat, lon, 50.0)
    assert np.abs(clat - lat).max() <= 0.5 * 50 / 111.19 + 1e-6 and len(np.unique(c)) > 50


def test_inner_split_holds_out_blocks_and_keeps_clusters_together():
    from sentinel.models.bioassess import bioset_ma as MA

    S = _toy().samples
    fit, val = MA.inner_split(S, S.index, frac=0.15, seed=0)
    fit2, val2 = MA.inner_split(S, S.index, frac=0.15, seed=0)
    assert list(fit) == list(fit2) and list(val) == list(val2)
    assert len(fit) and len(val) and len(fit) + len(val) == len(S) and not set(fit) & set(val)
    assert not set(S.loc[fit, "location_id"]) & set(S.loc[val, "location_id"])
    for c in ("uk", "fr"):
        m = S["source"].str[:2] == c
        share = S.loc[val, "location_id"][m.reindex(val).to_numpy()].nunique() / S.loc[m, "location_id"].nunique()
        assert 0.05 <= share <= 0.5
    one = S.iloc[:20].assign(lat=50.0, lon=1.0, source="fr")            # a single block: clusters are held out
    f1, v1 = MA.inner_split(one, one.index, frac=0.2, seed=0)
    assert len(v1) >= 1 and len(f1) >= 1 and not set(one.loc[f1, "location_id"]) & set(one.loc[v1, "location_id"])


# ------------------------------------------------------------------------------------------------ fit
def test_fit_ma_runs_and_is_deterministic():
    from sentinel.models.bioassess import bioset_ma as MA
    from sentinel.models.bioassess import fdata as FD
    from sentinel.models.bioassess import ftrack as F

    D = _toy()
    S = D.samples
    cfg = _small("ma_full")
    fitted, feat, pool, fit_ids, val_ids = _fit_toy(D, cfg)
    te = F.test_ids(S, "G1", "dev")
    assert len(te) and not set(te) & set(pool)
    q = feat.transform(D, te)
    p = fitted.predict(q)
    assert p["loc"].shape == p["scale"].shape == (len(te), len(FD.Y_COLUMNS))
    assert np.isfinite(p["loc"]).all() and (p["scale"] > 0).all() and p["nu"] == 4.0
    assert len(fitted.history) <= cfg["max_epochs"] and fitted.columns == list(FD.Y_COLUMNS) and fitted.n_params > 0
    again = _fit_toy(D, cfg)[0].predict(q)
    np.testing.assert_allclose(p["loc"], again["loc"], atol=1e-5)
    tr = feat.transform(D, fit_ids)
    fitted.fit_embedding(tr)
    e = fitted.embed_low(q)
    assert e.shape == (len(te), cfg["embed_dim"]) and np.isfinite(e).all()
    etr = fitted.embed_low(tr)
    np.testing.assert_allclose(etr.mean(0), 0.0, atol=1e-3)             # PCA centred on training rows


def test_fit_ma_region_context_coherence_and_invertebrate_only_recipes():
    from sentinel.models.bioassess import bioset_ma as MA
    from sentinel.models.bioassess import fdata as FD
    from sentinel.models.bioassess import ftrack as F

    D = _toy()
    S = D.samples
    te = F.test_ids(S, "G1", "dev")
    for name, over in (("ma_full_rc", {"region_max": 8, "episode_rows": 16}), ("ma_full_coh", {}), ("ma_inv", {})):
        cfg = _small(name, max_epochs=2, **over)
        fitted, feat, pool, _, _ = _fit_toy(D, cfg)
        q = feat.transform(D, te)
        region = np.array(["G1"] * len(te), object)
        p = fitted.predict(q, region=region, seed=0)
        assert p["loc"].shape == (len(te), len(FD.Y_COLUMNS)) and np.isfinite(p["loc"]).all(), name
        if name == "ma_full_rc":
            assert len(te) > cfg["region_max"]                          # the sampled-context path is exercised
            np.testing.assert_allclose(p["loc"], fitted.predict(q, region=region, seed=0)["loc"], atol=1e-6)
            none = fitted.predict(q)["loc"]                             # no region labels: token absent, still finite
            assert np.isfinite(none).all() and not np.allclose(none, p["loc"])
            mixed = region.copy()
            mixed[:3] = None
            assert np.isfinite(fitted.predict(q, region=mixed)["loc"]).all()
        if name == "ma_inv":
            assert S.loc[pool, "has_inv"].all() and set(feat.transform(D, te).mod) == {"invertebrate"}
        if name == "ma_full_coh":
            assert len(MA.default_constraints(fitted.columns, cfg["do_tol"])) == 8


def _fit_parts(D, cfg, g="G1"):
    from sentinel.models.bioassess import bioset_ma as MA
    from sentinel.models.bioassess import fdata as FD
    from sentinel.models.bioassess import ftrack as F

    S = D.samples
    pool = MA.usable_ids(S, F.training_ids(S, "F-A", g, "dev"), cfg["modalities"])
    F.assert_fit_ids(pool, S, "F-A", g, "dev")
    fit_ids, val_ids = MA.inner_split(S, pool, seed=0)
    feat = MA.MAFeaturizer(cfg).fit(D, pool)
    cols = list(FD.Y_COLUMNS)
    args = (cfg, feat, feat.transform(D, fit_ids), D.Y.loc[fit_ids, cols].to_numpy(float),
            feat.transform(D, val_ids), D.Y.loc[val_ids, cols].to_numpy(float))
    return args, {"columns": cols, "primary": FD.PRIMARY["F-A"]}, pool


def test_chunked_fit_equals_the_uninterrupted_fit_and_resume_points_pass_the_guard(tmp_path):
    """Lead ruling (o): a fit can stop and go on; on CPU the result equals the uninterrupted fit."""
    from sentinel.models.bioassess import bioset_ma as MA
    from sentinel.models.bioassess import ftrack as F

    D = _toy()
    S = D.samples
    te = F.test_ids(S, "G1", "dev")
    for name, over in (("ma_full_rc", {"region_max": 8, "episode_rows": 16, "coherence_weight": 0.1}),
                       ("ma_full", {"patience": 2, "max_epochs": 6})):
        cfg = _small(name, **{"max_epochs": 4, "patience": 4, **over})
        args, kw, pool = _fit_parts(D, cfg)
        full = MA.fit_ma(*args, seed=3, **kw)
        state, fitted, n = None, None, 0
        while fitted is None:
            fitted, state = MA.fit_ma_chunk(*args, seed=3, state=state, chunk_epochs=1, **kw)
            n += 1
            assert state["epoch"] == n and state["done"] == (fitted is not None)
            if fitted is None and n == 2:                               # through the disk and the guarded loader
                ck = tmp_path / f"{name}_G1_seed3.resume.pt"
                side = MA.save_fit_state(ck, state, pool, track="F-A", g="G1", design="dev")
                assert side.exists()
                with pytest.raises(F.FitGuardError):
                    MA.load_fit_state(ck, S, "F-A", "G2", "dev")
                wrong = dict(state)
                with pytest.raises(ValueError):
                    MA.fit_ma_chunk(*args, seed=4, state=wrong, chunk_epochs=1, **kw)      # another seed
                state = MA.load_fit_state(ck, S, "F-A", "G1", "dev")
                assert "f_guard" not in state
        assert n == len(full.history) == len(fitted.history) and n >= 3, name
        assert fitted.best_epoch == full.best_epoch and fitted.info["chunks"] == [
            {"start_epoch": k, "end_epoch": k + 1} for k in range(n)]
        assert full.info["chunks"] == [{"start_epoch": 0, "end_epoch": n}]
        np.testing.assert_allclose(np.asarray(fitted.history), np.asarray(full.history), rtol=0, atol=1e-6)
        q = args[1].transform(D, te)
        region = np.array(["G1"] * len(te), object)
        a, b = full.predict(q, region=region, seed=3), fitted.predict(q, region=region, seed=3)
        np.testing.assert_allclose(a["loc"], b["loc"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(a["scale"], b["scale"], rtol=0, atol=1e-6)
        with pytest.raises(ValueError):
            MA.fit_ma_chunk(*args, seed=3, state=state, **kw)           # already ended
        two, st = MA.fit_ma_chunk(*args, seed=3, chunk_epochs=2, **kw)  # a chunk of two epochs, then to the end
        assert two is None and st["epoch"] == 2
        rest, st = MA.fit_ma_chunk(*args, seed=3, state=st, **kw)
        np.testing.assert_allclose(rest.predict(q, region=region, seed=3)["loc"], a["loc"], rtol=0, atol=1e-6)
    with pytest.raises(ValueError):
        MA.fit_ma_chunk(*args, seed=3, chunk_epochs=0, **kw)


def test_scored_region_context_uses_the_unit_index_of_the_split_file():
    from sentinel.models.bioassess import ftrack as F

    D = _toy()
    S = D.samples
    cfg = _small("ma_full_rc", max_epochs=1, region_max=8, episode_rows=16)
    fitted, feat, _, _, _ = _fit_toy(D, cfg)
    te = F.test_ids(S, "G1", "dev")
    q = feat.transform(D, te)
    region = np.array(["G1"] * len(te), object)
    assert len(te) > cfg["region_max"]
    base = fitted.predict(q, region=region, seed=0)["loc"]
    same = fitted.predict(q, region=region, seed=0, region_index={"G1": 0})["loc"]
    np.testing.assert_allclose(same, base, atol=1e-6)
    other = fitted.predict(q, region=region, seed=0, region_index={"G1": 5})["loc"]
    assert np.isfinite(other).all() and not np.allclose(other, base)    # another unit index: another sample
    pick = F.mirror_pick(q.ids, 5, 0, cfg["region_max"])
    assert len(pick) == cfg["region_max"]
    with pytest.raises(KeyError):
        fitted.predict(q, region=region, seed=0, region_index={})


def test_checkpoint_roundtrip_only_through_the_guard(tmp_path):
    from sentinel.models.bioassess import bioset_ma as MA
    from sentinel.models.bioassess import ftrack as F

    D = _toy()
    S = D.samples
    cfg = _small("ma_full", max_epochs=1)
    fitted, feat, pool, fit_ids, _ = _fit_toy(D, cfg)
    fitted.fit_embedding(feat.transform(D, fit_ids))
    ck = tmp_path / "ma_full_G1_seed0.pt"
    side = MA.save_checkpoint(ck, fitted, feat, pool, track="F-A", g="G1", design="dev")
    assert side.exists() and str(side).endswith(F.SIDECAR_SUFFIX)
    f2, feat2 = MA.load_fitted(ck, S, "F-A", "G1", "dev")
    assert f2.info == fitted.info and f2.info["chunks"] == [{"start_epoch": 0, "end_epoch": 1}]
    te = F.test_ids(S, "G1", "dev")
    np.testing.assert_allclose(fitted.predict(feat.transform(D, te))["loc"], f2.predict(feat2.transform(D, te))["loc"],
                               atol=1e-6)
    np.testing.assert_allclose(fitted.embed_low(feat.transform(D, te)), f2.embed_low(feat2.transform(D, te)), atol=1e-5)
    with pytest.raises(F.FitGuardError):
        MA.load_fitted(ck, S, "F-A", "G2", "dev")                       # G1's pool holds rows of fold G2
    side.unlink()
    with pytest.raises(F.FitGuardError):
        MA.load_fitted(ck, S, "F-A", "G1", "dev")


# ------------------------------------------------------------------------------------------------ events (F-E)
def test_french_events_are_v1_event_defs_with_the_protocol_thresholds():
    from sentinel.models.bioassess import fdata as FD
    from sentinel.models.bioassess import fevents as FE

    EV = FE.events_module()
    assert EV.__version__ == "1.0"
    tab = FE.table().set_index("code")
    want = {"E2a_lab": ("log10_chloride_mg_l", "above", 230.0), "E5b": ("log10_tp_mg_l", "above", 0.1),
            "E6_snap": ("ph", "outside", (6.5, 9.0)), "F_NO3": ("log10_no3n_mg_l", "above", 11.295),
            "F_PO4": ("log10_po4p_mg_l", "above", 0.16305), "F_NH4": ("log10_nh4n_mg_l", "above", 0.38825),
            "F_TP": ("log10_tp_mg_l", "above", 0.2), "F_PH": ("ph", "outside", (6.0, 9.0)),
            "F_DOSAT": ("do_sat_pct", "below", 70.0), "F_BOD": ("log10_bod5_mg_l", "above", 6.0)}
    assert list(tab.index) == list(want) == list(FE.CODES)
    for code, (target, direction, thr) in want.items():
        ev = FE.event_def(code)
        assert isinstance(ev, EV.EventDef) and ev.code == code and ev.label_source == "measurement"
        assert FE.column(code) == f"{target}__pm30d" and FE.column(code) in FD.Y_COLUMNS
        assert ev.direction == direction == tab.loc[code, "direction"]
        np.testing.assert_allclose(np.asarray(ev.threshold, float), np.asarray(thr, float), rtol=1e-12)
        assert (code in EV.EVENTS) == (code in FE.REUSED) == bool(tab.loc[code, "reused_v1"])   # v1.0 is not edited
        assert (FE.TRACK[code][1] == "log10") == target.startswith("log10_")
    assert FE.event_def("F_NO3").unit == "mg N/L" and FE.event_def("F_PO4").unit == "mg P/L"
    with pytest.raises(KeyError):
        FE.event_def("E3a_turb")
    lab = FE.labels(np.log10([11.295, 11.2951, 5.0, np.nan]), "F_NO3")
    np.testing.assert_array_equal(lab, [0.0, 1.0, 0.0, np.nan])
    np.testing.assert_array_equal(FE.labels(np.log10([0.2, 0.21, 0.11]), "F_TP"), [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(FE.labels(np.log10([0.2, 0.21, 0.11]), "E5b"), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(FE.labels([5.9, 6.0, 6.2, 9.0, 9.1], "F_PH"), [1.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(FE.labels([5.9, 6.0, 6.2, 9.0, 9.1], "E6_snap"), [1.0, 1.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(FE.labels([69.9, 70.0, 110.0], "F_DOSAT"), [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(FE.labels(np.log10([6.0, 6.5]), "F_BOD"), [0.0, 1.0])


def test_event_probabilities_power_rule_and_development_only_recalibration():
    from scipy.stats import norm
    from scipy.stats import t as student_t

    from sentinel.models.bioassess import fevents as FE

    rng = np.random.default_rng(0)
    loc = np.r_[np.linspace(-1.5, 2.5, 801), np.nan]
    sc = np.r_[rng.uniform(0.05, 0.6, 801), 0.3]
    thr = np.log10(FE.LIMITS["F_NO3"])
    g = FE.derived_probability(loc, sc, "F_NO3")
    np.testing.assert_allclose(g[:-1], norm.sf((thr - loc[:-1]) / sc[:-1]), atol=1e-9)
    assert np.isnan(g[-1])
    p = FE.derived_probability(loc, sc, "F_NO3", nu=4.0, chunk=97)      # Student-t through the quantile route
    exact = student_t.sf((thr - loc[:-1]) / sc[:-1], 4.0)
    assert np.isnan(p[-1]) and np.abs(p[:-1] - exact).max() < 0.002
    np.testing.assert_allclose(FE.derived_probability(np.array([thr]), 0.2, "F_NO3", nu=4.0), 0.5, atol=1e-6)
    ph = np.linspace(4.5, 10.5, 301)
    out = FE.derived_probability(ph, 0.4, "F_PH", nu=4.0)
    exact_ph = student_t.cdf((6.0 - ph) / 0.4, 4.0) + student_t.sf((9.0 - ph) / 0.4, 4.0)
    np.testing.assert_allclose(out, exact_ph, atol=0.004)
    low = FE.derived_probability(np.array([40.0, 70.0, 100.0]), 10.0, "F_DOSAT", nu=4.0)
    assert low[0] > 0.95 and abs(low[1] - 0.5) < 1e-6 and low[2] < 0.05
    assert np.isnan(FE.derived_probability(np.array([1.0]), np.array([0.0]), "F_NO3", nu=4.0)).all()
    two = FE.derived_probability(np.tile(loc[:6], (2, 1)), np.tile(sc[:6], (2, 1)), "F_NO3", nu=4.0)
    assert two.shape == (2, 6) and np.allclose(two[0], p[:6])

    y = np.r_[np.ones(29), np.zeros(200)]
    cl = np.r_[np.arange(29) % 12, np.arange(200)]
    assert FE.power(y, cl)["flag"] == "underpowered" and not FE.power(y, cl)["evaluated"]         # 29 positives
    y[29] = 1.0
    cl9 = np.r_[np.arange(30) % 9, np.arange(199)]
    assert FE.power(y, cl9)["n_clusters_pos"] == 9 and not FE.power(y, cl9)["evaluated"]          # 9 clusters
    cl10 = np.r_[np.arange(30) % 10, np.arange(199)]
    ok = FE.power(y, cl10)
    assert ok == {"n": 229, "n_pos": 30, "n_clusters_pos": 10, "flag": "feasible", "evaluated": True}

    raw = rng.uniform(0.01, 0.99, 600)
    evt = (rng.random(600) < raw ** 2).astype(float)                    # raw probabilities are too high
    fold = np.array([f"G{k % 4 + 1}" for k in range(600)], object)
    cf = FE.crossfit_calibrated(raw, evt, fold)
    assert np.isfinite(cf).all() and ((cf >= 0) & (cf <= 1)).all()
    assert abs(cf.mean() - evt.mean()) < abs(raw.mean() - evt.mean())
    new = np.r_[np.linspace(0.02, 0.98, 25), np.nan]
    fz = FE.frozen_calibrated(raw, evt, new)
    assert np.isnan(fz[-1]) and (np.diff(fz[:-1]) > 0).all()
    np.testing.assert_array_equal(fz, FE.frozen_calibrated(raw, evt, new))
    one = FE.crossfit_calibrated(raw[:8], np.zeros(8), fold[:8])        # one class only: left as it is
    np.testing.assert_array_equal(one, raw[:8])


# ------------------------------------------------------------------------------------------------ hybrid
def test_hellinger_featurizer_top_k_and_missing_modality():
    from sentinel.models.bioassess import pfn_hybrid as PH

    D = _toy()
    S = D.samples
    tr = S.index[::2]
    hz = PH.HellingerFeaturizer(k_taxa={"invertebrate": 8, "diatom": 5}, min_prevalence=3).fit(D, tr)
    assert len(hz.taxa["invertebrate"]) == 8 and len(hz.taxa["diatom"]) == 5 and len(hz.columns) == 13
    ti = D.tokens["invertebrate"]
    prev = ti[ti["key"].isin(set(S.loc[tr, "inv_key"].dropna()))].groupby("mixed_otu")["key"].nunique()
    assert prev[hz.taxa["invertebrate"]].min() >= prev.drop(hz.taxa["invertebrate"]).max()
    X = hz.transform(D, S.index)
    assert X.shape == (len(S), 13) and X.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(X[:, :8]).all(1), ~S["has_inv"].to_numpy())
    np.testing.assert_array_equal(np.isnan(X[:, 8:]).all(1), S["diatom_key"].isna().to_numpy())
    i = 0
    t0 = ti[ti["key"] == S["inv_key"].iloc[i]]
    want = [np.sqrt(t0.loc[t0["mixed_otu"] == x, "count"].sum() / t0["count"].sum()) for x in hz.taxa["invertebrate"]]
    np.testing.assert_allclose(X[i, :8], want, atol=1e-6)
    assert PH.assemble(X, np.zeros((len(S), 4)), np.zeros((len(S), 2))).shape == (len(S), 19)


def test_group_queries_every_row_once_and_bounded():
    from sentinel.models.bioassess import pfn_hybrid as PH

    rng = np.random.default_rng(0)
    E = np.concatenate([rng.normal(0, 1, (900, 4)), rng.normal(6, 0.2, (150, 4))])
    lab = PH.group_queries(E, 200, seed=0)
    assert lab.shape == (1050,) and np.bincount(lab).max() <= 200 and np.bincount(lab).min() >= 1
    assert set(lab) == set(range(lab.max() + 1)) and np.array_equal(lab, PH.group_queries(E, 200, seed=0))
    assert (PH.group_queries(E[:100], 200) == 0).all()
    same = PH.group_queries(np.zeros((450, 3)), 200)                    # identical points still split
    assert np.bincount(same).max() <= 200


def test_retrieve_context_budget_labels_and_controls():
    from sentinel.models.bioassess import pfn_hybrid as PH

    rng = np.random.default_rng(0)
    E = rng.normal(size=(2000, 3))
    labelled = rng.random(2000) < 0.8
    strata = rng.choice(["uk|A", "uk|B", "fr|C"], 2000, p=[0.6, 0.3, 0.1])
    c = np.array([2.0, 0.0, 0.0])
    ctx = PH.retrieve_context(E, labelled, c, strata, 400, seed=0)
    assert len(ctx) == 400 == len(set(ctx)) and labelled[ctx].all() and (np.diff(ctx) > 0).all()
    d = ((E - c) ** 2).sum(1)
    near = np.flatnonzero(labelled)[np.argsort(d[labelled], kind="stable")[:200]]
    assert set(near) <= set(ctx)                                        # half the budget = the nearest labelled rows
    rest = np.setdiff1d(ctx, near)
    share = np.mean(strata[rest] == "fr|C")
    pool = np.setdiff1d(np.flatnonzero(labelled), near)
    assert abs(share - np.mean(strata[pool] == "fr|C")) < 0.02          # the other half is stratified
    knn = PH.retrieve_context(E, labelled, c, strata, 400, seed=0, how="knn")
    assert set(knn) == set(np.flatnonzero(labelled)[np.argsort(d[labelled], kind="stable")[:400]])
    rnd = PH.retrieve_context(E, labelled, c, strata, 400, seed=0, how="random")
    assert len(rnd) == 400 and labelled[rnd].all() and d[rnd].mean() > d[ctx].mean()
    small = PH.retrieve_context(E, labelled, c, strata, 5000, seed=0)
    assert np.array_equal(small, np.flatnonzero(labelled))              # everything fits: all labelled rows
    with pytest.raises(ValueError):
        PH.retrieve_context(E, labelled, c, strata, 10, 0, how="other")


def test_hybrid_predict_with_a_stub_regressor():
    from sentinel.models.bioassess import pfn_hybrid as PH

    rng = np.random.default_rng(0)
    Xtr, Etr = rng.normal(size=(1500, 6)).astype(np.float32), rng.normal(size=(1500, 4))
    y = rng.normal(size=1500)
    y[::5] = np.nan
    strata = rng.choice(["a", "b"], 1500)
    Xq, Eq = rng.normal(size=(700, 6)).astype(np.float32), rng.normal(size=(700, 4))
    calls = []

    def stub(Xa, ya, Xb, seed):
        assert np.isfinite(ya).all() and len(Xa) == len(ya) <= 300 and Xa.shape[1] == Xb.shape[1]
        calls.append((len(ya), len(Xb)))
        return np.full(len(Xb), ya.mean())

    pred, info = PH.hybrid_predict(Xtr, Etr, y, strata, Xq, Eq, budget=300, group_rows=250, seed=0, regressor=stub,
                                   return_info=True)
    assert pred.shape == (700,) and np.isfinite(pred).all()
    assert sum(q for _, q in calls) == 700 and max(q for _, q in calls) <= 250 and len(info) == len(calls) >= 3
    assert all(i["n_context"] == 300 for i in info)
    again = PH.hybrid_predict(Xtr, Etr, y, strata, Xq, Eq, budget=300, group_rows=250, seed=0, regressor=stub)
    np.testing.assert_array_equal(pred, again)
    none = PH.hybrid_predict(Xtr, Etr, np.full(1500, np.nan), strata, Xq, Eq, regressor=stub)
    assert np.isnan(none).all()


def test_hybrid_default_regressor_is_the_comparator_tabpfn_call(monkeypatch):
    from sentinel.models.bioassess import pfn_hybrid as PH
    from sentinel.models.bioassess import xdev as XD

    seen = []

    def fake(kind, Xtr, y, Xte, seed, n_jobs, device="cuda"):
        seen.append((kind, Xtr.shape, device))
        return np.zeros(len(Xte))

    monkeypatch.setattr(XD, "_fit_learner", fake)
    rng = np.random.default_rng(0)
    out = PH.hybrid_predict(rng.normal(size=(50, 3)), rng.normal(size=(50, 2)), rng.normal(size=50),
                            np.array(["a"] * 50), rng.normal(size=(10, 3)), rng.normal(size=(10, 2)),
                            regressor=PH.tabpfn_regressor(device="cpu"))
    assert out.shape == (10,) and seen == [("tabpfn", (50, 3), "cpu")]
    assert PH.HYBRID["budget"] == XD.TABPFN_MAX_ROWS == XD.TABPFN_PARAMS["max_train_rows"]


# ------------------------------------------------------------------------------------------------ UK allowed rows
def _have_uk() -> bool:
    from sentinel.core import paths

    return (paths.PROCESSED / "uk_ea_ecology" / "samples_diatom.parquet").is_file() and \
        (paths.PROCESSED / "taxonomy_harmonised" / "taxon_map.parquet").is_file()


@pytest.fixture(scope="module")
def uk_allowed():
    from sentinel.models.bioassess import ftrack as F

    return F.uk_allowed()


@pytest.mark.skipif(not _have_uk(), reason="processed UK EA ecology tables not on this machine")
def test_uk_allowed_rows_metadata(uk_allowed):
    """Metadata only (dates, roles, methods): no label value is read."""
    from sentinel.core import paths
    from sentinel.models.bioassess import ftrack as F
    from sentinel.models.bioassess import uk as UK
    from sentinel.models.bioassess import xdev as XD

    P = paths.PROCESSED / "uk_ea_ecology"
    A = uk_allowed
    assert A["invertebrate"] and A["diatom"] and A["counts"]["reserved_sites"] > 0
    si = pd.read_parquet(P / "samples_invertebrate.parquet", columns=["site_id", "analysis_id", "sample_date"])
    si = si[si["analysis_id"].astype(str).isin(A["invertebrate"])]
    assert pd.to_datetime(si["sample_date"]).dt.date.max() <= F.UK_ALLOWED_TO == dt.date(2020, 12, 1)
    sites = pd.read_parquet(P / "sites.parquet", columns=["site_id", "assemblage", "waterbody_type", "lat", "lon"])
    sites["site_id"] = sites["site_id"].astype(str)
    roles = UK.uk_site_roles(sites[(sites["assemblage"] == "invertebrate") & (sites["waterbody_type"] != "WBES")])
    assert set(si["site_id"].astype(str).map(roles).unique()) <= {"pool_fit", "pool_val"}
    frozen = {i for (s, i) in XD.frozen_keys_cached() if s == "uk"}
    assert frozen and not frozen & A["invertebrate"]
    sd = pd.read_parquet(P / "samples_diatom.parquet", columns=["site_id", "analysis_id", "sample_date", "analysis_type"])
    sd = sd[sd["analysis_id"].astype(str).isin(A["diatom"])]
    assert len(sd) == len(A["diatom"]) and set(sd["analysis_type"]) == {"LABP"}
    assert pd.to_datetime(sd["sample_date"]).dt.date.max() <= F.UK_ALLOWED_TO
    assert not set(sd["site_id"].astype(str)) & set(roles.index[roles == "reserved_test"])


@pytest.mark.skipif(not _have_uk(), reason="processed UK EA ecology tables not on this machine")
def test_uk_allowed_design_and_smoke_fit_shapes_only(uk_allowed):
    """A small deterministic subset of allowed UK rows goes through the loader, the fit guard and one training
    epoch. Shapes and finiteness only: no metric is computed."""
    from sentinel.core import paths
    from sentinel.models.bioassess import bioset_ma as MA
    from sentinel.models.bioassess import fdata as FD
    from sentinel.models.bioassess import ftrack as F

    P = paths.PROCESSED / "uk_ea_ecology"
    sd = pd.read_parquet(P / "samples_diatom.parquet", columns=["site_id", "analysis_id"]).astype(str)
    sd = sd[sd["analysis_id"].isin(uk_allowed["diatom"])].sort_values("analysis_id")
    si = pd.read_parquet(P / "samples_invertebrate.parquet", columns=["site_id", "analysis_id"]).astype(str)
    si = si[si["analysis_id"].isin(uk_allowed["invertebrate"])].sort_values("analysis_id")
    shared = sorted(set(sd["site_id"]) & set(si["site_id"]))[:60]
    sub = {"invertebrate": set(si[si["site_id"].isin(shared)]["analysis_id"].head(500)),
           "diatom": set(sd[sd["site_id"].isin(shared)]["analysis_id"].head(300)),
           "counts": {"subset": True}}
    D = FD.load_uk_allowed(allowed=sub)
    S = D.samples
    assert len(S) and S.index.is_unique and set(S["source"]) <= {"uk", "ukd"} and S["uk_allowed"].all()
    assert set(F.SPLIT_COLUMNS) <= set(S.columns) and list(D.Y.columns) == list(FD.Y_COLUMNS)
    assert pd.to_datetime(S["date"]).dt.date.max() <= F.UK_ALLOWED_TO
    assert {k[3:] for k in S.index[S["has_inv"]]} <= sub["invertebrate"]
    assert {k[4:] for k in S["diatom_key"].dropna()} <= sub["diatom"]
    att = S[S["has_inv"] & S["diatom_key"].notna()]
    assert (att["diatom_offset_days"].abs() <= FD.ATTACH_DAYS).all()
    assert set(D.tokens["invertebrate"]["key"]) == set(S["inv_key"].dropna())
    assert set(S["diatom_key"].dropna()) <= set(D.tokens["diatom"]["key"])
    assert D.tokens["diatom"].groupby("key").size().min() >= FD.MIN_TAXA
    ids = S.index
    F.assert_fit_ids(ids, S, "F-Z", None, "dev")                        # allowed UK rows are the F-Z training pool
    cfg = MA.recipe("ma_zs", d_model=32, trunk=48, max_epochs=1, patience=1, batch_size=128)
    fit_ids, val_ids = MA.inner_split(S, ids, seed=0)
    feat = MA.MAFeaturizer(cfg).fit(D, ids)
    tr, va = feat.transform(D, fit_ids), feat.transform(D, val_ids)
    assert tr.mod["invertebrate"].lev.shape[1] <= 150 and tr.mod["diatom"].lev.shape[1] <= 100
    cols = list(FD.Y_COLUMNS)
    fitted = MA.fit_ma(cfg, feat, tr, D.Y.loc[fit_ids, cols].to_numpy(float), va,
                       D.Y.loc[val_ids, cols].to_numpy(float), columns=cols, primary=FD.PRIMARY["F-Z"], seed=0)
    p = fitted.predict(va, region=S.loc[val_ids, "region"].to_numpy(object))
    assert p["loc"].shape == (len(val_ids), len(cols)) and np.isfinite(p["loc"]).all() and (p["scale"] > 0).all()


# ------------------------------------------------------------------------------------------------ locked store
# Fetcher, extractor and counting reader of ftrack, on synthetic responses from a fake transport. No network.
class _FakeHubeau:
    """Stand-in for the API: count responses (JSON) and ';'-separated CSV pages of ``n(endpoint, query)`` records."""

    def __init__(self, n):
        self.n, self.calls = n, []

    def __call__(self, url, headers):
        import json
        from urllib.parse import parse_qs, urlsplit

        assert headers["User-Agent"]
        self.calls.append(url)
        u = urlsplit(url)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        n = int(self.n(u.path.split("/")[-1].replace(".csv", ""), q))
        if not u.path.endswith(".csv"):
            assert q["size"] == "1" and "," not in q["fields"]
            return 200, json.dumps({"count": n, "data": [{q["fields"]: "X"}]}).encode()
        size, page = int(q["size"]), int(q["page"])
        k = max(0, min(size, n - (page - 1) * size))
        head = q["fields"].split(",")
        body = "\n".join([";".join(head)] + [";".join("1" for _ in head)] * k) + "\n"
        return (206 if page * size < n else 200), body.encode()


class _Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(round(s, 6))
        self.t += s


_STAMPED = ("docs/preregistration/bioassess_f_protocol.md", "docs/preregistration/bioassess_f_lead_rulings.md")
_G2 = ("docs/preregistration/bioassess_f_lock.json", "docs/preregistration/bioassess_f_codes.json")
_G4 = ("docs/preregistration/bioassess_f_split.json", "docs/preregistration/bioassess_f_codes.json")


def _gate_row(repo, experiment, files, workstream="bioassess", written_by="lead"):
    """Write ``files`` read-only in a toy repository and append the registry row that records their sha256."""
    import hashlib
    import json

    sha = {}
    for f in files:
        q = repo / f
        q.parent.mkdir(parents=True, exist_ok=True)
        if not q.exists():
            q.write_text(f"text of {f}")
        os.chmod(q, 0o444)
        sha[f] = hashlib.sha256(q.read_bytes()).hexdigest()
    with open(repo / "registry.jsonl", "a") as fh:
        extra = {"sha256": sha, **({"written_by": written_by} if written_by else {})}
        fh.write(json.dumps({"workstream": workstream, "experiment": experiment, "extra": extra}) + "\n")


def _fetch_kw(tmp_path, stamped=True, name="store"):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "registry.jsonl").touch()
    if stamped and not (repo / _STAMPED[1]).exists():      # the rulings file: written with the full row only
        _gate_row(repo, "bioassess_f/prereg_snapshot", _STAMPED)
    return dict(repo=repo, registry_path=repo / "registry.jsonl", root=tmp_path / name,
                manifest=tmp_path / f"{name}_manifest.jsonl", other_dirs=[], disk=lambda: (True, 100.0),
                today=dt.date(2026, 11, 10))


def _open_g2(tmp_path):
    _gate_row(tmp_path / "repo", "bioassess_f/gate_g2", _G2)


def test_stamp_guard_passes_only_on_the_lead_row_with_both_current_hashes_and_read_only_files(tmp_path):
    """Lead clarification L1: pass, changed file, missing row, writable file (plus near-miss rows)."""
    import json

    from sentinel.models.bioassess import ftrack as F

    assert F.STAMPED_FILES == _STAMPED and (F.STAMP_WORKSTREAM, F.STAMP_EXPERIMENT) == (
        "bioassess", "bioassess_f/prereg_snapshot")
    repo, reg = tmp_path / "repo", tmp_path / "repo" / "registry.jsonl"
    repo.mkdir()
    reg.touch()
    for f in _STAMPED:                                                   # files exist read-only, but no row yet
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text(f"text of {f}")
        os.chmod(repo / f, 0o444)
    with pytest.raises(F.StampError, match="no registry row"):           # case: missing row
        F.assert_stamped(reg, repo)
    _gate_row(repo, "bioassess_f/prereg_snapshot", _STAMPED, workstream="downstream")     # wrong workstream
    _gate_row(repo, "bioassess_f/smoke_synthetic", _STAMPED)                              # wrong experiment
    _gate_row(repo, "bioassess_f/prereg_snapshot", _STAMPED[:1])                          # protocol only
    _gate_row(repo, "bioassess_f/prereg_snapshot", _STAMPED, written_by="engineer")       # not the lead's row
    _gate_row(repo, "bioassess_f/prereg_snapshot", _STAMPED, written_by=None)
    with pytest.raises(F.StampError):
        F.assert_stamped(reg, repo)
    assert not F.stamp_ok(reg, repo)
    _gate_row(repo, "bioassess_f/prereg_snapshot", _STAMPED)
    row = F.assert_stamped(reg, repo)                                    # case: pass
    assert set(row["extra"]["sha256"]) == set(_STAMPED) and F.stamp_ok(reg, repo)

    for f in _STAMPED:                                                   # case: writable file (content unchanged)
        os.chmod(repo / f, 0o644)
        with pytest.raises(F.StampError, match="writable"):
            F.assert_stamped(reg, repo)
        os.chmod(repo / f, 0o444)
        assert F.stamp_ok(reg, repo)
    for f in _STAMPED:                                                   # case: changed file (read-only again)
        old = (repo / f).read_text()
        os.chmod(repo / f, 0o644)
        (repo / f).write_text(old + " edited")
        os.chmod(repo / f, 0o444)
        with pytest.raises(F.StampError, match="current sha256"):
            F.assert_stamped(reg, repo)
        os.chmod(repo / f, 0o644)
        (repo / f).write_text(old)
        os.chmod(repo / f, 0o444)
        assert F.stamp_ok(reg, repo)
    os.chmod(repo / _STAMPED[1], 0o644)                                  # a missing file is refused as well
    (repo / _STAMPED[1]).unlink()
    with pytest.raises(F.StampError, match="does not exist"):
        F.assert_stamped(reg, repo)
    assert json.loads(reg.read_text().splitlines()[-1])["experiment"] == "bioassess_f/prereg_snapshot"


def test_the_real_stamp_row_passes_the_guard_and_names_both_files():
    """The lead's row of 2026-11-10 in the real registry; read-only check of the two real files."""
    from sentinel.models.bioassess import ftrack as F

    row = F.assert_stamped()
    assert row["workstream"] == "bioassess" and set(F.STAMPED_FILES) <= set(row["extra"]["sha256"])
    with pytest.raises(F.StampError):                                    # the lead has not logged G2 or G4
        F.assert_g2()
    with pytest.raises(F.StampError):
        F.assert_g4()


_BIO = {"code_departement": "01", "code_support": "13", "date_debut_prelevement": "2019-01-01",
        "date_fin_prelevement": "2019-12-31"}
_PC14 = "1301,1302,1303,1305,1311,1312,1313,1319,1335,1337,1339,1340,1350,1433"
_PC = {"code_station": "S1,S2,S3,S4", "code_parametre": _PC14, "date_debut_prelevement": "2020-01-01",
       "date_fin_prelevement": "2020-12-31"}


def test_fetcher_sends_nothing_before_the_stamp_and_no_chemistry_request_before_the_lock(tmp_path):
    from sentinel.models.bioassess import ftrack as F

    api = _FakeHubeau(lambda ep, q: 10)
    kw = _fetch_kw(tmp_path, stamped=False)
    repo = kw["repo"]
    with pytest.raises(F.FetchRefused, match="G0"):                      # no stamped files, no row
        F.LockedFetcher(transport=api, **kw).pull("taxons", _BIO)
    _gate_row(repo, "bioassess_f/prereg_snapshot", _STAMPED[:1])
    with pytest.raises(F.FetchRefused, match="G0"):                      # a row that names the protocol only
        F.LockedFetcher(transport=api, **kw).pull("taxons", _BIO)
    assert api.calls == [] and not (tmp_path / "store").exists()         # nothing sent, nothing written

    kw = _fetch_kw(tmp_path)
    fx = F.LockedFetcher(transport=api, **kw)
    assert fx.count("taxons", _BIO) == 10 and len(api.calls) == 1        # biology is allowed after the stamp
    os.chmod(repo / _STAMPED[1], 0o644)                                  # the guard runs again before EVERY request
    with pytest.raises(F.FetchRefused, match="G0"):
        fx.pull("taxons", {**_BIO, "code_departement": "02"})
    os.chmod(repo / _STAMPED[1], 0o444)
    for ep, q in (("analyse_pc", _PC), ("station_pc", {"code_station": "S1"})):
        with pytest.raises(F.FetchRefused, match="G2"):                  # no lock list, no codes file, no lead row
            fx.pull(ep, q)
    for f in _G2:
        (repo / f).write_text("{}")
        os.chmod(repo / f, 0o444)
    _gate_row(repo, "bioassess_f/gate_g2", _G2, written_by="engineer")
    with pytest.raises(F.FetchRefused, match="G2"):                      # both files read-only; no row of the lead
        fx.pull("analyse_pc", _PC)
    assert len(api.calls) == 1
    _open_g2(tmp_path)                                                   # the lead's G2 row with both hashes
    assert fx.pull("analyse_pc", _PC)["records"] == 10
    os.chmod(repo / _G2[0], 0o644)
    with pytest.raises(F.FetchRefused, match="G2"):                      # lock list made writable afterwards
        fx.pull("analyse_pc", {**_PC, "code_station": "S9"})
    os.chmod(repo / _G2[0], 0o444)

    n = len(api.calls)
    bad = [("operation_pc", _PC), ("condition_environnementale_pc", _PC),
           ("analyse_pc", {**_PC, "code_parametre": "1304"}),            # conductivity at 20 degC is never requested
           ("analyse_pc", {**_PC, "code_parametre": "1340,1335"}),       # one form only: all 14 codes per request
           ("analyse_pc", {**_PC, "date_debut_prelevement": "2021-06-01", "date_fin_prelevement": "2022-06-01"}),
           ("analyse_pc", {**_PC, "code_station": ",".join(f"S{i}" for i in range(51))}),
           ("analyse_pc", {k: v for k, v in _PC.items() if k != "date_fin_prelevement"}),
           ("taxons", {**_BIO, "code_support": "4"}),
           ("station_pc", {"code_station": "S1", "fields": "code_station,durete"})]
    for ep, p in bad:
        with pytest.raises(F.FetchRefused):
            fx.pull(ep, p)
    with pytest.raises(F.FetchRefused, match="G1"):                      # disk floor
        F.LockedFetcher(transport=api, **{**kw, "disk": lambda: (False, 50.0)}).pull("taxons", {**_BIO, "x": 1})
    assert len(api.calls) == n
    assert "durete" not in ",".join(sum(map(list, F.WHITELIST.values()), []))


def test_fetcher_writes_only_the_store_and_the_manifest_throttles_splits_and_resumes(tmp_path):
    import gzip
    import hashlib
    import json

    from sentinel.models.bioassess import ftrack as F

    def n(ep, q):
        if ep != "analyse_pc":
            return 12000
        k = len(q["code_station"].split(","))
        a, b = q["date_debut_prelevement"], q["date_fin_prelevement"]
        if a[:7] != b[:7]:
            return 50000                                                 # a whole year: above the access depth
        return 7500 * k if a[5:7] == "03" else 250 * k                   # March needs the station split too

    api, clock = _FakeHubeau(n), _Clock()
    kw = _fetch_kw(tmp_path)
    _open_g2(tmp_path)
    before = {p for p in tmp_path.rglob("*")}
    fx = F.LockedFetcher(transport=api, sleep=clock.sleep, clock=clock, **kw)
    res = fx.resolve("analyse_pc", [_PC])
    assert len(res) == 13 and max(c for _, c in res) <= F.API_MAX_RECORDS
    assert sum(c for _, c in res) == 11 * 1000 + 30000
    march = sorted(p["code_station"] for p, _ in res if p["date_debut_prelevement"] == "2020-03-01")
    assert march == ["S1,S2", "S3,S4"]
    assert {(p["date_debut_prelevement"], p["date_fin_prelevement"]) for p, _ in res} >= {("2020-02-01", "2020-02-29")}
    with pytest.raises(F.FetchRefused, match="stop and report"):         # one station, one month, still too large
        F.split_block("analyse_pc", {**_PC, "code_station": "S1", "date_fin_prelevement": "2020-01-31"})

    out = fx.pull("taxons", _BIO)                                        # 12,000 records: pages of 5,000
    pages = [u for u in api.calls if "taxons.csv" in u]
    assert out["records"] == 12000 and len(pages) == 3 and all("size=5000" in u and "sort=asc" in u for u in pages)
    assert clock.sleeps and set(clock.sleeps) == {0.5} and len(clock.sleeps) == len(api.calls) - 1

    new = {p for p in tmp_path.rglob("*")} - before
    files = [p for p in new if p.is_file()]
    assert all(kw["root"] in [p, *p.parents] or p == kw["manifest"] for p in new)   # nothing written elsewhere
    stored = [p for p in files if p != kw["manifest"]]
    assert stored and all(p.stat().st_mode & 0o777 == 0o400 for p in stored)
    assert all(p.stat().st_mode & 0o777 == 0o700 for p in new if p.is_dir())
    rows = fx.manifest_rows()
    assert len(rows) == len(api.calls) == len(stored)
    for r in rows:
        assert {"url", "utc", "status", "bytes", "sha256", "licence", "attribution", "file"} <= set(r)
        assert r["utc"].endswith("Z") and r["status"] in (200, 206) and "Etalab" in r["licence"]
        body = gzip.open(kw["root"] / r["file"], "rb").read()
        assert len(body) == r["bytes"] and hashlib.sha256(body).hexdigest() == r["sha256"]

    n_calls = len(api.calls)                                             # resume: completed requests are not repeated
    fx2 = F.LockedFetcher(transport=api, sleep=clock.sleep, clock=clock, **kw)
    assert fx2.pull("taxons", _BIO)["records"] == 12000 and len(api.calls) == n_calls
    assert fx2.resolve("analyse_pc", [_PC]) == res and len(api.calls) == n_calls

    class Failing(_FakeHubeau):                                          # an HTTP error is logged and stops the pull
        def __call__(self, url, headers):
            self.calls.append(url)
            return 503, b""
    bad = Failing(n)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        F.LockedFetcher(transport=bad, **kw).pull("indices", _BIO)
    last = json.loads(kw["manifest"].read_text().splitlines()[-1])
    assert last["status"] == 503 and last["file"] is None and len(bad.calls) == 1

    class Flaky(_FakeHubeau):                                            # timeout, then 503, then the answer
        def __call__(self, url, headers):
            k = sum(u == url for u in self.calls)
            if k == 0:
                self.calls.append(url)
                raise TimeoutError("timed out")
            if k == 1:
                self.calls.append(url)
                return 503, b""
            return super().__call__(url, headers)
    flaky, c2 = Flaky(n), _Clock()
    q = {**_BIO, "code_departement": "05"}
    fy = F.LockedFetcher(transport=flaky, sleep=c2.sleep, clock=c2, retries=3, **kw)
    assert fy.count("indices", q) == 12000 and len(flaky.calls) == 3     # every attempt is one manifest row
    tail = [json.loads(ln) for ln in kw["manifest"].read_text().splitlines()[-3:]]
    assert [r["status"] for r in tail] == [0, 503, 200] and len({r["url"] for r in tail}) == 1
    assert tail[0]["error"] == "TimeoutError" and [r.get("attempt", 0) for r in tail] == [0, 1, 2]
    assert [w for w in c2.sleeps if w > 1] == [F.RETRY_WAIT_S, 2 * F.RETRY_WAIT_S]

    class Gone(_FakeHubeau):                                             # not a transient status: never sent again
        def __call__(self, url, headers):
            self.calls.append(url)
            return 404, b""
    gone = Gone(n)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        F.LockedFetcher(transport=gone, sleep=c2.sleep, clock=c2, retries=3, **kw).count("indices", {**q, "x": 1})
    assert len(gone.calls) == 1
    always = Failing(n)                                                  # retries are bounded
    with pytest.raises(RuntimeError, match="HTTP 503"):
        F.LockedFetcher(transport=always, sleep=c2.sleep, clock=c2, retries=2, **kw).count("indices", {**q, "x": 2})
    assert len(always.calls) == 3

    def big_month(ep, q):                                                # one department-month above the depth
        a, b = q["date_debut_prelevement"], q["date_fin_prelevement"]
        return 60000 if a[:7] != b[:7] else (25000 if a[5:7] == "07" else 100)
    q = {**_BIO, "code_departement": "06"}
    with pytest.raises(F.FetchRefused, match="stop and report"):
        F.LockedFetcher(transport=_FakeHubeau(big_month), **kw).resolve("taxons", [q])
    left: list = []
    api3 = _FakeHubeau(big_month)
    out = F.LockedFetcher(transport=api3, **kw).run_step("taxons", [q], unsplit=left)
    assert out["blocks"] == 11 and out["records"] == 1100                # the July block is recorded, not pulled
    assert [(p["date_debut_prelevement"], k) for p, k in left] == [("2019-07-01", 25000)]
    assert not any("taxons.csv" in u and "2019-07-01" in u for u in api3.calls)


def test_fetcher_stops_when_one_and_a_half_times_the_projection_exceeds_the_cap(tmp_path):
    from sentinel.models.bioassess import ftrack as F

    blocks = [{**_BIO, "code_departement": d} for d in ("01", "02", "03")]
    probe = F.LockedFetcher(transport=_FakeHubeau(lambda ep, q: 4000), **_fetch_kw(tmp_path, name="probe"))
    one = probe.run_step("taxons", blocks[:1])
    assert one["records"] == 4000 and one["gz_bytes_per_record"] > 0
    api = _FakeHubeau(lambda ep, q: 4000)
    kw = _fetch_kw(tmp_path, name="capped")
    fx = F.LockedFetcher(transport=api, cap_gb=2.2 * probe.used_gb(), **kw)   # room for two blocks, not for 1.5 x two
    with pytest.raises(F.FetchRefused, match="stop and report"):
        fx.run_step("taxons", blocks)
    assert sum("taxons.csv" in u for u in api.calls) == 1                # the first block measured; no second pull
    assert fx.used_gb() <= 2.2 * probe.used_gb()


def test_request_blocks_follow_the_route(tmp_path):
    from sentinel.models.bioassess import ftrack as F

    today = dt.date(2026, 11, 10)
    b = F.biology_blocks(today)
    assert len(b["stations_hydrobio"]) == len(F.METROPOLITAN)
    assert len(b["taxons"]) == len(b["indices"]) == len(F.METROPOLITAN) * 2 * 20       # 2007..2026
    assert {x["code_support"] for x in b["taxons"]} == {"13", "10"}
    assert min(x["date_debut_prelevement"] for x in b["taxons"]) == "2007-01-01"
    assert max(x["date_fin_prelevement"] for x in b["taxons"]) == "2026-11-10"
    with pytest.raises(F.FetchRefused):
        F.biology_blocks(today, departments=["971"])                     # overseas departments are excluded
    st = [f"0{i:07d}" for i in range(120)]
    c = F.chemistry_blocks(st + st[:5], today)
    assert len(c) == 3 * 21 and all(len(x["code_station"].split(",")) <= 50 for x in c)      # 2006..2026
    assert all(x["code_parametre"].split(",") == list(F.PC_PARAMETERS) for x in c)
    lock = F.LOCK_FROM.isoformat()
    assert all((x["date_debut_prelevement"] < lock) == (x["date_fin_prelevement"] < lock) for x in c)
    assert ",".join(F.PC_PARAMETERS) == _PC14 and "per_parameter" not in F.chemistry_blocks.__code__.co_varnames
    assert set(F.PC_PARAMETERS) == {"1302", "1303", "1337", "1340", "1335", "1433", "1350", "1305", "1311", "1312",
                                    "1313", "1339", "1319", "1301"}


def _store_csv(root, endpoint, name, header, rows):
    import gzip

    d = root / endpoint
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / f"page_{name}.csv.gz", "wt", encoding="utf-8") as fh:
        fh.write(";".join(header) + "\n" + "".join(";".join(map(str, r)) + "\n" for r in rows))


def _toy_store(root):
    from sentinel.models.bioassess import ftrack as F

    dates = ["2019-05-01", "2021-12-31", "2022-01-01", "2024-07-01T00:00:00Z", "", "not a date", "1985-01-01",
             "2027-01-01"]
    head = ["code_station_hydrobio", "date_prelevement", "code_support", "code_appel_taxon", "resultat_taxon",
            "profondeur", "commentaire"]
    _store_csv(root, "taxons", "a", head, [("ST1", d, "13", "T9", "4", "DEPTH", "FREE TEXT") for d in dates])
    _store_csv(root, "stations_hydrobio", "a", ["code_station_hydrobio", "longitude", "latitude", "libelle"],
               [("ST1", "2.5", "46.5", "NAME")])
    _store_csv(root, "station_pc", "a", ["code_station", "altitude_point_caracteristique", "durete"],
               [("ST1", "210", "HARDNESS")])
    chem = list(F.CHEM_WHITELIST) + ["durete", "libelle_qualification"]
    rows = []
    for i, d in enumerate(dates + ["2021-06-15", "2023-03-03"]):
        r = dict.fromkeys(chem, "1")
        r.update(code_station="ST1", date_prelevement=d, code_parametre="1340", durete="HARDNESS",
                 resultat=f"SECRET{d[:4]}_{i}", limite_quantification=f"SECRET{d[:4]}_LQ")
        rows.append([r[c] for c in chem])
    _store_csv(root, "analyse_pc", "a", chem, rows)


def _all_cells(folder):
    out = set()
    for f in sorted(Path(folder).rglob("*.parquet")):
        out |= set(pd.read_parquet(f).astype(str).to_numpy().ravel())
    return out


def test_extractor_copies_whitelisted_columns_and_routes_rows_by_date_only(tmp_path):
    from sentinel.models.bioassess import ftrack as F

    root, out, day = tmp_path / "store", tmp_path / "out", dt.date(2026, 11, 10)
    _toy_store(root)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "registry.jsonl").touch()
    kw = dict(root=root, out=out, registry_path=repo / "registry.jsonl", repo=repo)
    with pytest.raises(F.StampError):                                    # the extractor calls the stamp guard first
        F.extract_locked(day, **kw)
    _gate_row(repo, "bioassess_f/prereg_snapshot", _STAMPED)
    c = F.extract_locked(day, **kw)                                      # default: no chemistry at all
    assert c["tables"] == {"stations_hydrobio": 1, "station_pc": 1, "taxons_unlocked": 2, "taxons_locked_inputs": 2}
    assert c["stay_locked_bad_date"] == {"taxons": 4} and not list(out.glob("analyse_pc*"))
    un = pd.concat(pd.read_parquet(f) for f in (out / "taxons_unlocked").glob("*.parquet"))
    lk = pd.concat(pd.read_parquet(f) for f in (out / "taxons_locked_inputs").glob("*.parquet"))
    assert sorted(un["date_prelevement"]) == ["2019-05-01", "2021-12-31"]
    assert sorted(lk["date_prelevement"].str[:10]) == ["2022-01-01", "2024-07-01"]
    assert set(un.columns) <= set(F.BIO_WHITELIST) and "profondeur" not in un and "commentaire" not in lk
    assert not {"DEPTH", "FREE TEXT", "NAME", "HARDNESS"} & _all_cells(out)
    assert list(pd.read_parquet(next((out / "station_pc").glob("*.parquet"))).columns) == [
        "code_station", "altitude_point_caracteristique"]

    c = F.extract_locked(day, chemistry="codes", **kw)                   # G4 tabulation: codes, no value column
    assert c["tables"]["analyse_pc_codes_unlocked"] == 3 and c["chemistry_rows_left_locked"] == 3
    assert c["stay_locked_bad_date"]["analyse_pc"] == 4
    codes = pd.concat(pd.read_parquet(f) for f in (out / "analyse_pc_codes_unlocked").glob("*.parquet"))
    assert not set(F.CHEM_VALUE_COLS) & set(codes.columns) and codes["date_prelevement"].max() <= "2021-12-31"
    assert not any(str(v).startswith("SECRET") for v in _all_cells(out))

    with pytest.raises(F.FetchRefused, match="G4"):                      # values need the lead's G4 row
        F.extract_locked(day, chemistry="full", **kw)
    for f in _G4:
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text("{}")
        os.chmod(repo / f, 0o444)
    _gate_row(repo, "bioassess_f/gate_g4", _G4, written_by="engineer")
    with pytest.raises(F.FetchRefused, match="G4"):                      # files read-only, but no row of the lead
        F.extract_locked(day, chemistry="full", **kw)
    _gate_row(repo, "bioassess_f/gate_g4", _G4)
    c = F.extract_locked(day, chemistry="full", **kw)
    full = pd.concat(pd.read_parquet(f) for f in (out / "analyse_pc_unlocked").glob("*.parquet"))
    assert list(full.columns) == list(F.CHEM_WHITELIST) and c["chemistry_rows_left_locked"] == 3
    assert sorted(full["resultat"].str[:10]) == ["SECRET2019", "SECRET2021", "SECRET2021"]   # copied as text
    cells = _all_cells(out)
    assert not any(str(v).startswith(("SECRET2022", "SECRET2024", "SECRET2023", "SECRET1985", "SECRET2027",
                                      "SECRET_", "SECRETnot")) for v in cells)       # locked or undated: never out
    assert "HARDNESS" not in cells


def test_locked_availability_applies_rules_1_to_8_and_returns_counts_only(tmp_path, monkeypatch):
    from sentinel.models.bioassess import ftrack as F

    root, head = tmp_path / "store", list(F.CHEM_WHITELIST)
    kw = {k: v for k, v in _fetch_kw(tmp_path).items() if k in ("repo", "registry_path")}

    def row(st, d, par="1340", res="7.77", **over):
        r = dict.fromkeys(head, "1")
        r.update(code_station=st, date_prelevement=d, code_parametre=par, code_fraction="23", code_unite="173",
                 code_remarque="1", code_qualification="1", code_support="3", code_insitu="0",
                 heure_prelevement="10:00:00", resultat=res, limite_quantification="SECRET")
        r.update(over)
        return [r[c] for c in head]

    _store_csv(root, "analyse_pc", "a", head, [
        row("A", "2023-06-01"), row("A", "2023-06-01"),                  # rule 7: an exact duplicate counts once
        row("A", "2023-06-01", res="8.5"),                               # same day, another result: a second analysis
        row("A", "2023-06-20", code_insitu="1"),                         # rule 6: in situ and laboratory both count
        row("A", "2023-07-15T00:00:00Z", code_remarque="10"),            # rule 5: censored below kept; 35 days later
        row("A", "2022-09-01", code_qualification="2"),                  # rule 3: "incorrecte" is dropped
        row("A", "2023-06-02", code_qualification="3"),
        row("A", "2023-06-03", code_support="6"),                        # rule 1: not water
        row("A", "2023-06-04", code_unite="999"),                        # rule 2: unit outside the table
        row("A", "2023-06-04", code_fraction="3", code_unite="168"),     # rule 2: filtered phase, mg N/L: kept
        row("A", "2023-06-05", par="1304", code_unite="147"),            # rule 2: conductivity at 20 degC: no label
        row("A", "2023-06-06", code_remarque="0"), row("A", "2023-06-06", code_remarque="5"),     # rule 5
        row("A", "2023-06-07", res=""),                                  # rule 5: empty result
        row("A", "2023-06-08", res="n.d."), row("A", "2023-06-08", res="inf"),                    # rule 8
        row("A", "2023-06-05", par="1302", code_unite="264", res="7.9"),
        row("A", "bad date"), row("C", "2023-06-10")])                   # no date; a station that is not paired
    _store_csv(root, "analyse_pc", "b", head, [row("A", "2023-06-01"), row("B", "2023-06-10", code_remarque="3")])
    ops = pd.DataFrame({"key": ["FR:A:2023-06-10:13", "FR:B:2023-06-10:13", "FR:Z:2023-06-10:13"],
                        "station": ["A", "B", "Z"], "date": ["2023-06-10"] * 3})
    seen = []
    real = F._read_stored
    monkeypatch.setattr(F, "_read_stored", lambda path, columns: (seen.append(set(columns)), real(path, columns))[1])
    a = F.locked_availability(ops, root=root, **kw)
    assert seen and all(c == set(F.CHEM_COUNT_COLS) for c in seen)       # the limit columns are never loaded
    assert not {"limite_detection", "limite_quantification", "limite_saturation"} & set(F.CHEM_COUNT_COLS)
    assert list(a.columns) == ["key", "parameter", "window", "n_obs", "n_dates"] and set(a.attrs) == {"dropped"}
    assert all(isinstance(v, int) for v in a.attrs["dropped"].values())  # per-rule drop counts, nothing else
    got = {(k, q, w): (n, d) for k, q, w, n, d in a.itertuples(index=False)}
    assert got == {("FR:A:2023-06-10:13", "1340", "pm30d"): (4, 3), ("FR:A:2023-06-10:13", "1340", "prior365d"): (3, 2),
                   ("FR:A:2023-06-10:13", "1302", "pm30d"): (1, 1), ("FR:A:2023-06-10:13", "1302", "prior365d"): (1, 1),
                   ("FR:B:2023-06-10:13", "1340", "pm30d"): (1, 1), ("FR:B:2023-06-10:13", "1340", "prior365d"): (1, 1)}
    assert a.attrs["dropped"] == {"rule1_support": 1, "rule2_parameter_fraction_unit": 2, "rule3_qualification": 2,
                                  "rule5_remark_or_empty": 3, "rule8_not_numeric": 2, "no_date": 1,
                                  "rule7_exact_duplicates": 2}
    flat = a.astype(str).to_numpy().ravel().tolist() + [str(a.attrs)]
    assert not any(v in " ".join(flat) for v in ("7.77", "8.5", "7.9", "SECRET"))      # counts only, no result value
    b = F.locked_availability(ops, root=root, codes={"qualification": ("1",)}, **kw)    # the code-1-only sensitivity
    assert len(b) == len(a)
    assert len(F.locked_availability(ops, root=tmp_path / "empty", **kw)) == 0
    with pytest.raises(F.StampError):                                    # the reader calls the stamp guard first
        F.locked_availability(ops, root=root, registry_path=tmp_path / "none.jsonl", repo=kw["repo"])
