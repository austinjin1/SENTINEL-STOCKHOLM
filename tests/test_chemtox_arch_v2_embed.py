"""Tests for the arch-v2 embedding block. Synthetic encoder; no network, no project row."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.models.chemtox import arch_v2_embed as E

SPLIT_SHA = "a" * 64


def fake_encoder(dim=8, seed=0):
    def encode(smiles):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(len(smiles), dim)).astype(np.float32)
    return encode


def _block(n=6):
    return pd.DataFrame({"chem_smiles": [f"C{i%3}" for i in range(n)],
                         "logp": np.linspace(0, 3, n),
                         "duration_h": 48.0,
                         "maccs_1": 1})


# ------------------------------------------------------------------------------ the cache key

def test_cache_key_requires_the_split_sha256():
    with pytest.raises(E.EmbedError, match="PRETRAINED-CACHE"):
        E.cache_key(split_sha256="short")
    assert len(E.cache_key(split_sha256=SPLIT_SHA)) == 64


def test_cache_key_changes_with_the_split():
    a = E.cache_key(split_sha256="a" * 64)
    b = E.cache_key(split_sha256="b" * 64)
    assert a != b, "a cache built against another split must not be loadable"


def test_cache_key_changes_with_source_and_pooling():
    base = E.cache_key(split_sha256=SPLIT_SHA)
    assert E.cache_key(split_sha256=SPLIT_SHA, pooling="cls") != base
    assert E.cache_key(split_sha256=SPLIT_SHA, repo_id="other/model") != base


def test_cache_key_refuses_an_uncleared_source():
    for repo in E.UNCLEARED_SOURCES:
        with pytest.raises(E.EmbedError, match="not licence-cleared"):
            E.cache_key(split_sha256=SPLIT_SHA, repo_id=repo)


def test_chemberta_is_in_the_uncleared_list():
    assert "DeepChem/ChemBERTa-77M-MLM" in E.UNCLEARED_SOURCES
    assert E.MOLFORMER["licence"] == "apache-2.0" and E.MOLFORMER["gated"] is False


# --------------------------------------------------------------------------------- the cache

def test_cache_round_trip_and_manifest(tmp_path):
    key = E.cache_key(split_sha256=SPLIT_SHA, n_structures=3)
    cache = E.EmbeddingCache(key=key, root=tmp_path)
    assert not cache.exists()
    v = np.arange(12, dtype=np.float32).reshape(3, 4)
    cache.write(["C0", "C1", "C2"], v, meta={"repo_id": "x", "split_sha256": SPLIT_SHA})
    assert cache.exists()
    back = cache.read()
    assert list(back["chem_smiles"]) == ["C0", "C1", "C2"]
    assert cache.dim == 4
    man = json.loads(cache.manifest_path.read_text())
    assert man["n_structures"] == 3 and man["dim"] == 4
    assert "structures, no labels" in man["disclosure"].lower() or "STRUCTURES" in man["disclosure"]


def test_cache_rejects_duplicates_and_shape_mismatch(tmp_path):
    cache = E.EmbeddingCache(key="k" * 64, root=tmp_path)
    with pytest.raises(E.EmbedError, match="duplicates"):
        cache.write(["C0", "C0"], np.zeros((2, 3), dtype=np.float32), meta={})
    with pytest.raises(E.EmbedError, match="n_smiles"):
        cache.write(["C0", "C1"], np.zeros((3, 3), dtype=np.float32), meta={})


def test_cache_read_without_a_file(tmp_path):
    with pytest.raises(E.EmbedError, match="no embedding cache"):
        E.EmbeddingCache(key="z" * 64, root=tmp_path).read()


# ------------------------------------------------------ concatenate, never substitute

def test_concatenate_keeps_every_published_column():
    block = _block()
    emb = pd.DataFrame({"chem_smiles": ["C0", "C1", "C2"],
                        "emb_0": [0.1, 0.2, 0.3], "emb_1": [1.0, 2.0, 3.0]})
    out = E.concatenate(block, emb)
    for c in block.columns:
        assert c in out.columns, c
    assert len(out) == len(block)
    assert out.attrs["embedding_columns"] == ["emb_0", "emb_1"]


def test_concatenate_refuses_to_overwrite_a_published_column():
    block = _block().assign(emb_0=1.0)
    emb = pd.DataFrame({"chem_smiles": ["C0", "C1", "C2"], "emb_0": [0.1, 0.2, 0.3]})
    with pytest.raises(E.EmbedError, match="concatenated, never substituted"):
        E.concatenate(block, emb)


def test_concatenate_refuses_a_duplicated_embedding_table():
    emb = pd.DataFrame({"chem_smiles": ["C0", "C0"], "emb_0": [0.1, 0.2]})
    with pytest.raises(E.EmbedError, match="one row per unique SMILES"):
        E.concatenate(_block(), emb)


def test_concatenate_needs_embedding_columns():
    emb = pd.DataFrame({"chem_smiles": ["C0"], "not_an_embedding": [1.0]})
    with pytest.raises(E.EmbedError, match="emb_"):
        E.concatenate(_block(), emb)


def test_concatenate_reports_unmatched_rows():
    block = pd.DataFrame({"chem_smiles": ["C0", "CZ"], "logp": [1.0, 2.0]})
    emb = pd.DataFrame({"chem_smiles": ["C0"], "emb_0": [0.5]})
    out = E.concatenate(block, emb)
    assert out.attrs["embedding_unmatched_rows"] == 1


# ------------------------------------------------------------------------------ embed_frame

def test_embed_frame_builds_then_reuses_the_cache(tmp_path):
    block = _block()
    out1, rep1 = E.embed_frame(block, split_sha256=SPLIT_SHA, encoder=fake_encoder(),
                               root=tmp_path)
    assert rep1["cache_built_now"] is True
    assert rep1["n_structures"] == 3 and rep1["embedding_dim"] == 8
    assert rep1["n_embedding_columns"] == 8
    assert rep1["split_sha256"] == SPLIT_SHA
    out2, rep2 = E.embed_frame(block, split_sha256=SPLIT_SHA, encoder=None, root=tmp_path)
    assert rep2["cache_built_now"] is False, "the second call must reuse the cache, not download"
    assert out1.shape == out2.shape


def test_embed_frame_refuses_to_build_when_not_allowed(tmp_path):
    with pytest.raises(E.EmbedError, match="allow_build is False"):
        E.embed_frame(_block(), split_sha256=SPLIT_SHA, root=tmp_path, allow_build=False)


def test_embed_frame_rejects_a_bad_encoder(tmp_path):
    def bad(smiles):
        return np.zeros((len(smiles) + 1, 4), dtype=np.float32)
    with pytest.raises(E.EmbedError, match="returned"):
        E.embed_frame(_block(), split_sha256=SPLIT_SHA, encoder=bad, root=tmp_path)


def test_embed_frame_needs_a_smiles_column(tmp_path):
    with pytest.raises(E.EmbedError, match="chem_smiles"):
        E.embed_frame(pd.DataFrame({"x": [1]}), split_sha256=SPLIT_SHA,
                      encoder=fake_encoder(), root=tmp_path)


def test_a_cache_from_another_split_is_not_reused(tmp_path):
    block = _block()
    E.embed_frame(block, split_sha256="a" * 64, encoder=fake_encoder(), root=tmp_path)
    _out, rep = E.embed_frame(block, split_sha256="b" * 64, encoder=fake_encoder(), root=tmp_path)
    assert rep["cache_built_now"] is True, "a different split must force a rebuild"


def test_disclosure_is_carried_on_the_frame_and_the_report(tmp_path):
    out, rep = E.embed_frame(_block(), split_sha256=SPLIT_SHA, encoder=fake_encoder(),
                             root=tmp_path)
    assert "no labels" in rep["disclosure"]
    assert out.attrs["embedding_disclosure"] == E.DISCLOSURE


def test_module_reads_no_project_frame():
    src = Path(E.__file__).read_text()
    assert "ecotox_aquatic" not in src and "load_test_block" not in src
