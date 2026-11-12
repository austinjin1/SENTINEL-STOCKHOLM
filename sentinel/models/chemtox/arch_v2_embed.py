"""The concatenated foundation-model embedding block for the M4 arch-v2 tracks.

New module, no project row read: every function takes SMILES or frames the caller already obtained
through ``arch_v2_guard``. The encoder is **pluggable**, so the tests exercise the cache, the
concatenation and the leakage rules on a synthetic encoder and never touch the network.

Governing text in precedence order: ``chemtox_arch_v2_lead_rulings.md`` (sha256 ``553be413…``),
``chemtox_arch_v2_addendum_01.md``, the stamped amendment §10.5 and §5.6.

The rules this module enforces
------------------------------
* **Concatenate, never substitute** (§4.2 of the architecture plan, amendment §10.1).
  :func:`concatenate` raises if a single published column would be dropped. ``embeddings v1`` scored
  1.1579 and was discarded because it *replaced* the exposure covariates; that result says nothing
  about embeddings, and this function makes repeating the mistake impossible.
* **One source only, by the §5.6 precedence** — MolFormer-XL (Apache-2.0, licence checked
  2026-11-10); ChemBERTa-2 is **not cleared** and :data:`UNCLEARED_SOURCES` refuses it by name.
  A second source is a new entry against the declared configuration total (§5.5 iii).
* **PRETRAINED-CACHE**: the cache key records the **split file's sha256**, so a cache built against
  another split cannot be loaded, and :func:`cache_key` refuses to build a key without one.
* The frozen third-party weights saw the held-out *structures* — **structures, no labels and none of
  our rows**. :data:`DISCLOSURE` carries that sentence into every write-up.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from sentinel.core import paths

__all__ = ["EmbedError", "MOLFORMER", "UNCLEARED_SOURCES", "DISCLOSURE", "CACHE_ROOT",
           "EMB_PREFIX", "cache_key", "EmbeddingCache", "molformer_encoder", "concatenate",
           "embed_frame"]


class EmbedError(ValueError):
    """A pre-registered rule about the embedding block was violated."""


#: The chosen source. Recorded in ``configs/chemtox/arch_v2.yaml`` before any fit.
MOLFORMER = {"repo_id": "ibm-research/MoLFormer-XL-both-10pct", "licence": "apache-2.0",
             "licence_checked": "2026-11-10", "gated": False, "pooling": "mean",
             "citation": "Ross et al., Nat. Mach. Intell. 4(12):1256-1264 (2022), "
                         "doi 10.1038/s42256-022-00580-7"}

#: Refused by name: no licence is stated for these weights (§5.6, docs/licenses.md).
UNCLEARED_SOURCES = {
    "DeepChem/ChemBERTa-77M-MLM": "no licence tag and no model card; the DeepChem MIT licence "
                                  "covers the code, not these weights",
    "DeepChem/ChemBERTa-10M-MLM": "no licence stated",
    "seyonec/ChemBERTa-zinc-base-v1": "no licence stated",
}

DISCLOSURE = (
    "The frozen pretrained embeddings were trained by third parties on public chemical corpora "
    "that include the held-out structures. They saw STRUCTURES, no labels and none of our rows. "
    "The entry `lightgbm_noemb` exists so the embedding's contribution is measured, not assumed.")

CACHE_ROOT = paths.DATA_ROOT / "processed" / "chemtox_arch_v2" / "embeddings"

#: Every embedding column carries this prefix, so :func:`concatenate` can tell them apart from the
#: published block and a feature-name parity record stays readable.
EMB_PREFIX = "emb_"


def cache_key(*, split_sha256: str, repo_id: str = MOLFORMER["repo_id"],
              pooling: str = "mean", revision: str | None = None,
              n_structures: int | None = None) -> str:
    """The cache identity. **Includes the split sha256** (PRETRAINED-CACHE).

    A cache built against a different split, a different source or a different pooling therefore
    has a different key and cannot silently be reused.
    """
    if not isinstance(split_sha256, str) or len(split_sha256) != 64:
        raise EmbedError(
            "cache_key needs the 64-character sha256 of the pinned split file: a cache that does "
            "not record which split it was built against must not be loadable (PRETRAINED-CACHE)")
    if repo_id in UNCLEARED_SOURCES:
        raise EmbedError(f"{repo_id} is not licence-cleared: {UNCLEARED_SOURCES[repo_id]}")
    payload = {"split_sha256": split_sha256, "repo_id": repo_id, "pooling": pooling,
               "revision": revision, "n_structures": n_structures}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass
class EmbeddingCache:
    """A keyed, on-disk cache of one embedding per canonical SMILES.

    ``root`` defaults to the project cache; the tests pass a temporary directory. Writing is
    atomic-ish (write then rename) and the manifest records the key's inputs so a stale cache is
    identifiable from disk alone.
    """

    key: str
    root: Path = CACHE_ROOT
    dim: int | None = None

    @property
    def path(self) -> Path:
        return Path(self.root) / f"emb_{self.key[:16]}.parquet"

    @property
    def manifest_path(self) -> Path:
        return Path(self.root) / f"emb_{self.key[:16]}.manifest.json"

    def exists(self) -> bool:
        return self.path.is_file()

    def write(self, smiles: Sequence[str], vectors: np.ndarray, *, meta: Mapping) -> Path:
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim != 2 or v.shape[0] != len(smiles):
            raise EmbedError(f"vectors must be (n_smiles, dim); got {v.shape} for "
                             f"{len(smiles)} smiles")
        if len(set(smiles)) != len(smiles):
            raise EmbedError("the cache holds one row per unique SMILES; duplicates were passed")
        Path(self.root).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(v, columns=[f"{EMB_PREFIX}{i}" for i in range(v.shape[1])])
        df.insert(0, "chem_smiles", list(smiles))
        tmp = self.path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(self.path)
        self.manifest_path.write_text(json.dumps(
            {"key": self.key, "n_structures": len(smiles), "dim": int(v.shape[1]),
             "disclosure": DISCLOSURE, **dict(meta)}, indent=1, sort_keys=True) + "\n")
        self.dim = int(v.shape[1])
        return self.path

    def read(self) -> pd.DataFrame:
        if not self.exists():
            raise EmbedError(f"no embedding cache at {self.path}")
        df = pd.read_parquet(self.path)
        if "chem_smiles" not in df.columns:
            raise EmbedError(f"{self.path} is not an embedding cache")
        self.dim = sum(c.startswith(EMB_PREFIX) for c in df.columns)
        return df


def molformer_encoder(repo_id: str = MOLFORMER["repo_id"], *, pooling: str = "mean",
                      device: str = "cpu", batch_size: int = 64) -> Callable[[Sequence[str]],
                                                                            np.ndarray]:
    """Build the real MolFormer-XL encoder. **Downloads weights on first use.**

    Not called by any test: the tests inject a synthetic encoder. Kept here so the one place that
    touches the network is explicit, licence-checked and named.
    """
    if repo_id in UNCLEARED_SOURCES:
        raise EmbedError(f"{repo_id} is not licence-cleared: {UNCLEARED_SOURCES[repo_id]}")
    if pooling not in ("mean", "cls"):
        raise EmbedError(f"unsupported pooling {pooling!r}")

    def encode(smiles: Sequence[str]) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(repo_id, trust_remote_code=True).to(device).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(smiles), batch_size):
                chunk = [str(s) for s in smiles[i:i + batch_size]]
                enc = tok(chunk, padding=True, truncation=True, return_tensors="pt").to(device)
                hidden = mdl(**enc).last_hidden_state
                if pooling == "cls":
                    vec = hidden[:, 0]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(vec.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    return encode


def concatenate(block: pd.DataFrame, embeddings: pd.DataFrame, *,
                on: str = "chem_smiles") -> pd.DataFrame:
    """Append the embedding columns to the published block. **Never substitutes.**

    Raises if any published column would be lost, if the embedding would overwrite one, or if a row
    would be dropped or duplicated by the join.
    """
    if on not in block.columns:
        raise EmbedError(f"the block has no {on!r} column to join on")
    if on not in embeddings.columns:
        raise EmbedError(f"the embedding table has no {on!r} column")
    emb_cols = [c for c in embeddings.columns if c.startswith(EMB_PREFIX)]
    if not emb_cols:
        raise EmbedError(f"the embedding table has no {EMB_PREFIX}* columns")
    clash = sorted(set(block.columns) & set(emb_cols))
    if clash:
        raise EmbedError(f"the embedding would overwrite published columns {clash}: the block is "
                         f"concatenated, never substituted")
    if embeddings[on].duplicated().any():
        raise EmbedError("the embedding table must hold one row per unique SMILES")
    before = len(block)
    out = block.merge(embeddings[[on, *emb_cols]], on=on, how="left")
    if len(out) != before:
        raise EmbedError(f"the join changed the row count {before} -> {len(out)}")
    missing = sorted(set(block.columns) - set(out.columns))
    if missing:
        raise EmbedError(f"published columns were lost: {missing}")
    unmatched = int(out[emb_cols[0]].isna().sum())
    out.attrs["embedding_columns"] = emb_cols
    out.attrs["embedding_unmatched_rows"] = unmatched
    out.attrs["embedding_disclosure"] = DISCLOSURE
    return out


def embed_frame(frame: pd.DataFrame, *, split_sha256: str,
                encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
                repo_id: str = MOLFORMER["repo_id"], pooling: str = "mean",
                root: Path | None = None, smiles_column: str = "chem_smiles",
                allow_build: bool = True) -> tuple[pd.DataFrame, dict]:
    """Attach the embedding to ``frame``, building or reusing the keyed cache.

    ``encoder`` is injected; when it is ``None`` the real MolFormer encoder is built, which
    downloads weights. Returns ``(frame_with_embedding, report)``; the report carries counts and the
    cache identity only.
    """
    if smiles_column not in frame.columns:
        raise EmbedError(f"the frame has no {smiles_column!r} column")
    uniq = sorted({str(s) for s in frame[smiles_column].dropna().unique()})
    if not uniq:
        raise EmbedError("no structure to embed")
    key = cache_key(split_sha256=split_sha256, repo_id=repo_id, pooling=pooling,
                    n_structures=len(uniq))
    cache = EmbeddingCache(key=key, root=Path(root) if root is not None else CACHE_ROOT)
    if cache.exists():
        table, built = cache.read(), False
    else:
        if not allow_build:
            raise EmbedError(f"no cache for key {key[:16]} and allow_build is False")
        enc = encoder if encoder is not None else molformer_encoder(repo_id, pooling=pooling)
        vectors = np.asarray(enc(uniq), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(uniq):
            raise EmbedError(f"the encoder returned {vectors.shape} for {len(uniq)} structures")
        cache.write(uniq, vectors, meta={"repo_id": repo_id, "pooling": pooling,
                                         "split_sha256": split_sha256,
                                         "licence": MOLFORMER["licence"],
                                         "licence_checked": MOLFORMER["licence_checked"]})
        table, built = cache.read(), True
    out = concatenate(frame, table, on=smiles_column)
    report = {"cache_key": key, "cache_path": str(cache.path), "cache_built_now": built,
              "n_structures": len(uniq), "embedding_dim": cache.dim,
              "n_embedding_columns": len(out.attrs["embedding_columns"]),
              "unmatched_rows": out.attrs["embedding_unmatched_rows"],
              "repo_id": repo_id, "licence": MOLFORMER["licence"],
              "split_sha256": split_sha256, "disclosure": DISCLOSURE}
    return out, report
