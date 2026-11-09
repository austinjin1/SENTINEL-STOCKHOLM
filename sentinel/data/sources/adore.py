# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""ADORE — A benchmark Dataset for machine learning in ecotOxicology (Schür et al. 2023, Sci Data).

Source: Eawag ERIC open data, DOI 10.25678/0008C9, CC BY 4.0 (LICENSE file in the release).
The CKAN API (``package_show?id=adore``) lists three resources (README.md, LICENSE and the 396 MB
``adore_dataset.zip``) with sha256 hashes; the HTML landing page sits behind a bot check, the API
and the direct resource URLs do not.

This module
  * converts every CSV/TSV member of the release zip to Parquet (lossless: numbers are parsed
    only when every non-empty value round-trips; ID, CAS, split and bit-string columns stay
    strings), keeping the member's relative path;
  * loads a challenge file (``processed/<challenge>_mortality``) as a tidy frame with the
    official split columns untouched;
  * exposes the official train/test and CV-fold assignments — it never re-splits.

Split columns hold ``0``..``4`` (CV fold of a training row), ``test`` (held-out row) or
``trainvalid`` (training row without a fold label, used by the leave-one-out / leave-last-out
scaffold variants). Target: ``result_conc1_mean_mol_log`` = log10(LC50/EC50 in mol/L).
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from sentinel.core import paths

CKAN_PACKAGE_URL = "https://opendata.eawag.ch/api/3/action/package_show?id=adore"
DOI = "10.25678/0008C9"
ZIP_RESOURCE_NAME = "ADORE full dataset (396MB)"
LICENSE = "CC BY 4.0"

CHALLENGES = (
    "t-F2F",
    "t-C2C",
    "t-A2A",
    "s-F2F-1",
    "s-F2F-2",
    "s-F2F-3",
    "s-C2C",
    "s-A2A",
    "a-CA2F-same",
    "a-CA2F-diff",
    "a-FCA2FCA",
)
#: challenges that make up the SENTINEL M4 benchmark (RESEARCH_PLAN.md §5.4)
BENCHMARK_CHALLENGES = ("t-F2F", "t-C2C", "a-CA2F-same", "a-CA2F-diff")
PRIMARY_SPLIT = "split_occurrence"
SECONDARY_SPLIT = "split_scaffold-murcko"
TARGET = "result_conc1_mean_mol_log"  # log10 mol/L
TARGET_MASS = "result_conc1_mean_log"  # log10 mg/L
CHEM_KEY = "test_cas"
SPECIES_KEY = "tax_gs"
ROW_KEY = "result_id"

#: identifier / code columns that must stay strings even when every value looks numeric
_STRING_COLUMNS = re.compile(
    r"(^split_|_id$|^id$|cas|dtxsid|dtxcid|_cid$|inchi|smiles|_fp$|_number$|ncbi|taxid|code$|_op$|"
    r"^tax_gs$|^tax_all$|^tax_pdm$)",
    re.I,
)
_INT_RE = re.compile(r"^[+-]?\d+$")
_LEADING_ZERO_RE = re.compile(r"^[+-]?0\d")


# --------------------------------------------------------------------------- release handling


def zip_resource(package: dict) -> dict:
    """The resource record (url, sha256, size) of the full-dataset zip in a CKAN package_show."""
    res = package["result"]["resources"] if "result" in package else package["resources"]
    for r in res:
        if str(r.get("url", "")).endswith("adore_dataset.zip") or r.get("name") == ZIP_RESOURCE_NAME:
            return r
    raise KeyError("ADORE zip resource not found in CKAN metadata")


def _convert_column(s: pd.Series, name: str) -> pd.Series:
    """String column -> numeric only when lossless; empty strings become NaN/None."""
    if _STRING_COLUMNS.search(name):
        return s.where(s != "", None)
    nonempty = s[s != ""]
    if nonempty.empty:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    if nonempty.isin(["True", "False"]).all():
        return s.map({"True": True, "False": False}).astype("boolean")
    if nonempty.str.match(_LEADING_ZERO_RE).any() and not nonempty.str.contains(r"[.eE]").all():
        return s.where(s != "", None)  # bit strings / zero-padded codes
    num = pd.to_numeric(nonempty, errors="coerce")
    if num.isna().any():
        return s.where(s != "", None)
    if nonempty.str.match(_INT_RE).all():
        if (num.abs() < 2**53).all():
            return pd.to_numeric(s.where(s != "", None), errors="coerce").astype("Int64")
        return s.where(s != "", None)
    return pd.to_numeric(s.where(s != "", None), errors="coerce").astype("float64")


def sniff_separator(first_line: str, member_name: str) -> str:
    """Tab or comma, from the header line (``pubchem_bits.csv`` is tab-separated despite its name)."""
    tabs, commas = first_line.count("\t"), first_line.count(",")
    if tabs > commas:
        return "\t"
    if commas > tabs:
        return ","
    return "\t" if member_name.lower().endswith(".tsv") else ","


def read_member_table(raw: bytes | io.BufferedIOBase, member_name: str) -> pd.DataFrame:
    """Parse one CSV/TSV member to a typed frame (see module docstring for the rules)."""
    data = raw if isinstance(raw, bytes) else raw.read()
    first = data[:200000].split(b"\n", 1)[0].decode("utf-8", "replace")
    sep = sniff_separator(first, member_name)
    df = pd.read_csv(
        io.BytesIO(data),
        sep=sep,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        low_memory=False,
        encoding="utf-8",
    )
    df.columns = [str(c) for c in df.columns]
    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "row_index"})
    return pd.DataFrame({c: _convert_column(df[c], c) for c in df.columns})


def convert_release_zip(zip_path: str | Path, out_dir: str | Path, *, log=None) -> list[dict]:
    """Convert every member of the ADORE zip: tables -> Parquet (zstd), other files copied.

    Returns one record per member: member, output path, rows, columns, member bytes.
    """
    out_dir = Path(out_dir)
    records = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            rel = Path(*[p for p in Path(name).parts if p not in ("", ".")])
            if rel.parts and rel.parts[0].lower().startswith("adore") and len(rel.parts) > 1:
                rel = Path(*rel.parts[1:])  # strip a top-level folder if present
            if name.lower().endswith((".csv", ".tsv")):
                dest = out_dir / rel.with_suffix(".parquet")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as fh:
                    df = read_member_table(fh, name)
                df.to_parquet(dest, index=False, compression="zstd")
                rec = {
                    "member": name,
                    "path": str(dest),
                    "rows": int(len(df)),
                    "columns": int(df.shape[1]),
                    "member_bytes": int(info.file_size),
                    "member_crc32": f"{info.CRC:08x}",
                }
                if log:
                    log.info("converted %s -> %s (%d x %d)", name, dest.name, len(df), df.shape[1])
            else:
                dest = out_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as fh, open(dest, "wb") as g:
                    g.write(fh.read())
                rec = {
                    "member": name,
                    "path": str(dest),
                    "rows": None,
                    "columns": None,
                    "member_bytes": int(info.file_size),
                    "member_crc32": f"{info.CRC:08x}",
                }
            records.append(rec)
    return records


def verify_member(raw: bytes, member_name: str, parquet_path: str | Path) -> dict:
    """Cell-by-cell check that a converted Parquet equals the CSV text it came from.

    Strings must match exactly; numbers must equal ``float(text)`` exactly; empty text must be
    null. Returns counts of compared cells and mismatches per column (only columns with any).
    """
    first = raw[:200000].split(b"\n", 1)[0].decode("utf-8", "replace")
    sep = sniff_separator(first, member_name)
    txt = pd.read_csv(
        io.BytesIO(raw),
        sep=sep,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        low_memory=False,
        encoding="utf-8",
    )
    txt.columns = [str(c) for c in txt.columns]
    if txt.columns[0] in ("", "Unnamed: 0"):
        txt = txt.rename(columns={txt.columns[0]: "row_index"})
    pq_df = pd.read_parquet(parquet_path)
    bad: dict[str, int] = {}
    if list(pq_df.columns) != list(txt.columns) or len(pq_df) != len(txt):
        return {"member": member_name, "shape_ok": False, "cells": 0, "mismatches": -1, "bad_columns": {}}
    cells = 0
    for c in txt.columns:
        t = txt[c]
        v = pq_df[c]
        empty = (t == "").to_numpy()
        isnull = v.isna().to_numpy()
        n_bad = int((empty != isnull).sum())
        m = ~empty & ~isnull
        if m.any():
            if pd.api.types.is_bool_dtype(v.dtype):
                n_bad += int((t[m].to_numpy() != v[m].map({True: "True", False: "False"}).to_numpy()).sum())
            elif pd.api.types.is_numeric_dtype(v.dtype):
                n_bad += int((pd.to_numeric(t[m]).to_numpy(dtype=float) != v[m].to_numpy(dtype=float)).sum())
            else:
                n_bad += int((t[m].to_numpy() != v[m].astype(str).to_numpy()).sum())
        cells += len(t)
        if n_bad:
            bad[c] = n_bad
    return {
        "member": member_name,
        "shape_ok": True,
        "cells": cells,
        "mismatches": int(sum(bad.values())),
        "bad_columns": bad,
    }


def verify_release(zip_path: str | Path, members: list[dict], *, log=None) -> list[dict]:
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for rec in members:
            if rec["rows"] is None:
                continue
            res = verify_member(z.read(rec["member"]), rec["member"], rec["path"])
            if log:
                log.info("verify %s: %d cells, %d mismatches", rec["member"], res["cells"], res["mismatches"])
            out.append(res)
    return out


# --------------------------------------------------------------------------- loading


def release_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else paths.RAW / "adore" / "release"


def challenge_path(challenge: str, root: str | Path | None = None) -> Path:
    if challenge not in CHALLENGES:
        raise KeyError(f"unknown ADORE challenge {challenge!r}; known: {CHALLENGES}")
    return release_dir(root) / "processed" / f"{challenge}_mortality.parquet"


def feature_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    """Group challenge-file columns: metadata ('core'), split columns, and representations.

    Representations: fingerprint bit strings ``chem_<NAME>_fp`` (MACCS, Morgan, ToxPrint, pcp =
    PubChem), ``mol2vec`` (chem_mol2vec0..299) and ``mordred`` (chem_mordred_*).
    """
    groups: dict[str, list[str]] = {"core": [], "split": []}
    for c in columns:
        if c.startswith("split_"):
            groups["split"].append(c)
        elif re.fullmatch(r"chem_[A-Za-z]+_fp", c):
            groups.setdefault("fp_" + c[5:-3], []).append(c)
        elif re.fullmatch(r"chem_mol2vec\d+", c):
            groups.setdefault("mol2vec", []).append(c)
        elif c.startswith("chem_mordred_"):
            groups.setdefault("mordred", []).append(c)
        else:
            groups["core"].append(c)
    return groups


def load_challenge(
    challenge: str, *, columns: str | Iterable[str] = "core", root: str | Path | None = None
) -> pd.DataFrame:
    """Load one challenge file.

    ``columns``: ``"core"`` (metadata + target + split columns, the default), ``"all"`` (all
    1,000+ columns including fingerprints/descriptors), or an explicit list (split and key columns
    are always added).
    """
    import pyarrow.parquet as pq

    path = challenge_path(challenge, root)
    names = pq.ParquetFile(path).schema_arrow.names
    if columns == "all":
        cols = None
    elif columns == "core":
        g = feature_groups(names)
        cols = g["core"] + g["split"]
    else:
        want = list(columns)
        extra = [c for c in names if c.startswith("split_")] + [ROW_KEY, CHEM_KEY, SPECIES_KEY, TARGET]
        cols = list(dict.fromkeys([c for c in want + extra if c in names]))
    df = pd.read_parquet(path, columns=cols)
    df.attrs["challenge"] = challenge
    return df


def split_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("split_")]


def split_assignments(df: pd.DataFrame, *, key: str = ROW_KEY) -> pd.DataFrame:
    """Official split assignments in long form: key, split, assignment, role, fold.

    ``role`` is ``train`` (fold 0-4 or ``trainvalid``) or ``test``; ``fold`` is the CV fold for
    training rows that carry one (nullable integer).
    """
    rows = []
    for c in split_columns(df):
        a = df[c].astype("string")
        role = pd.Series(
            np.where(a == "test", "test", np.where(a.isna(), None, "train")), index=df.index, dtype="object"
        )
        fold = pd.to_numeric(a.where(a.str.fullmatch(r"\d+").fillna(False)), errors="coerce").astype("Int64")
        part = pd.DataFrame(
            {
                key: df[key].astype("string").to_numpy(),
                "split": c,
                "assignment": a.astype("string").to_numpy(),
                "role": role.to_numpy(),
            }
        )
        part["role"] = part["role"].astype("string")
        part["fold"] = pd.array(fold.to_numpy(dtype="float64", na_value=np.nan), dtype="Int64")
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def train_test(df: pd.DataFrame, split: str = PRIMARY_SPLIT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows of the official training and test partitions of ``split`` (no re-splitting)."""
    if split not in df.columns:
        raise KeyError(f"{split} not in this challenge (available: {split_columns(df)})")
    a = df[split].astype("string")
    test = a == "test"
    train = a.notna() & ~test
    return df[train.to_numpy()], df[test.to_numpy()]


def cv_folds(df: pd.DataFrame, split: str = PRIMARY_SPLIT) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (fold, train_positions, validation_positions) over the official folds 0..4."""
    a = df[split].astype("string")
    is_fold = a.str.fullmatch(r"\d+").fillna(False).to_numpy()
    folds = sorted(int(v) for v in a[is_fold].unique())
    for k in folds:
        val = np.flatnonzero(is_fold & (a == str(k)).to_numpy())
        trn = np.flatnonzero(is_fold & (a != str(k)).to_numpy())
        yield k, trn, val


def chemical_overlap(df: pd.DataFrame, split: str, *, chem_col: str = CHEM_KEY) -> dict:
    """Train/test row and chemical counts and the number of chemicals on both sides."""
    tr, te = train_test(df, split)
    ctr, cte = set(tr[chem_col].dropna()), set(te[chem_col].dropna())
    return {
        "split": split,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_unassigned": int(len(df) - len(tr) - len(te)),
        "n_chem_train": len(ctr),
        "n_chem_test": len(cte),
        "chem_overlap": len(ctr & cte),
        "n_species_train": int(tr[SPECIES_KEY].nunique()),
        "n_species_test": int(te[SPECIES_KEY].nunique()),
        "species_overlap": len(set(tr[SPECIES_KEY].dropna()) & set(te[SPECIES_KEY].dropna())),
        "assignment_counts": {
            str(k): int(v)
            for k, v in df[split].astype("string").fillna("<NA>").value_counts().sort_index().items()
        },
    }


def summarize_challenge(df: pd.DataFrame) -> dict:
    """Rows/chemicals/species, taxon groups, target range and per-split overlap for one challenge."""
    y = pd.to_numeric(df[TARGET], errors="coerce")
    out = {
        "challenge": df.attrs.get("challenge"),
        "rows": int(len(df)),
        "chemicals": int(df[CHEM_KEY].nunique()),
        "species": int(df[SPECIES_KEY].nunique()),
        "tax_groups": {str(k): int(v) for k, v in df["tax_group"].value_counts().items()}
        if "tax_group" in df.columns
        else {},
        "target": TARGET,
        "target_n": int(y.notna().sum()),
        "target_mean": float(y.mean()),
        "target_sd": float(y.std(ddof=0)),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "splits": {c: chemical_overlap(df, c) for c in split_columns(df)},
    }
    return out


def official_splits(challenge: str | None = None, *, root: str | Path | None = None) -> pd.DataFrame:
    """Long-form official assignments written by scripts/data/build_adore.py (processed/adore)."""
    base = Path(root) if root is not None else paths.PROCESSED / "adore"
    df = pd.read_parquet(base / "split_assignments.parquet")
    return df if challenge is None else df[df["challenge"] == challenge].reset_index(drop=True)


def fingerprint_matrix(df: pd.DataFrame, column: str = "chem_MACCS_fp") -> np.ndarray:
    """Bit-string fingerprint column (e.g. ``chem_MACCS_fp``) -> uint8 array (rows x bits)."""
    s = df[column].astype("string")
    if s.isna().any():
        raise ValueError(f"{column} has missing fingerprints")
    lens = s.str.len().unique()
    if len(lens) != 1:
        raise ValueError(f"{column}: fingerprints of different lengths {sorted(lens)}")
    buf = "".join(s.tolist()).encode("ascii")
    return (np.frombuffer(buf, dtype=np.uint8) - ord("0")).reshape(len(s), int(lens[0]))
