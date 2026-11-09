# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Upload validation for ``POST /predict/{modality}``: format, schema, units and size limits.

Validation runs before any model is looked up, so users can check a file even while no model
is trained. Parsing is defensive: NPZ files are loaded with ``allow_pickle=False``, GeoTIFFs
through rasterio's in-memory reader, and tables as strings first (so missing-value codes and
non-numeric entries can be reported instead of silently coerced).

Units are part of a variable's identity (CONTRIBUTING_V2.md): sonde columns must use SENTINEL
variable names, which encode the unit (``turb_fnu`` vs ``turb_ntu``); names that do not
identify a unit (``turbidity``) are rejected with the unit-specific alternatives.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..data.sonde_daily import PLAUSIBLE
from . import variables as V
from .config import MODALITIES

EXTENSIONS = {
    "sonde": (".csv", ".txt"),
    "orbit": (".tif", ".tiff", ".npz"),
    "lite": (".tif", ".tiff", ".npz"),
    "microbe": (".csv", ".tsv", ".txt"),
    "chemtox": (".csv", ".txt", ".smi"),
    "bioassess": (".csv", ".tsv"),
}
MAX_ROWS = 1_000_000
MAX_COLUMNS = 64
MAX_RASTER_SIDE = 8192
MAX_BANDS = 64
MAX_FEATURES = 500_000
MAX_SAMPLES = 10_000
MAX_CHEMICALS = 10_000
MAX_IDENTIFIER_LEN = 500
MAX_EXAMPLES = 10

# Sentinel codes converted to NaN at parse time (CONTRIBUTING_V2.md, "Missing codes").
MISSING_CODES = {
    "-999999",
    "-999999.0",
    "-9999",
    "-9999.0",
    "-99999",
    "999999",
    "ice",
    "eqp",
    "***",
    "dis",
    "ssn",
    "nd",
    "bkw",
    "mnt",
    "rat",
    "fld",
    "pr",
    "tst",
    "zfl",
    "dry",
    "na",
    "nan",
    "null",
    "",
}
TIME_COLUMNS = ("time_utc", "time", "datetime", "date")
META_COLUMNS = {"site_id", "site_no", "monitoring_location_id"}
FEATURE_ID_COLUMNS = (
    "asv_id",
    "feature_id",
    "#otu id",
    "otu_id",
    "#otu_id",
    "asv",
    "feature",
    "sequence",
    "id",
)
TAXON_LEVELS = {
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "subfamily",
    "tribe",
    "genus",
    "species",
    "subspecies",
}
_DNA = re.compile(r"^[ACGTURYSWKMBDHVNacgturyswkmbdhvn]+$")
_CAS = re.compile(r"^(\d{2,7})-(\d{2})-(\d)$")
_SMILES_CHARS = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$:/\\.%*]+$")

FORMAT_HELP = {
    "sonde": "CSV time series. One time column: `time_utc` (ISO 8601, UTC), `time`/`datetime` "
    "(ISO 8601 with an explicit UTC offset) or `date` (YYYY-MM-DD, daily values). Value "
    "columns use SENTINEL variable names, which carry the unit: e.g. `temp_c`, `do_mgl`, "
    "`spcond_uscm`, `ph`, `turb_fnu`, `no3no2_mgl`, `chl_ugl`; daily files may add `_mean`, "
    "`_min` or `_max`. Optional `site_id` and `<variable>_qualifier` columns. Missing-value "
    "codes (e.g. -999999, Ice, Eqp) are treated as missing.",
    "orbit": "Satellite chip as GeoTIFF (bands = layers; band descriptions give band names) or NPZ "
    "with array `chip` shaped (bands, height, width) and optional `band_names`.",
    "lite": "Drone RGB+NIR image: 4-band GeoTIFF or NPZ array `rgbn` shaped (4, H, W) or (H, W, 4). "
    "Band order defaults to R,G,B,N; set option `band_order` otherwise.",
    "microbe": "ASV/feature table (CSV/TSV): first column a feature id (`asv_id`, `feature_id`, "
    "`#OTU ID` or `sequence`), one column per sample holding non-negative read counts; "
    "optional `taxonomy` and `sequence` columns.",
    "chemtox": "List of chemicals, one per line (SMILES or CAS registry numbers), or a CSV with a "
    "`smiles` and/or `cas` column.",
    "bioassess": "Long-format taxa table (CSV/TSV) with columns `sample_id`, `taxon`, `count` "
    "(non-negative); optional `taxonomic_level`, `life_stage`, `date`, `lat`, `lon`, `site_id`.",
}


@dataclass
class ValidatedUpload:
    """What a model plugin receives: parsed content plus the validation summary."""

    modality: str
    filename: str
    sha256: str
    n_bytes: int
    data: Any
    summary: dict
    warnings: list[dict]
    options: dict


@dataclass
class ValidationReport:
    modality: str
    filename: str
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    data: Any = None
    sha256: str = ""
    n_bytes: int = 0
    options: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, where: str, message: str) -> None:
        self.errors.append({"field": where, "message": message})

    def warn(self, where: str, message: str) -> None:
        self.warnings.append({"field": where, "message": message})

    def public(self) -> dict:
        return {
            "modality": self.modality,
            "filename": self.filename,
            "ok": self.ok,
            "sha256": self.sha256,
            "bytes": self.n_bytes,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self.summary,
        }

    def upload(self) -> ValidatedUpload:
        if not self.ok:
            raise ValueError("cannot build a ValidatedUpload from a failed validation")
        return ValidatedUpload(
            self.modality,
            self.filename,
            self.sha256,
            self.n_bytes,
            self.data,
            self.summary,
            self.warnings,
            self.options,
        )


def _examples(items: Iterable) -> list:
    out = []
    for x in items:
        out.append(x)
        if len(out) >= MAX_EXAMPLES:
            break
    return out


def _decode(content: bytes, rep: ValidationReport) -> str | None:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        rep.error("file", "file is not UTF-8 text")
        return None


def _read_table(text: str, filename: str, rep: ValidationReport) -> pd.DataFrame | None:
    lines = [ln for ln in text.splitlines() if not ln.startswith("#") or ln.lower().startswith("#otu")]
    if not lines:
        rep.error("file", "file is empty")
        return None
    body = "\n".join(lines)
    if filename.lower().endswith(".tsv"):
        sep = "\t"
    elif filename.lower().endswith(".csv"):
        sep = ","
    else:
        sep = "\t" if lines[0].count("\t") > lines[0].count(",") else ","
    try:
        df = pd.read_csv(
            io.StringIO(body),
            sep=sep,
            dtype=str,
            keep_default_na=False,
            nrows=MAX_ROWS + 1,
            skipinitialspace=True,
        )
    except (pd.errors.ParserError, ValueError) as e:
        rep.error("file", f"could not parse as a delimited table: {str(e)[:200]}")
        return None
    if len(df) > MAX_ROWS:
        rep.error("file", f"more than {MAX_ROWS} rows; split the file")
        return None
    if df.empty:
        rep.error("file", "table has a header but no rows")
        return None
    df.columns = [str(c).strip() for c in df.columns]
    dup = [c for c in df.columns if list(df.columns).count(c) > 1]
    if dup:
        rep.error("columns", f"duplicate column names: {sorted(set(dup))}")
        return None
    return df


def _numeric(sr: pd.Series, name: str, rep: ValidationReport, *, codes: bool = True) -> pd.Series:
    raw = sr.astype(str).str.strip()
    is_code = raw.str.lower().isin(MISSING_CODES) if codes else (raw == "")
    vals = pd.to_numeric(raw.mask(is_code), errors="coerce")
    bad = vals.isna() & ~is_code
    if bad.any():
        rep.error(name, f"{int(bad.sum())} non-numeric value(s), e.g. {_examples(raw[bad].unique())}")
    n_code = int((is_code & (raw != "")).sum())
    if n_code:
        rep.warn(
            name,
            f"{n_code} missing-value code(s) treated as missing, e.g. "
            f"{_examples(raw[is_code & (raw != '')].unique())}",
        )
    return vals.astype("float64")


# ------------------------------------------------------------------ sonde


def _validate_sonde(content: bytes, rep: ValidationReport, required: tuple[str, ...]) -> None:
    text = _decode(content, rep)
    if text is None:
        return
    df = _read_table(text, rep.filename, rep)
    if df is None:
        return
    df.columns = [c.lower() for c in df.columns]
    if len(df.columns) > MAX_COLUMNS:
        rep.error("columns", f"more than {MAX_COLUMNS} columns")
        return
    tcols = [c for c in df.columns if c in TIME_COLUMNS]
    if len(tcols) != 1:
        rep.error("columns", f"need exactly one time column out of {list(TIME_COLUMNS)}; found {tcols}")
        return
    tcol = tcols[0]
    raw_t = df[tcol].astype(str).str.strip()
    if tcol == "date":
        t = pd.to_datetime(raw_t, format="%Y-%m-%d", errors="coerce")
    else:
        if tcol != "time_utc":
            has_offset = raw_t.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
            if not has_offset.all():
                rep.error(
                    tcol,
                    "timestamps without an explicit UTC offset are ambiguous; add an offset or "
                    "name the column time_utc if the values are UTC; e.g. "
                    f"{_examples(raw_t[~has_offset].unique())}",
                )
                return
        t = pd.to_datetime(raw_t, utc=True, errors="coerce", format="ISO8601")
    bad_t = t.isna()
    if bad_t.any():
        rep.error(
            tcol, f"{int(bad_t.sum())} unparseable time value(s), e.g. {_examples(raw_t[bad_t].unique())}"
        )
        return
    stats = V.DAILY_STATS if tcol == "date" else ()
    out = pd.DataFrame({tcol: t})
    value_cols, meta_cols = [], []
    for c in df.columns:
        if c == tcol:
            continue
        if c in META_COLUMNS or c.endswith(("_qualifier", "_flag", "_cd")):
            meta_cols.append(c)
            out[c] = df[c].astype(str)
            continue
        base = V.split_column(c, stats)[0] if stats else c
        if base in V.CODE_BY_NAME:
            value_cols.append(c)
            continue
        if c in V.AMBIGUOUS:
            rep.error(c, f"column name does not identify a unit; rename it to one of {V.AMBIGUOUS[c]}")
        else:
            rep.error(
                c,
                "unknown column; value columns must be SENTINEL variable names "
                f"(one of {sorted(V.CODE_BY_NAME)}{', optionally with _mean/_min/_max' if stats else ''})",
            )
    if not value_cols:
        rep.error("columns", "no value columns with a recognised SENTINEL variable name")
    if rep.errors:
        return
    summary_cols = []
    for c in value_cols:
        vals = _numeric(df[c], c, rep)
        base = V.split_column(c, stats)[0] if stats else c
        code = V.CODE_BY_NAME[base]
        lo, hi = PLAUSIBLE.get(code, (-np.inf, np.inf))
        outside = vals.notna() & ((vals < lo) | (vals > hi))
        if outside.any():
            rep.error(
                c,
                f"{int(outside.sum())} value(s) outside the plausible range [{lo}, {hi}] for USGS "
                f"parameter {code} ({V.UNITS.get(base)}); check the units",
            )
        out[c] = vals
        summary_cols.append(
            {
                "name": c,
                "unit": V.UNITS.get(base),
                "parameter_code": code,
                "n_values": int(vals.notna().sum()),
                "n_missing": int(vals.isna().sum()),
            }
        )
    key = [tcol] + [c for c in meta_cols if c in META_COLUMNS]
    dups = out.duplicated(subset=key)
    if dups.any():
        rep.error(
            tcol,
            f"{int(dups.sum())} duplicate timestamp(s) (per site), e.g. "
            f"{_examples(out.loc[dups, tcol].astype(str).unique())}",
        )
    if not out[tcol].is_monotonic_increasing:
        rep.warn(tcol, "rows were not in time order; they were sorted")
        out = out.sort_values(key).reset_index(drop=True)
    if all(x["n_values"] == 0 for x in summary_cols):
        rep.error("values", "every value is missing")
    if required:
        need_time = [r for r in required if r in TIME_COLUMNS]
        if need_time and tcol not in need_time:
            rep.error("columns", f"model requires the time column to be one of {need_time}; found {tcol!r}")
        present = set(value_cols) | {V.split_column(c, V.DAILY_STATS)[0] for c in value_cols}
        missing_req = [r for r in required if r not in TIME_COLUMNS and r not in present]
        if missing_req:
            rep.error("columns", f"model requires columns {missing_req}")
    steps = out[tcol].drop_duplicates().diff().dropna()
    rep.summary.update(
        {
            "rows": int(len(out)),
            "time_column": tcol,
            "start": str(out[tcol].min()),
            "end": str(out[tcol].max()),
            "median_step_seconds": float(steps.median().total_seconds()) if len(steps) else None,
            "value_columns": summary_cols,
            "meta_columns": meta_cols,
        }
    )
    rep.data = out


# ------------------------------------------------------------------ rasters (orbit, lite)


def _read_npz(
    content: bytes, rep: ValidationReport, keys: tuple[str, ...]
) -> tuple[np.ndarray | None, list | None]:
    try:
        npz = np.load(io.BytesIO(content), allow_pickle=False)
    except (ValueError, OSError, EOFError) as e:
        rep.error("file", f"not a readable NPZ archive without pickled objects ({e.__class__.__name__})")
        return None, None
    names = list(npz.files)
    key = next((k for k in keys if k in names), None)
    if key is None:
        arrays = [n for n in names if n not in ("band_names", "crs", "transform")]
        if len(arrays) == 1:
            key = arrays[0]
        else:
            rep.error("file", f"NPZ must contain one of {list(keys)}; found {names}")
            return None, None
    try:
        arr = npz[key]
        band_names = [str(b) for b in npz["band_names"].tolist()] if "band_names" in names else None
    except ValueError as e:
        rep.error("file", f"NPZ member could not be read without pickle ({e})")
        return None, None
    return arr, band_names


def _read_tiff(content: bytes, rep: ValidationReport) -> tuple[np.ndarray | None, list | None, dict]:
    try:
        from rasterio.io import MemoryFile
    except ImportError:
        rep.error("file", "GeoTIFF support needs the rasterio package on the server; upload NPZ instead")
        return None, None, {}
    try:
        with MemoryFile(content) as mem, mem.open() as ds:
            if ds.count > MAX_BANDS or ds.width > MAX_RASTER_SIDE or ds.height > MAX_RASTER_SIDE:
                rep.error(
                    "file",
                    f"raster too large ({ds.count} bands, {ds.width}x{ds.height}); limits: "
                    f"{MAX_BANDS} bands, {MAX_RASTER_SIDE} px per side",
                )
                return None, None, {}
            arr = ds.read()
            desc = list(ds.descriptions or [])
            names = desc if desc and all(desc) else None
            geo = {
                "crs": ds.crs.to_string() if ds.crs else None,
                "transform": list(ds.transform)[:6] if ds.transform else None,
                "nodata": ds.nodata,
                "driver": ds.driver,
            }
            if ds.nodata is not None:
                arr = (
                    np.where(arr == ds.nodata, np.nan, arr.astype("float64"))
                    if arr.dtype.kind == "f"
                    else arr
                )
            return arr, names, geo
    except Exception as e:  # noqa: BLE001 — rasterio raises several error types for bad files
        rep.error("file", f"not a readable GeoTIFF ({e.__class__.__name__})")
        return None, None, {}


def _check_array(arr: np.ndarray, rep: ValidationReport) -> np.ndarray | None:
    if arr.dtype.kind not in "iuf":
        rep.error("array", f"array dtype {arr.dtype} is not numeric")
        return None
    if arr.ndim != 3:
        rep.error("array", f"expected a 3-D array (bands, height, width); got shape {list(arr.shape)}")
        return None
    c, h, w = arr.shape
    if not (1 <= c <= MAX_BANDS) or h > MAX_RASTER_SIDE or w > MAX_RASTER_SIDE or h < 2 or w < 2:
        rep.error(
            "array", f"shape {list(arr.shape)} outside limits (1-{MAX_BANDS} bands, 2-{MAX_RASTER_SIDE} px)"
        )
        return None
    if arr.dtype.kind == "f":
        finite = np.isfinite(arr)
        if not finite.any():
            rep.error("array", "every pixel is NaN/inf")
            return None
        if not finite.all():
            rep.warn("array", f"{int((~finite).sum())} non-finite pixel value(s)")
    return arr


def _validate_orbit(content: bytes, rep: ValidationReport, required: tuple[str, ...]) -> None:
    geo: dict = {}
    if rep.filename.lower().endswith(".npz"):
        arr, names = _read_npz(content, rep, ("chip", "bands", "image"))
    else:
        arr, names, geo = _read_tiff(content, rep)
    if arr is None:
        return
    arr = _check_array(arr, rep)
    if arr is None:
        return
    if names is not None and len(names) != arr.shape[0]:
        rep.error("band_names", f"{len(names)} band names for {arr.shape[0]} bands")
        return
    if required:
        if names is None:
            rep.error(
                "band_names",
                f"model requires named bands {list(required)}; provide band_names (NPZ) "
                "or band descriptions (GeoTIFF)",
            )
        else:
            miss = [b for b in required if b not in names]
            if miss:
                rep.error("band_names", f"model requires bands {miss}, not present in {names}")
    if geo.get("crs") is None:
        rep.warn("georeference", "no CRS supplied; the chip's location is unknown")
    rep.summary.update(
        {
            "bands": int(arr.shape[0]),
            "height": int(arr.shape[1]),
            "width": int(arr.shape[2]),
            "dtype": str(arr.dtype),
            "band_names": names,
            **{k: v for k, v in geo.items()},
        }
    )
    rep.data = {"array": arr, "band_names": names, "georeference": geo}


def _validate_lite(content: bytes, rep: ValidationReport, required: tuple[str, ...], options: dict) -> None:
    geo: dict = {}
    if rep.filename.lower().endswith(".npz"):
        arr, names = _read_npz(content, rep, ("rgbn", "image", "chip"))
    else:
        arr, names, geo = _read_tiff(content, rep)
    if arr is None:
        return
    if arr.ndim == 3 and arr.shape[-1] == 4 and arr.shape[0] != 4:
        arr = np.moveaxis(arr, -1, 0)
    arr = _check_array(arr, rep)
    if arr is None:
        return
    if arr.shape[0] != 4:
        rep.error("bands", f"RGB+NIR input needs exactly 4 bands; got {arr.shape[0]}")
        return
    order_raw = options.get("band_order") or (",".join(names) if names else "R,G,B,N")
    order = [b.strip().upper() for b in str(order_raw).split(",")]
    order = ["N" if b in ("NIR", "N") else b for b in order]
    if sorted(order) != ["B", "G", "N", "R"]:
        rep.error("band_order", f"band order must be a permutation of R,G,B,N; got {order_raw!r}")
        return
    arr = arr[[order.index(b) for b in ("R", "G", "B", "N")]]
    rep.summary.update(
        {
            "bands": ["R", "G", "B", "N"],
            "input_band_order": order,
            "height": int(arr.shape[1]),
            "width": int(arr.shape[2]),
            "dtype": str(arr.dtype),
            **geo,
        }
    )
    rep.data = {"rgbn": arr, "georeference": geo}


# ------------------------------------------------------------------ microbe


def _validate_microbe(content: bytes, rep: ValidationReport, required: tuple[str, ...]) -> None:
    text = _decode(content, rep)
    if text is None:
        return
    df = _read_table(text, rep.filename, rep)
    if df is None:
        return
    first = df.columns[0]
    if first.lower() not in FEATURE_ID_COLUMNS:
        rep.error(
            "columns", f"first column must be a feature id ({list(FEATURE_ID_COLUMNS)}); found {first!r}"
        )
        return
    ids = df[first].astype(str).str.strip()
    if (ids == "").any():
        rep.error(first, f"{int((ids == '').sum())} empty feature id(s)")
    if ids.duplicated().any():
        rep.error(first, f"duplicate feature ids, e.g. {_examples(ids[ids.duplicated()].unique())}")
    side = [c for c in df.columns[1:] if c.lower() in ("taxonomy", "sequence", "confidence")]
    samples = [c for c in df.columns[1:] if c not in side]
    if not samples:
        rep.error("columns", "no sample columns")
        return
    if len(samples) > MAX_SAMPLES or len(df) > MAX_FEATURES:
        rep.error(
            "file",
            f"table too large ({len(df)} features x {len(samples)} samples); limits "
            f"{MAX_FEATURES} x {MAX_SAMPLES}",
        )
        return
    seq_col = (
        first if first.lower() == "sequence" else next((c for c in side if c.lower() == "sequence"), None)
    )
    if seq_col is not None:
        bad = ~df[seq_col].astype(str).str.strip().str.match(_DNA)
        if bad.any():
            rep.error(seq_col, f"{int(bad.sum())} sequence(s) with non-nucleotide characters")
    counts = pd.DataFrame({c: pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce") for c in samples})
    nonnum = counts.isna()
    if nonnum.any().any():
        rep.error("counts", f"{int(nonnum.sum().sum())} non-numeric or empty count cell(s)")
        return
    if (counts < 0).any().any():
        rep.error("counts", "negative counts")
        return
    if not np.allclose(counts.to_numpy(), np.round(counts.to_numpy())):
        rep.warn(
            "counts",
            "counts are not integers; if these are relative abundances, a model may need raw read counts",
        )
    empty = [c for c in samples if counts[c].sum() == 0]
    if empty:
        rep.warn("samples", f"{len(empty)} sample(s) with zero total count, e.g. {_examples(empty)}")
    miss = [r for r in required if r not in df.columns]
    if miss:
        rep.error("columns", f"model requires columns {miss}")
    counts.index = ids
    rep.summary.update(
        {
            "features": int(len(df)),
            "samples": len(samples),
            "feature_id_column": first,
            "side_columns": side,
            "sample_names": _examples(samples),
        }
    )
    rep.data = {"counts": counts, "side": df[side].set_index(ids) if side else None}


# ------------------------------------------------------------------ chemtox


def cas_checksum_ok(cas: str) -> bool:
    m = _CAS.match(cas)
    if not m:
        return False
    digits = (m.group(1) + m.group(2))[::-1]
    return sum((i + 1) * int(d) for i, d in enumerate(digits)) % 10 == int(m.group(3))


def _smiles_ok(s: str) -> tuple[bool, str]:
    if len(s) > MAX_IDENTIFIER_LEN:
        return False, "too long"
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return (Chem.MolFromSmiles(s) is not None), "rdkit"
    except ImportError:
        pass
    if not _SMILES_CHARS.match(s):
        return False, "syntax"
    depth = 0
    for ch in s:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            return False, "syntax"
    return depth == 0 and s.count("[") == s.count("]"), "syntax"


def _validate_chemtox(content: bytes, rep: ValidationReport, required: tuple[str, ...]) -> None:
    text = _decode(content, rep)
    if text is None:
        return
    lines = [ln.strip() for ln in text.splitlines()]
    header = lines[0].lower() if lines else ""
    entries: list[tuple[int, str, str | None]] = []  # (line, identifier, declared kind)
    if any(h in [x.strip() for x in re.split(r"[,\t]", header)] for h in ("smiles", "cas")):
        df = _read_table(
            text, rep.filename if rep.filename.lower().endswith((".csv", ".tsv")) else "x.csv", rep
        )
        if df is None:
            return
        cols = {c.lower(): c for c in df.columns}
        for kind in ("smiles", "cas"):
            if kind in cols:
                for i, v in enumerate(df[cols[kind]].astype(str).str.strip(), start=2):
                    if v:
                        entries.append((i, v, kind))
    else:
        for i, v in enumerate(lines, start=1):
            if v and not v.startswith("#"):
                entries.append((i, v.split()[0], None))
    if not entries:
        rep.error("file", "no chemical identifiers found")
        return
    if len(entries) > MAX_CHEMICALS:
        rep.error("file", f"more than {MAX_CHEMICALS} identifiers")
        return
    parsed, bad, method = [], [], None
    for line, ident, kind in entries:
        looks_cas = bool(_CAS.match(ident))
        if kind == "cas" or (kind is None and looks_cas):
            if cas_checksum_ok(ident):
                parsed.append({"line": line, "identifier": ident, "kind": "cas"})
            else:
                bad.append(f"line {line}: {ident!r} is not a valid CAS RN (format or check digit)")
            continue
        ok, method = _smiles_ok(ident)
        if ok:
            parsed.append({"line": line, "identifier": ident, "kind": "smiles"})
        else:
            bad.append(f"line {line}: {ident[:60]!r} is not a parseable SMILES")
    if bad:
        rep.error("identifiers", f"{len(bad)} invalid identifier(s): {_examples(bad)}")
    if method == "syntax":
        rep.warn("identifiers", "RDKit is not installed on the server; SMILES were only syntax-checked")
    kinds = {p["kind"] for p in parsed}
    if required and not kinds <= set(required):
        rep.error("identifiers", f"model accepts only {list(required)} identifiers; found {sorted(kinds)}")
    ids = [p["identifier"] for p in parsed]
    if len(set(ids)) < len(ids):
        rep.warn("identifiers", f"{len(ids) - len(set(ids))} duplicate identifier(s)")
    rep.summary.update(
        {
            "identifiers": len(parsed),
            "cas": sum(p["kind"] == "cas" for p in parsed),
            "smiles": sum(p["kind"] == "smiles" for p in parsed),
        }
    )
    rep.data = parsed


# ------------------------------------------------------------------ bioassess


def _validate_bioassess(content: bytes, rep: ValidationReport, required: tuple[str, ...]) -> None:
    text = _decode(content, rep)
    if text is None:
        return
    df = _read_table(text, rep.filename, rep)
    if df is None:
        return
    df.columns = [c.lower() for c in df.columns]
    need = ["sample_id", "taxon", "count"]
    miss = [c for c in need + [r for r in required if r not in need] if c not in df.columns]
    if miss:
        rep.error("columns", f"missing required column(s) {miss}")
        return
    for c in ("sample_id", "taxon"):
        empty = df[c].astype(str).str.strip() == ""
        if empty.any():
            rep.error(c, f"{int(empty.sum())} empty value(s)")
    cnt = pd.to_numeric(df["count"].astype(str).str.strip(), errors="coerce")
    if cnt.isna().any():
        rep.error("count", f"{int(cnt.isna().sum())} non-numeric or empty count(s)")
    elif (cnt < 0).any():
        rep.error("count", "negative counts")
    elif not np.allclose(cnt, np.round(cnt)):
        rep.warn("count", "counts are not integers (densities?); a model may require raw counts")
    if "taxonomic_level" in df.columns:
        lv = df["taxonomic_level"].astype(str).str.strip().str.lower()
        badlv = ~lv.isin(TAXON_LEVELS) & (lv != "")
        if badlv.any():
            rep.error(
                "taxonomic_level",
                f"unknown level(s) {_examples(lv[badlv].unique())}; use {sorted(TAXON_LEVELS)}",
            )
    for c, lo, hi in (("lat", -90, 90), ("lon", -180, 180)):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            if (v.notna() & ~v.between(lo, hi)).any() or (
                v.isna() & (df[c].astype(str).str.strip() != "")
            ).any():
                rep.error(c, f"{c} must be numeric decimal degrees in [{lo}, {hi}]")
    if "date" in df.columns:
        d = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y-%m-%d", errors="coerce")
        if (d.isna() & (df["date"].astype(str).str.strip() != "")).any():
            rep.error("date", "dates must be YYYY-MM-DD")
    key = ["sample_id", "taxon"] + (["life_stage"] if "life_stage" in df.columns else [])
    dup = df.duplicated(subset=key)
    if dup.any():
        rep.error("rows", f"{int(dup.sum())} duplicate {key} row(s); aggregate them before upload")
    if rep.errors:
        return
    out = df.copy()
    out["count"] = cnt
    rep.summary.update(
        {
            "rows": int(len(out)),
            "samples": int(out["sample_id"].nunique()),
            "taxa": int(out["taxon"].nunique()),
            "columns": list(out.columns),
        }
    )
    rep.data = out


_VALIDATORS = {
    "sonde": _validate_sonde,
    "orbit": _validate_orbit,
    "microbe": _validate_microbe,
    "chemtox": _validate_chemtox,
    "bioassess": _validate_bioassess,
}


def validate(
    modality: str,
    filename: str,
    content: bytes,
    *,
    options: dict | None = None,
    required_inputs: Iterable[str] = (),
) -> ValidationReport:
    """Validate an upload for ``modality``. Size limits are enforced by the caller (API)."""
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")
    rep = ValidationReport(modality=modality, filename=filename or "", options=dict(options or {}))
    rep.n_bytes = len(content)
    rep.sha256 = hashlib.sha256(content).hexdigest()
    name = (filename or "").lower()
    if not name.endswith(EXTENSIONS[modality]):
        rep.error(
            "file", f"file type not accepted for {modality}; expected one of {list(EXTENSIONS[modality])}"
        )
        return rep
    if not content:
        rep.error("file", "file is empty")
        return rep
    req = tuple(required_inputs)
    if modality == "lite":
        _validate_lite(content, rep, req, rep.options)
    else:
        _VALIDATORS[modality](content, rep, req)
    return rep
