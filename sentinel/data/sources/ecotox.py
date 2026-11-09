# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EPA ECOTOX Knowledgebase — quarterly ASCII release -> Parquet tables -> tidy aquatic endpoints.

Source: https://gaftp.epa.gov/ecotox/ecotox_ascii_MM_DD_YYYY.zip (pipe-delimited ASCII, one file
per table; ``release_notes.txt`` lists each table's record and column counts, which are checked
after parsing). Documentation: the ``ECOTOX 5 ASCII Download`` data dictionary shipped as a PDF
in the release. Aquatic records are those with ``tests.organism_habitat == 'Water'``.

Raw tables are stored as all-string Parquet (exactly the published text, '' -> null) so that
codes such as ``NR``/``NC``/``--`` and operators survive; typing and unit standardisation happen
only in :mod:`sentinel.data.chemtox_ecotox`.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from sentinel.core import paths

INDEX_URL = "https://gaftp.epa.gov/ecotox/"
LICENSE = "US Government work (public domain status not explicitly stated by EPA)"
_RELEASE_RE = re.compile(r"ecotox_ascii_(\d{2})_(\d{2})_(\d{4})\.zip")

CORE_TABLES = ("tests", "results", "species", "chemicals", "references", "media_characteristics")


def releases_from_index(html: str) -> list[tuple[str, str]]:
    """(YYYY-MM-DD, filename) for every ASCII release linked from the directory index, sorted."""
    found = {}
    for m in _RELEASE_RE.finditer(html):
        mm, dd, yyyy = m.groups()
        found[f"{yyyy}-{mm}-{dd}"] = m.group(0)
    return sorted(found.items())


def latest_release(html: str) -> tuple[str, str]:
    rel = releases_from_index(html)
    if not rel:
        raise RuntimeError("no ecotox_ascii_*.zip found in the ECOTOX index page")
    return rel[-1]


def parse_release_notes(text: str) -> dict[str, dict[str, int]]:
    """``release_notes.txt`` -> {file_name: {"records": n, "columns": k}}."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_]+\.txt)\s+(\d+)\s+(\d+)\s*$", line)
        if m:
            out[m.group(1)] = {"records": int(m.group(2)), "columns": int(m.group(3))}
    return out


def decode_table(raw: bytes) -> tuple[str, str]:
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), "cp1252"


def read_pipe_table(raw: bytes) -> tuple[pd.DataFrame, dict]:
    """Parse one pipe-delimited ECOTOX table as strings ('' -> null); no quoting rules apply.

    Lines whose field count differs from the header are counted and reported, not guessed at.
    """
    text, enc = decode_table(raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    header = lines[0].split("|")
    ncol = len(header)
    good, bad, repaired = [], 0, 0
    for ln in lines[1:]:
        parts = ln.split("|")
        if len(parts) == ncol:
            good.append(parts)
        elif len(parts) == ncol + 1 and parts[-1] == "":
            good.append(parts[:-1])  # trailing delimiter
        else:
            fixed = merge_quoted_pipes(parts)
            if len(fixed) == ncol:
                good.append(fixed)
                repaired += 1
            else:
                bad += 1
    df = pd.DataFrame(good, columns=[h.strip() for h in header], dtype="string")
    df = df.replace("", pd.NA)
    return df, {
        "encoding": enc,
        "rows": len(good),
        "columns": ncol,
        "bad_lines": bad,
        "repaired_lines": repaired,
    }


def merge_quoted_pipes(parts: list[str]) -> list[str]:
    """Re-join fields split by a '|' inside a double-quoted value, keeping the text verbatim.

    One results.txt record (2026-09-15 release) has ``"UNCHARACTERISED PROTEIN (TR|M7BW30|...)"``
    in a comment field; plain splitting gives two extra fields.
    """
    out, buf = [], None
    for p in parts:
        if buf is None:
            if p.startswith('"') and not (len(p) > 1 and p.endswith('"')):
                buf = [p]
            else:
                out.append(p)
        else:
            buf.append(p)
            if p.endswith('"'):
                out.append("|".join(buf))
                buf = None
    if buf is not None:  # unbalanced quote: give up, return the plain split
        return parts
    return out


def convert_release_zip(zip_path: str | Path, out_dir: str | Path, *, log=None) -> dict:
    """Convert every .txt table of a release zip to Parquet; copy other members (PDF help)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"tables": {}, "other": []}
    with zipfile.ZipFile(zip_path) as z:
        names = [i for i in z.infolist() if not i.is_dir()]
        notes = [i for i in names if re.search(r"release_notes[^/]*\.txt$", i.filename)]
        if notes:
            report["release_notes"] = parse_release_notes(z.read(notes[0]).decode("utf-8", "replace"))
            report["release_notes_text"] = z.read(notes[0]).decode("utf-8", "replace")
        for info in names:
            base = Path(info.filename).name
            if base.endswith(".txt") and not base.startswith("release_notes"):
                df, meta = read_pipe_table(z.read(info))
                table = base[:-4]
                dest = out_dir / f"{table}.parquet"
                df.to_parquet(dest, index=False, compression="zstd")
                meta.update({"member": info.filename, "path": str(dest), "member_bytes": info.file_size})
                exp = report.get("release_notes", {}).get(base)
                meta["expected_found"] = exp is not None
                if exp:
                    meta["expected_records"] = exp["records"]
                    meta["expected_columns"] = exp["columns"]
                    meta["counts_match"] = (
                        exp["records"] == meta["rows"] and exp["columns"] == meta["columns"]
                    )
                report["tables"][table] = meta
                if log:
                    log.info(
                        "%s: %d rows x %d cols (bad lines %d, %s) %s",
                        table,
                        meta["rows"],
                        meta["columns"],
                        meta["bad_lines"],
                        meta["encoding"],
                        "" if meta.get("counts_match", True) else "COUNT MISMATCH vs release notes",
                    )
            else:
                dest = out_dir / base
                with open(dest, "wb") as g:
                    g.write(z.read(info))
                report["other"].append({"member": info.filename, "path": str(dest), "bytes": info.file_size})
    return report


def table_dir(release: str | None = None, root: str | Path | None = None) -> Path:
    """Directory of the converted tables of ``release`` (YYYY-MM-DD); default = newest present."""
    base = Path(root) if root is not None else paths.RAW / "ecotox"
    if release is None:
        cands = sorted(p for p in base.glob("release_*") if p.is_dir())
        if not cands:
            raise FileNotFoundError(f"no converted ECOTOX release under {base}")
        return cands[-1]
    return base / f"release_{release}"


def read_table(
    name: str, *, release: str | None = None, columns: list[str] | None = None, root: str | Path | None = None
) -> pd.DataFrame:
    return pd.read_parquet(table_dir(release, root) / f"{name}.parquet", columns=columns)


def read_csv_strings(text: str, sep: str = "|") -> pd.DataFrame:
    """Small helper for fixtures: pipe text -> string frame with '' -> null."""
    df = pd.read_csv(
        io.StringIO(text), sep=sep, dtype="string", keep_default_na=False, quoting=csv.QUOTE_NONE
    )
    return df.replace("", pd.NA)
