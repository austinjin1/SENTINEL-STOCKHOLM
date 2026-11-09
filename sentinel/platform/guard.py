# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Metric guard: the only path by which the SENTINEL Console renders a metric.

SENTINEL v1's demo showed generated numbers under an "observed" label. To make that
impossible in v2, every page and endpoint that shows a performance number passes it through
:func:`guarded_metric` / :func:`render_metric` / :func:`metric_records`, which

* accept only a results-registry entry id (``results/registry.jsonl``) or a registry row,
* refuse raw numbers (``float``, ``int``, numpy scalars, ``Decimal``, numeric strings),
* refuse ids that are unknown or superseded, and
* refuse a row whose fields differ from the stored registry entry (no hand-edited rows).

The module also holds the source scanner used by ``tests/test_platform.py``: it flags metric
claims typed into platform strings (a metric name followed by a number, percentages,
plus-minus intervals, event-count fractions, claim labels) and any use of random-number
generators in platform code.
"""

from __future__ import annotations

import ast
import math
import numbers
import re
import threading
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core import paths
from ..core import registry as _registry


class MetricGuardError(ValueError):
    """Raised when something other than a current registry entry is offered as a metric."""


_COMPARE_FIELDS = (
    "value",
    "metric",
    "model",
    "experiment",
    "split",
    "target",
    "ci_low",
    "ci_high",
    "ci_method",
    "n",
    "is_baseline",
    "workstream",
)
_cache_lock = threading.Lock()
_cache: dict[str, tuple[tuple, dict, dict]] = {}


def _stamp(path: Path) -> tuple:
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return (None, None)


def _read_rows(path: Path) -> list[dict]:
    for attempt in range(3):  # a concurrent append can expose a partial last line
        try:
            return _registry.read_registry(path)
        except ValueError:
            if attempt == 2:
                raise
            time.sleep(0.05)
    return []


def registry_index(registry_path: str | Path | None = None) -> tuple[dict[str, dict], dict[str, str]]:
    """(current entries by id, superseded id -> superseding id). Cached on file mtime/size."""
    path = Path(registry_path) if registry_path else paths.REGISTRY_PATH
    key = str(path)
    stamp = _stamp(path)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] == stamp:
            return hit[1], hit[2]
    rows = _read_rows(path) if stamp[0] is not None else []
    superseded_by = {r["supersedes"]: r["id"] for r in rows if r.get("supersedes")}
    current = {r["id"]: r for r in rows if r.get("id") and r["id"] not in superseded_by}
    with _cache_lock:
        _cache[key] = (stamp, current, superseded_by)
    return current, superseded_by


@dataclass(frozen=True)
class GuardedMetric:
    """A metric that is known to be a current registry entry."""

    id: str
    workstream: str | None
    experiment: str
    model: str
    metric: str
    value: float
    split: str
    target: str | None
    ci_low: float | None
    ci_high: float | None
    ci_method: str | None
    n: int | None
    is_baseline: bool
    timestamp: str | None
    data_manifests: tuple[str, ...]

    def text(self, digits: int = 4) -> str:
        parts = [f"{self.metric} = {fmt_number(self.value, digits)}"]
        if self.ci_low is not None and self.ci_high is not None:
            ci = f"CI [{fmt_number(self.ci_low, digits)}, {fmt_number(self.ci_high, digits)}]"
            parts.append(f"{ci} ({self.ci_method})" if self.ci_method else ci)
        else:
            parts.append("no CI recorded")
        ctx = [f"split {self.split}"]
        if self.target:
            ctx.append(f"target {self.target}")
        if self.n is not None:
            ctx.append(f"n = {self.n}")
        ctx.append("baseline" if self.is_baseline else "SENTINEL model")
        return f"{', '.join(parts)}; {', '.join(ctx)}; model {self.model} [registry {self.id}]"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["data_manifests"] = list(self.data_manifests)
        d["display"] = self.text()
        d["source"] = "results/registry.jsonl"
        return d


def fmt_number(v: float | None, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{digits}g}"


def _is_raw_number(ref: Any) -> bool:
    """Numbers of any kind (strings are judged separately, after the id lookup)."""
    if isinstance(ref, (bool, numbers.Number, Decimal)):
        return True
    try:
        import numpy as np

        if isinstance(ref, (np.ndarray, np.generic)):
            return True
    except ImportError:  # pragma: no cover
        pass
    return False


def _numeric_string(ref: str) -> bool:
    try:
        float(ref.strip().rstrip("%"))
        return True
    except ValueError:
        return False


def _same(a: Any, b: Any) -> bool:
    def missing(x):
        return x is None or (isinstance(x, float) and math.isnan(x))

    if missing(a) and missing(b):
        return True
    if missing(a) != missing(b):
        return False
    if (
        isinstance(a, (numbers.Number, Decimal))
        and isinstance(b, (numbers.Number, Decimal))
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0)
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    return str(a) == str(b)


def guarded_metric(ref: Any, *, registry_path: str | Path | None = None) -> GuardedMetric:
    """Resolve ``ref`` (registry id or registry row) to a :class:`GuardedMetric`."""
    if _is_raw_number(ref):
        raise MetricGuardError(
            f"refusing to render a raw number ({type(ref).__name__}); metrics must be passed as "
            "results-registry entry ids or registry rows"
        )
    if hasattr(ref, "to_dict") and not isinstance(ref, Mapping):  # pandas Series
        ref = ref.to_dict()
    if isinstance(ref, Mapping):
        row_in = dict(ref)
        entry_id = row_in.get("id")
    elif isinstance(ref, str):
        row_in, entry_id = None, ref.strip()
    else:
        raise MetricGuardError(f"unsupported metric reference type {type(ref).__name__}")
    if not isinstance(entry_id, str) or not entry_id:
        raise MetricGuardError("registry row has no 'id'; only registry entries can be rendered")
    current, superseded_by = registry_index(registry_path)
    if entry_id in superseded_by:
        raise MetricGuardError(f"registry entry {entry_id} is superseded by {superseded_by[entry_id]}")
    stored = current.get(entry_id)
    if stored is None:
        if row_in is None and _numeric_string(entry_id):  # ids are hex and may look numeric: look up first
            raise MetricGuardError(
                "refusing to render a raw number (numeric string); metrics must be passed "
                "as results-registry entry ids or registry rows"
            )
        raise MetricGuardError(f"unknown registry entry id {entry_id!r}")
    if row_in is not None:
        for f in _COMPARE_FIELDS:
            if f in row_in and not _same(row_in[f], stored.get(f)):
                raise MetricGuardError(f"row field {f!r} does not match registry entry {entry_id}")
    value = stored.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or math.isnan(float(value)):
        raise MetricGuardError(f"registry entry {entry_id} has no numeric value")
    return GuardedMetric(
        id=entry_id,
        workstream=stored.get("workstream"),
        experiment=str(stored.get("experiment")),
        model=str(stored.get("model")),
        metric=str(stored.get("metric")),
        value=float(value),
        split=str(stored.get("split")),
        target=stored.get("target"),
        ci_low=None if stored.get("ci_low") is None else float(stored["ci_low"]),
        ci_high=None if stored.get("ci_high") is None else float(stored["ci_high"]),
        ci_method=stored.get("ci_method"),
        n=None if stored.get("n") is None else int(stored["n"]),
        is_baseline=bool(stored.get("is_baseline")),
        timestamp=stored.get("timestamp"),
        data_manifests=tuple(stored.get("data_manifests") or ()),
    )


def render_metric(ref: Any, *, registry_path: str | Path | None = None, digits: int = 4) -> str:
    return guarded_metric(ref, registry_path=registry_path).text(digits)


def metric_records(refs: Iterable[Any], *, registry_path: str | Path | None = None) -> list[dict]:
    return [guarded_metric(r, registry_path=registry_path).as_dict() for r in refs]


# --------------------------------------------------------------------------------------------
# Source scanner for typed-in metric claims (used by tests/test_platform.py)
# --------------------------------------------------------------------------------------------

_NUM = r"[-+−]?(?:\d+(?:[.,]\d+)?|[.,]\d+)"
_OP = r"(?:\s*(?:of|=|:|≈|~|is|was|>|<|≥|≤|>=|<=))?\s*"
_ACRONYMS = [
    "R²",
    "R\\^2",
    "R2",
    "NSE",
    "KGE",
    "AUC",
    "AUROC",
    "AUPRC",
    "ROC",
    "F1",
    "RMSE",
    "MAE",
    "MAPE",
    "CRPS",
    "CRPSS",
    "BSS",
    "BS",
    "POD",
    "FAR",
    "MdSA",
    "ρ",
    "MCC",
]
_WORDS = [
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "skill",
    "skill score",
    "brier score",
    "correlation",
    "spearman",
    "pearson",
    "lead time",
    "false alarm rate",
    "hit rate",
]

CLAIM_RULES: list[tuple[str, re.Pattern]] = [
    (
        "metric name followed by a number",
        re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(_ACRONYMS) + r")(?![A-Za-z0-9_])" + _OP + _NUM),
    ),
    ("metric word followed by a number", re.compile(r"(?i)\b(?:" + "|".join(_WORDS) + r")\b" + _OP + _NUM)),
    ("percentage claim", re.compile(r"(?<![\w.])" + _NUM + r"\s?(?:%|percent\b|pct\b)", re.I)),
    ("plus-minus interval", re.compile(r"±\s*" + _NUM)),
    (
        "event-count fraction",
        re.compile(
            r"\b\d+\s*/\s*\d+\s+(?:events|onsets|sites|lakes|detected|detections|alarms|hits)\b", re.I
        ),
    ),
    (
        "claim label typed into the UI",
        re.compile(
            r"(?i)\bstate[- ]of[- ]the[- ]art\b|\bSOTA\b|\bfirst[- ]in[- ]class\b|"
            r"every (?:figure|number|value)s?(?: shown)?(?: here)? (?:is|are) real|"
            r"all (?:figures|numbers|values)(?: shown)?(?: here)? are real"
        ),
    ),
    ("v1-style all-caps observed label", re.compile(r"\bOBSERVED\b")),
]

_STRIP = [
    re.compile(r"\{[^{}]*\}"),  # str.format replacement fields, e.g. "{:.1%}"
    re.compile(
        r"(?i)\b(?:width|height|max-width|min-width|left|right|top|bottom|opacity|font-size|"
        r"line-height)\s*:\s*-?\d+(?:\.\d+)?%"
    ),  # CSS layout percentages
]

RANDOM_RULE = re.compile(
    r"\bnp\.random\b|\bnumpy\.random\b|^\s*import\s+random\b|^\s*from\s+random\s+import\b|"
    r"Math\.random|\bdefault_rng\s*\(|\bfaker\b|\btorch\.rand",
    re.M,
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    snippet: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.snippet!r}"


def scan_text(text: str) -> list[tuple[str, str]]:
    """Return (rule, matched text) for every typed-in metric claim in ``text``."""
    for pat in _STRIP:
        text = pat.sub(" ", text)
    hits = []
    for rule, pat in CLAIM_RULES:
        for m in pat.finditer(text):
            hits.append((rule, m.group(0)))
    return hits


def _python_strings(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FormattedValue) and node.format_spec is not None:
            skip.update(id(n) for n in ast.walk(node.format_spec))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            out.append((node.lineno, node.value))
    return out


def _scan_targets(platform_dir: Path) -> list[Path]:
    files = sorted(
        p
        for p in platform_dir.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix in {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".html", ".css"}
    )
    return files


def scan_platform_sources(
    extra: Iterable[str | Path] = (), *, platform_dir: str | Path | None = None
) -> list[Finding]:
    """Scan the platform package (and ``extra`` docs) for typed-in metric claims and RNG use."""
    root = Path(platform_dir) if platform_dir else Path(__file__).resolve().parent
    findings: list[Finding] = []
    for f in [*_scan_targets(root), *(Path(e) for e in extra)]:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        rel = str(f)
        if f.suffix == ".py":
            for lineno, s in _python_strings(text):
                for rule, snip in scan_text(s):
                    findings.append(Finding(rel, lineno, rule, snip))
            for m in RANDOM_RULE.finditer(text):
                findings.append(
                    Finding(
                        rel,
                        text.count("\n", 0, m.start()) + 1,
                        "random-number generator in platform code",
                        m.group(0).strip(),
                    )
                )
        else:
            for lineno, line in enumerate(text.splitlines(), 1):
                for rule, snip in scan_text(line):
                    findings.append(Finding(rel, lineno, rule, snip))
    return findings


def main() -> int:  # pragma: no cover - CLI helper
    findings = scan_platform_sources(
        [paths.REPO_ROOT / "docs" / "platform.md", paths.REPO_ROOT / "docs" / "model_cards" / "README.md"]
    )
    for f in findings:
        print(f)
    print(f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
