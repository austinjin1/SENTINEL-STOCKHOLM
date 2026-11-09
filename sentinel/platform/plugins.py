# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Model plugin registry for ``POST /predict/{modality}`` (``sentinel/platform/models.yaml``).

Contract for model workstreams (also in docs/platform.md):

* An entry in ``models.yaml`` declares ``modality``, ``model_id`` (the registry ``model``
  name), ``entry_point`` (``"package.module:callable"``), ``required_inputs``,
  ``registry_experiment`` (the registry ``experiment`` holding its validated metrics) and
  ``model_card`` (path under the repo). At most one enabled entry per modality.
* The entry point is called as ``predict(upload: ValidatedUpload) -> dict`` and must return a
  JSON-serialisable mapping with ``predictions`` (a list of records), ``units`` (mapping of
  each predicted quantity to its unit) and ``model_version``. It must not return metrics:
  skill is attached by the Console from the registry, never from the plugin.
* The Console serves an entry only when its model card exists and the registry holds at least
  one current entry for (``registry_experiment``, ``model_id``); otherwise it answers 503.
"""

from __future__ import annotations

import importlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from . import registry_view
from .config import MODALITIES, Settings
from .validation import ValidatedUpload

REQUIRED_FIELDS = (
    "modality",
    "model_id",
    "entry_point",
    "required_inputs",
    "registry_experiment",
    "model_card",
)
FORBIDDEN_OUTPUT_KEYS = {
    "metric",
    "metrics",
    "skill",
    "score",
    "scores",
    "accuracy",
    "performance",
    "validation",
    "r2",
    "nse",
    "kge",
    "auc",
    "rmse",
    "mae",
    "crps",
    "bss",
}


class PluginError(RuntimeError):
    """The plugin failed or returned output that violates the contract."""


@dataclass(frozen=True)
class PluginSpec:
    modality: str
    model_id: str
    entry_point: str
    required_inputs: tuple[str, ...]
    registry_experiment: str
    model_card: str
    enabled: bool = True
    description: str = ""
    options: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {
            "modality": self.modality,
            "model_id": self.model_id,
            "entry_point": self.entry_point,
            "required_inputs": list(self.required_inputs),
            "registry_experiment": self.registry_experiment,
            "model_card": self.model_card,
            "enabled": self.enabled,
            "description": self.description,
        }


def load_specs(s: Settings) -> tuple[list[PluginSpec], list[str]]:
    """Parse models.yaml; returns (specs, configuration errors). Never raises on bad config."""
    path = Path(s.models_yaml)
    if not path.exists():
        return [], [f"plugin registry {s.rel(path)} not found"]
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        return [], [f"plugin registry is not valid YAML ({e.__class__.__name__})"]
    entries = doc.get("models") if isinstance(doc, dict) else None
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        return [], ["'models' must be a list"]
    specs, errors = [], []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            errors.append(f"models[{i}] is not a mapping")
            continue
        missing = [k for k in REQUIRED_FIELDS if e.get(k) in (None, "", [])]
        if missing:
            errors.append(f"models[{i}] lacks {missing}")
            continue
        if e["modality"] not in MODALITIES:
            errors.append(f"models[{i}] has unknown modality {e['modality']!r}")
            continue
        if ":" not in str(e["entry_point"]):
            errors.append(f"models[{i}] entry_point must look like 'package.module:callable'")
            continue
        req = e["required_inputs"]
        specs.append(
            PluginSpec(
                modality=e["modality"],
                model_id=str(e["model_id"]),
                entry_point=str(e["entry_point"]),
                required_inputs=tuple([req] if isinstance(req, str) else [str(x) for x in req]),
                registry_experiment=str(e["registry_experiment"]),
                model_card=str(e["model_card"]),
                enabled=bool(e.get("enabled", True)),
                description=str(e.get("description", "")),
                options=dict(e.get("options") or {}),
            )
        )
    by_mod: dict[str, int] = {}
    for sp in specs:
        if sp.enabled:
            by_mod[sp.modality] = by_mod.get(sp.modality, 0) + 1
    for mod, n in by_mod.items():
        if n > 1:
            errors.append(f"more than one enabled model for modality {mod!r}; none of them will be served")
    return specs, errors


def spec_for(s: Settings, modality: str) -> tuple[PluginSpec | None, list[str]]:
    specs, errors = load_specs(s)
    enabled = [sp for sp in specs if sp.enabled and sp.modality == modality]
    if len(enabled) != 1:
        return None, errors
    return enabled[0], errors


def serving_gate(s: Settings, spec: PluginSpec) -> tuple[bool, str, list[dict]]:
    """(servable, reason, guarded registry entries for the plugin's declared experiment)."""
    card = (
        (s.repo_root / spec.model_card) if not Path(spec.model_card).is_absolute() else Path(spec.model_card)
    )
    if not card.exists():
        return False, f"model card {spec.model_card} not found", []
    df = registry_view.filtered(s, experiment=spec.registry_experiment, model=spec.model_id)
    if df.empty:
        return (
            False,
            (
                f"the results registry has no current entries for experiment "
                f"{spec.registry_experiment!r} and model {spec.model_id!r}"
            ),
            [],
        )
    return True, "ok", [e["guarded"] for e in registry_view.entries(s, df)]


_ep_lock = threading.Lock()
_ep_cache: dict[str, Callable] = {}


def resolve(spec: PluginSpec) -> Callable[[ValidatedUpload], dict]:
    with _ep_lock:
        fn = _ep_cache.get(spec.entry_point)
    if fn is not None:
        return fn
    mod_name, _, attr = spec.entry_point.partition(":")
    try:
        obj: Any = importlib.import_module(mod_name)
        for part in attr.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as e:
        raise PluginError(f"cannot import entry point {spec.entry_point!r}: {e.__class__.__name__}") from e
    if not callable(obj):
        raise PluginError(f"entry point {spec.entry_point!r} is not callable")
    with _ep_lock:
        _ep_cache[spec.entry_point] = obj
    return obj


def check_output(out: Any) -> dict:
    if not isinstance(out, dict):
        raise PluginError("plugin must return a dict")
    bad = sorted(k for k in out if str(k).lower() in FORBIDDEN_OUTPUT_KEYS)
    if bad:
        raise PluginError(f"plugin output contains {bad}; performance numbers must come from the registry")
    for k in ("predictions", "units", "model_version"):
        if k not in out:
            raise PluginError(f"plugin output lacks {k!r}")
    if not isinstance(out["predictions"], list):
        raise PluginError("'predictions' must be a list of records")
    if not isinstance(out["units"], dict) or not out["units"]:
        raise PluginError("'units' must be a non-empty mapping (units are part of a quantity's identity)")
    try:
        json.dumps(out, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise PluginError(f"plugin output is not JSON-serialisable without NaN/inf ({e})") from e
    return out


def run(spec: PluginSpec, upload: ValidatedUpload) -> dict:
    fn = resolve(spec)
    try:
        out = fn(upload)
    except PluginError:
        raise
    except Exception as e:  # noqa: BLE001 — surface plugin failures as 500 with a clear message
        raise PluginError(f"plugin {spec.model_id} raised {e.__class__.__name__}: {e}") from e
    return check_output(out)
