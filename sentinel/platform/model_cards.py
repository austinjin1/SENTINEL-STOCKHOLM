# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Model cards (``docs/model_cards/*.md`` with YAML front matter) joined to registry entries.

A card never contains metrics. It names the registry ``model`` values it documents
(``registry_models``) and optionally the experiments (``registry_experiments``); the Console
attaches every *current* registry entry for those models, resolved through the metric guard.
``README.md`` (the template) and files starting with ``_`` are not cards.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from . import registry_view
from .config import Settings

REQUIRED_KEYS = ("model_id", "name", "modality", "registry_models")
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)(.*)\Z", re.S)


def parse_card(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    m = _FRONT.match(text)
    card: dict[str, Any] = {"file": path.name}
    if not m:
        card["error"] = "no YAML front matter (file must start with a '---' block)"
        card["body"] = text
        return card
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        card["error"] = f"front matter is not valid YAML: {e.__class__.__name__}"
        card["body"] = m.group(2)
        return card
    if not isinstance(meta, dict):
        card["error"] = "front matter must be a mapping"
        return card
    missing = [k for k in REQUIRED_KEYS if not meta.get(k)]
    if missing:
        card["error"] = f"front matter lacks required keys: {missing}"
    rm = meta.get("registry_models") or []
    meta["registry_models"] = [rm] if isinstance(rm, str) else [str(x) for x in rm]
    re_ = meta.get("registry_experiments") or []
    meta["registry_experiments"] = [re_] if isinstance(re_, str) else [str(x) for x in re_]
    card.update(meta)
    card["body"] = m.group(2).strip()
    return card


def card_files(s: Settings) -> list[Path]:
    d = s.model_cards_dir
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.md") if p.name.lower() != "readme.md" and not p.name.startswith("_"))


def load_cards(s: Settings) -> list[dict]:
    return [parse_card(p) for p in card_files(s)]


def cards_with_metrics(s: Settings) -> dict:
    cards = load_cards(s)
    reg = registry_view.frame(s)
    covered: set[str] = set()
    for c in cards:
        models = c.get("registry_models") or []
        covered.update(models)
        if reg.empty or not models:
            c["registry_entries"] = []
            continue
        sel = reg[reg["model"].astype(str).isin(models)]
        exps = c.get("registry_experiments") or []
        if exps:
            sel = sel[sel["experiment"].astype(str).isin(exps)]
        c["registry_entries"] = registry_view.entries(s, sel)
    uncarded = []
    if not reg.empty:
        rest = reg[~reg["model"].astype(str).isin(covered)]
        for model, g in rest.groupby("model"):
            uncarded.append({"model": model, "registry_entries": registry_view.entries(s, g)})
    return {
        "cards": cards,
        "registry_models_without_card": uncarded,
        "model_cards_dir": s.rel(s.model_cards_dir),
    }
