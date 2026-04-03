"""
SENTINEL configuration loader.

Loads YAML configuration from configs/default.yaml with support for
overrides via CLI arguments and environment variables.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"

_cached_config: dict[str, Any] | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply environment-variable overrides.

    Environment variables matching ``SENTINEL__<SECTION>__<KEY>`` (double
    underscore separators) override the corresponding nested config value.
    Values are cast to int/float/bool when possible.

    Example::

        SENTINEL__DATA__SATELLITE__CLOUD_THRESHOLD=15
    """
    prefix = "SENTINEL__"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = [p.lower() for p in env_key[len(prefix) :].split("__")]
        # Navigate/create nested dicts
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # Cast value
        node[parts[-1]] = _cast_value(env_val)
    return cfg


def _cast_value(v: str) -> Any:
    """Attempt to cast a string to int, float, or bool."""
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _apply_cli_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dot-delimited CLI overrides.

    Each override has the form ``key.subkey=value``, e.g.
    ``data.satellite.cloud_threshold=15``.
    """
    for item in overrides:
        if "=" not in item:
            continue
        key_path, raw_value = item.split("=", 1)
        parts = key_path.strip().split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _cast_value(raw_value)
    return cfg


def load_config(
    config_path: str | Path | None = None,
    cli_args: list[str] | None = None,
    override_dict: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Load the SENTINEL configuration.

    Parameters
    ----------
    config_path:
        Path to a YAML config file.  Defaults to ``configs/default.yaml``.
    cli_args:
        List of ``key.subkey=value`` overrides (typically from argparse).
    override_dict:
        Dict that is deep-merged on top of the file config.
    use_cache:
        If *True*, return a cached copy on subsequent calls (the cache is
        invalidated when *config_path* or *override_dict* are provided).
    """
    global _cached_config
    if use_cache and _cached_config is not None and config_path is None and override_dict is None:
        return _cached_config

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    if override_dict:
        cfg = _deep_merge(cfg, override_dict)

    cfg = _apply_env_overrides(cfg)

    if cli_args:
        cfg = _apply_cli_overrides(cfg, cli_args)

    if use_cache:
        _cached_config = cfg
    return cfg


def get_config_section(*keys: str, config: dict[str, Any] | None = None) -> Any:
    """Retrieve a nested config section by key path.

    Example::

        get_config_section("data", "satellite", "bands")
    """
    cfg = config if config is not None else load_config()
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict):
            raise KeyError(f"Cannot descend into non-dict at key {k!r}")
        node = node[k]
    return node


def build_argparser() -> argparse.ArgumentParser:
    """Return an ArgumentParser that accepts ``--config`` and ``--set``."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. --set data.satellite.cloud_threshold=15",
    )
    return parser
