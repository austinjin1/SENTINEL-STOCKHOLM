# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL Console HTTP API (FastAPI). Run with ``python -m sentinel.platform.api``."""

from .app import create_app

__all__ = ["create_app"]
