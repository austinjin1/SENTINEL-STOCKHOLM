# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D4 TapRisk app registration, kept self-contained until the five pending downstream test looks are done
(lead instruction 2026-11-08: common.py is in their scoring hashes and must not change before then).

After the looks the lead's plan is: add ``"d4": "taprisk"`` to ``common.APPS`` and put the files in
``EXTRA_SCORING_FILES`` into ``common.APP_SCORING_FILES["taprisk"]`` before any D4 freeze; until then any D4
record uses :func:`scoring_files` / :func:`scoring_code_sha256` below.
"""

from __future__ import annotations

import hashlib

from ...core import paths
from . import common as C

APP = "taprisk"
APP_ID = "d4"
EXTRA_SCORING_FILES = (
    "sentinel/models/downstream/taprisk_models.py",
    "sentinel/models/downstream/taprisk_registry.py",
    "sentinel/models/downstream/wqp_track.py",          # huc_of_points, fit_logistic, fit_gbm
    "scripts/data/build_taprisk_links.py",
    "sentinel/data/downstream_links.py",
    "docs/preregistration/downstream_taprisk_split.json",
)


def scoring_files() -> list[str]:
    return sorted(set(C.scoring_files(APP)) | set(EXTRA_SCORING_FILES))


def scoring_code_sha256() -> str:
    h = hashlib.sha256()
    for rel in scoring_files():
        fp = paths.REPO_ROOT / rel
        h.update(rel.encode())
        h.update(fp.read_bytes() if fp.is_file() else b"<missing>")
    return h.hexdigest()
