# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""National track: every repo file in the import closure of the national test scorer is in the national
scoring-code hash list (lead order of 2026-11-10 19:42 EST; condition of the national freeze).

Imports only: the scorer module is loaded, nothing is scored and no data row is read. The 4-state counterpart
(``tests/test_lite.py::test_scoring_hash_covers_scorer_import_closure``) is left as it is — it records deviation 10
of ``lite_arch_v2_deviations.md``."""

from __future__ import annotations

import os
import subprocess
import sys

from sentinel.core import paths
from sentinel.models.lite import guard as G
from sentinel.models.lite import national_testlook as NT

SCORER = "scripts/eval/national_lite_score_test.py"
# the scorer, then the modules it imports lazily inside its functions (teacher, comparator, look guard, event layer)
CODE = ("import importlib.util,sys;s=importlib.util.spec_from_file_location('s',sys.argv[2]);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "import sentinel.models.orbit.teacher, sentinel.models.orbit.satwq, sentinel.models.lite.testlook;"
        "import sentinel.models.lite.national_testlook;"
        "import sentinel.events.lite as EL; EL.EV();"
        "r=sys.argv[1];print('\\n'.join(sorted({v.__file__[len(r):] for v in list(sys.modules.values()) "
        "if getattr(v,'__file__',None) and v.__file__.startswith(r)})))")


def _closure() -> list[str]:
    root = str(paths.REPO_ROOT) + "/"
    r = subprocess.run([sys.executable, "-B", "-c", CODE, root, SCORER], capture_output=True, text=True, cwd=root,
                       check=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
    return r.stdout.split()


def test_national_hash_list_covers_the_scorer_import_closure():
    closure = _closure()
    hashed = set(NT.national_frozen_files())
    assert len(closure) > 20 and SCORER.replace("scripts/eval/", "") not in closure
    missing = [f for f in closure if f not in hashed]
    assert not missing, f"in the national scorer's import closure but not hashed: {missing}"
    for f in ("sentinel/data/matchups.py", "sentinel/models/orbit/wqp_surface.py", "sentinel/models/orbit/data.py",
              "sentinel/models/litev2/__init__.py", "sentinel/models/litev2/tests_mask.py", SCORER):
        assert f in hashed
    assert all((paths.REPO_ROOT / f).is_file() for f in hashed), "a hashed file does not exist"


def test_the_four_state_list_is_not_changed_by_the_national_additions():
    """The additions live in ``NATIONAL_CODE`` only: ``FREEZE_lite.json`` and its hash list stay as they were."""
    four = set(G.frozen_files())
    for f in ("sentinel/data/matchups.py", "sentinel/models/orbit/wqp_surface.py", "sentinel/models/litev2/__init__.py"):
        assert f in NT.NATIONAL_CODE and f not in four and f not in G.SHARED_CODE
    assert four < set(NT.national_frozen_files())
