#!/usr/bin/env python3
"""Run the hardware/torch-free tests without pytest.

Discovers ``test_*.py`` under the package test dirs and this dir, calls every
``test_*`` function, and reports. Lets `make test` work on a bare machine
(the dev box here has no pytest). On a ROS box, plain ``pytest`` works too.
"""

import importlib.util
import pathlib
import sys
import traceback

WS = pathlib.Path(__file__).resolve().parent.parent
TEST_DIRS = [
    WS / "src" / "sentinel_camera" / "test",
    WS / "src" / "sentinel_inference" / "test",
    WS / "test",
]


def load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    total = passed = 0
    for d in TEST_DIRS:
        for path in sorted(d.glob("test_*.py")):
            mod = load(path)
            for name in [f for f in dir(mod) if f.startswith("test_")]:
                total += 1
                try:
                    getattr(mod, name)()
                    passed += 1
                except Exception:
                    print(f"FAIL {path.name}::{name}")
                    traceback.print_exc()
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
