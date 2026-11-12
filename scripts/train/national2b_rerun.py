#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Re-run an existing national-2 configuration on the corrected labels of the national-2 chips (label version
national-2b; lead order 2026-11-10 15:05 EST item 2(c); addendum section (vii)).

A logged re-run, not a new configuration: every setting, seed and code path is that of ``national2_run.py``; only the
label table differs (``labels2b.activate_rerun``), the 4 train chips at reserved new-site sites are not loaded
(deviations entry 3), and the run is named ``..._national2b`` and registered under ``lite_val_national2b``.
``national2_run.py`` is not edited: its hooks are imported from the file. Refuses to start before the lead's stamp.

Third and last exception to "unchanged settings" (lead order 2026-11-10 20:20 EST, A1): ``train`` applies the Test-S
KD mask — the teacher values of every chip whose pairing lies on an M2 Test-S site-date get weight 0 (308 training
pairings) through ``launch.activate_rerun_kd_mask``, which wraps the frozen KD loader in this process. A ``train``
re-run is one job of block ``reruns`` in the GPU-hour ledger (A7; ``--gpu-budget-h``, default 1.5, is an option of
this wrapper and is not passed on).

    python -B scripts/train/national2b_rerun.py train     --config configs/lite/<cfg>_national2.yaml --track national
    python -B scripts/train/national2b_rerun.py baselines --track national
    python -B scripts/train/national2b_rerun.py compare   --student <run> --baselines <run> --subset all
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ALLOWED = ("train", "baselines", "compare", "select")     # no "targets": a re-run computes no teacher target
RERUN_BUDGET_H = 1.5                                      # three seeds; the as-built runs took about 0.7 h


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rerun_name(config_path: str) -> str:
    """Run name of the re-run of ``config_path``: the configuration's own name with ``national2`` -> ``national2b``."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    name = str(cfg.get("name") or Path(config_path).stem)
    if not name.endswith("_national2"):
        raise SystemExit(f"{config_path}: not a national-2 configuration (name {name!r})")
    return name + "b"


def prepare_argv(argv: list[str]) -> list[str]:
    """Checked argument list of a re-run: target allowed, national track, no smoke, no own name, seeds as built."""
    if not argv or argv[0] not in ALLOWED:
        raise SystemExit(f"usage: national2b_rerun.py {{{','.join(ALLOWED)}}} [args]")
    out = list(argv)
    if out[0] in ("train", "baselines"):
        if "--track" not in out or out[out.index("--track") + 1] != "national":
            raise SystemExit("a re-run needs --track national")
    if out[0] == "train":
        for flag in ("--name", "--seeds", "--smoke"):
            if flag in out:
                raise SystemExit(f"a re-run takes no {flag}: name, seeds and length are fixed by the as-built run")
        out += ["--name", rerun_name(out[out.index("--config") + 1])]
    return out


def pop_budget(argv: list[str]) -> tuple[list[str], float]:
    """Take the wrapper's own ``--gpu-budget-h <hours>`` out of the argument list."""
    out = list(argv)
    if "--gpu-budget-h" not in out:
        return out, RERUN_BUDGET_H
    i = out.index("--gpu-budget-h")
    budget = float(out[i + 1])
    del out[i:i + 2]
    return out, budget


def activate(argv: list[str], n2run) -> None:
    """Everything that differs from the as-built run, in one place: the label table (and the reserved-site chips),
    the as-built hooks of ``national2_run.py``, and — for ``train`` — the Test-S KD mask."""
    from sentinel.models.litev2 import labels2b, launch

    labels2b.activate_rerun()
    n2run._ssl_init_hook()
    n2run._context_film_hook(argv)
    if argv[0] == "train":
        launch.activate_rerun_kd_mask()


def main(argv=None) -> int:
    argv, budget = pop_budget(list(sys.argv[1:] if argv is None else argv))
    argv = prepare_argv(argv)
    from sentinel.models.litev2 import launch, stamp

    stamp.require_stamp("a re-run on the national-2b labels")
    n2run = _load("scripts/train/national2_run.py", "national2_run")
    activate(argv, n2run)
    entry = n2run.TARGETS[argv[0]]
    mod = _load(entry, "lite_entry")
    sys.argv = [entry] + argv[1:]
    if argv[0] != "train":
        return mod.main(argv[1:])
    with launch.gpu_job("rerun_" + argv[argv.index("--name") + 1], "reruns", budget):
        return mod.main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
