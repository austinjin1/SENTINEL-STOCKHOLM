# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""C4: every repo module reachable (AST imports, transitively, incl. function-local imports) from the D4 test path,
plus every package __init__ on the way, is in taprisk_registry.scoring_files()."""
import ast
from pathlib import Path

from sentinel.models.downstream import taprisk_registry as REG

ROOT = Path(__file__).resolve().parents[1]


def _module_file(mod: str):
    p = ROOT.joinpath(*mod.split("."))
    if (p / "__init__.py").is_file():
        return p / "__init__.py"
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py")
    return None


def _imports(path: Path, pkg: str):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = pkg.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
            yield base
            for a in node.names:
                yield f"{base}.{a.name}"


def closure(start: Path) -> set:
    seen, todo, files = set(), [(start, "")], set()
    while todo:
        path, pkg = todo.pop()
        for mod in _imports(path, pkg):
            if not mod.startswith("sentinel"):
                continue
            parts = mod.split(".")
            for i in range(1, len(parts) + 1):            # package __init__ files on the way
                m = ".".join(parts[:i])
                f = _module_file(m)
                if f is None or m in seen:
                    continue
                seen.add(m)
                files.add(str(f.relative_to(ROOT)))
                todo.append((f, m if f.name == "__init__.py" else ".".join(parts[: i - 1]) if i > 1 else m))
    return files


def test_import_closure_is_hashed():
    files = closure(ROOT / "scripts" / "train" / "downstream_taprisk_test.py")
    missing = sorted(files - set(REG.scoring_files()))
    assert not missing, f"unhashed modules in the D4 test-path import closure: {missing}"
