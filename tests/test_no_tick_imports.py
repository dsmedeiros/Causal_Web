import ast
from pathlib import Path

FORBIDDEN = {"legacy.engine.tick_engine", "legacy.engine.models.tick"}

# The legacy tick engine should remain isolated within legacy modules.


def test_no_tick_engine_import_outside_legacy():
    """Ensure legacy tick modules are not imported outside legacy folders."""
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*.py"):
        if any("legacy" in part for part in path.parts):
            continue
        with path.open("r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read(), filename=str(path))
            except SyntaxError:
                continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in FORBIDDEN:
                        raise AssertionError(f"{path} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level > 0:
                    if module.endswith("tick") or any(
                        alias.name == "tick" for alias in node.names
                    ):
                        raise AssertionError(f"{path} uses relative import of tick")
                if module in FORBIDDEN:
                    raise AssertionError(f"{path} imports {module}")
                for alias in node.names:
                    full = f"{module}.{alias.name}" if module else alias.name
                    if full in FORBIDDEN:
                        raise AssertionError(f"{path} imports {full}")
