import ast
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1] / "Causal_Web" / "engine"


def test_engine_has_no_gui_imports():
    for path in ENGINE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("Causal_Web.gui"):
                        raise AssertionError(f"{path} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("Causal_Web.gui"):
                    raise AssertionError(f"{path} imports {module}")
