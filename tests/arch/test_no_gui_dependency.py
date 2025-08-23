def test_engine_has_no_gui_imports():
    import importlib
    import sys

    sys.modules.pop("ui_new", None)
    import Causal_Web  # noqa: F401
