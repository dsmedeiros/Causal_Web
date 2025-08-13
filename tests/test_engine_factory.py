from importlib import reload

from Causal_Web.engine.engine_v2 import adapter


def test_get_engine_singleton():
    adapter_module = reload(adapter)
    assert adapter_module._ENGINE is None
    eng1 = adapter_module.get_engine()
    eng2 = adapter_module.get_engine()
    assert eng1 is eng2


def test_get_engine_multi_run(monkeypatch):
    adapter_module = reload(adapter)
    eng1 = adapter_module.get_engine()
    eng1.stop()
    monkeypatch.setattr(adapter_module, "_ENGINE", None, raising=False)
    eng2 = adapter_module.get_engine()
    assert eng1 is not eng2
