import pytest


def test_engine_adapter_exposed() -> None:
    from Causal_Web import EngineAdapter

    assert EngineAdapter is not None


def test_node_manager_removed() -> None:
    with pytest.raises(RuntimeError):
        from Causal_Web import NodeManager  # noqa: F401
