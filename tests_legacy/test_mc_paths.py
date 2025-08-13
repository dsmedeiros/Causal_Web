import networkx as nx

from Causal_Web.engine.analysis.mc_paths import accumulate_path


def test_accumulate_path_aliases_legacy_fields() -> None:
    """accumulate_path should read legacy phase/attenuation fields."""

    g = nx.DiGraph()
    g.add_edge("s", "t", delay=1, phase_shift=0.25, attenuation=0.5)
    info = accumulate_path(g, ["s", "t"])
    assert info.delay == 1
    assert info.phase == 0.25
    assert info.attenuation == 0.5
