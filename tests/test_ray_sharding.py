import networkx as nx

from Causal_Web.engine.backend.zone_partitioner import partition_zones
from Causal_Web.engine.backend import ray_cluster


def test_zone_partitioner_splits_graph():
    g = nx.Graph()
    for i in range(1000):
        g.add_node(f"N{i}", is_classical=(i % 2 == 0))
        if i > 0:
            g.add_edge(f"N{i-1}", f"N{i}")
    zones = partition_zones(g)
    assert len(zones) >= 2
    classical_nodes = sum(1 for i in range(1000) if i % 2 == 0)
    assert sum(len(z) for z in zones) == classical_nodes


def test_map_zones_parallel_order(monkeypatch):
    calls = []

    class FakeRay:
        def is_initialized(self):
            return True

        def remote(self, func):
            class Remote:
                def remote(self_inner, *args, **kwargs):
                    return func(*args, **kwargs)

            return Remote()

        def get(self, futures):
            # Reverse to emulate out-of-order completion
            return list(reversed(futures))

    fake = FakeRay()
    monkeypatch.setattr(ray_cluster, "ray", fake)
    ray_cluster._apply = fake.remote(lambda f, *a, **k: f(*a, **k))

    zones = [1, 2, 3]

    def work(z):
        calls.append(z)
        return z

    result = ray_cluster.map_zones(work, zones)
    assert result == [3, 2, 1]
    assert calls == zones


def test_map_zones_fallback_logs(caplog, monkeypatch):
    monkeypatch.setattr(ray_cluster, "ray", None)
    with caplog.at_level("INFO"):
        res = ray_cluster.map_zones(lambda z: z, [1, 2])
    assert "Ray cluster unavailable" in caplog.text
    assert res == [1, 2]
