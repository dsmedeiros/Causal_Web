import time
import networkx as nx
import pytest

from Causal_Web.engine.backend import ray_cluster
from Causal_Web.engine.backend.zone_partitioner import partition_zones


@pytest.mark.skipif(ray_cluster.ray is None, reason="ray not available")
def test_ray_zone_parallel_performance():
    """Benchmark Ray sharding against sequential CPU processing for 20 zones."""
    ray_cluster.init_cluster(num_cpus=2, ignore_reinit_error=True)
    g = nx.Graph()
    # Build alternating classical/quantum chain producing 20 isolated classical zones
    for i in range(40):
        g.add_node(f"N{i}", is_classical=(i % 2 == 0))
        if i > 0:
            g.add_edge(f"N{i-1}", f"N{i}")

    zones = partition_zones(g)
    assert len(zones) == 20

    def work(_):
        total = 0
        for i in range(1000):
            total += i * i
        return total

    # Warm up
    [work(z) for z in zones]
    ray_cluster.map_zones(lambda z: z, [])

    start = time.perf_counter()
    [work(z) for z in zones]
    cpu_time = time.perf_counter() - start

    start = time.perf_counter()
    ray_cluster.map_zones(work, zones)
    ray_time = time.perf_counter() - start

    # Allow generous margin to account for Ray overhead
    assert ray_time < cpu_time * 4
