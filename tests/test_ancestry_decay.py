import numpy as np

from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet


def test_delta_m_decay_no_q_arrivals():
    adapter = EngineAdapter()
    graph = {
        "params": {"W0": 2},
        "nodes": [{"id": "0"}],
        "edges": [{"from": "0", "to": "0", "delay": 1.0}],
    }
    adapter.build_graph(graph)

    v_arr = adapter._arrays.vertices
    v_arr["m0"][0] = 1.0
    v_arr["m1"][0] = 0.0
    v_arr["m2"][0] = 0.0
    v_arr["m_norm"][0] = 1.0

    lccm = adapter._vertices[0]["lccm"]
    lccm.layer = "Θ"

    adapter._scheduler.push(0, 0, 0, Packet(src=0, dst=0))
    adapter.run_until_next_window_or(limit=10)

    m = np.array([v_arr["m0"][0], v_arr["m1"][0], v_arr["m2"][0]])
    assert np.allclose(m, np.array([1.0, 0.0, 0.0]))
    assert v_arr["m_norm"][0] < 1.0

    adapter._update_ancestry(0, 0, 0, 0, np.pi / 2, 1.0)

    m = np.array([v_arr["m0"][0], v_arr["m1"][0], v_arr["m2"][0]])
    assert m[1] > 0.0


def test_lambda_q_downweights_only_q_arrivals():
    adapter = EngineAdapter()
    adapter.build_graph({"nodes": [{"id": "0"}], "edges": []})

    v_arr = adapter._arrays.vertices
    v_arr["m0"][0] = 1.0
    v_arr["m1"][0] = 0.0
    v_arr["m2"][0] = 0.0
    v_arr["m_norm"][0] = 1.0

    lccm = adapter._vertices[0]["lccm"]
    lccm._lambda = 100
    lccm._lambda_q = 0

    adapter._update_ancestry(0, 0, 0, 0, np.pi / 2, 1.0)

    m = np.array([v_arr["m0"][0], v_arr["m1"][0], v_arr["m2"][0]])
    assert m[1] > 0.05


def test_batch_updates_all_q_packets():
    graph = {
        "nodes": [{"id": "0"}],
        "edges": [{"from": "0", "to": "0", "delay": 1.0}],
    }

    # Run with a single packet
    adapter_single = EngineAdapter()
    adapter_single.build_graph(graph)
    payload = {
        "psi": np.array([1.0], dtype=np.complex64),
        "p": np.array([1.0], dtype=np.float32),
    }
    pkt_single = Packet(src=0, dst=0, payload=payload)
    adapter_single._scheduler.push(0, 0, 0, pkt_single)
    adapter_single.run_until_next_window_or(limit=10)
    v_single = adapter_single._arrays.vertices
    hash_single = np.array(
        [v_single["h0"][0], v_single["h1"][0], v_single["h2"][0], v_single["h3"][0]],
        dtype=np.uint64,
    )

    # Run with two packets in the same batch
    adapter_batch = EngineAdapter()
    adapter_batch.build_graph(graph)
    pkt1 = Packet(src=0, dst=0, payload=payload)
    pkt2 = Packet(src=0, dst=0, payload=payload)
    adapter_batch._scheduler.push(0, 0, 0, pkt1)
    adapter_batch._scheduler.push(0, 0, 0, pkt2)
    adapter_batch.run_until_next_window_or(limit=10)
    v_batch = adapter_batch._arrays.vertices
    hash_batch = np.array(
        [v_batch["h0"][0], v_batch["h1"][0], v_batch["h2"][0], v_batch["h3"][0]],
        dtype=np.uint64,
    )

    assert not np.array_equal(hash_single, hash_batch)
