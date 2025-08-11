"""Graph loader for struct-of-arrays representation.

The :func:`load_graph_arrays` helper converts a graph JSON dictionary
into arrays suitable for the experimental engine. Missing fields are
filled from :mod:`Causal_Web.config.Config` defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from ...config import Config


@dataclass
class GraphArrays:
    """Struct-of-arrays container returned by :func:`load_graph_arrays`."""

    id_map: Dict[str, int]
    vertices: Dict[str, Any]
    edges: Dict[str, Any]
    adjacency: Dict[str, List[int]]


def _identity_matrix(dim: int) -> List[List[float]]:
    """Return a ``dim`` x ``dim`` identity matrix."""

    return [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]


def load_graph_arrays(graph_json: Dict[str, Any]) -> GraphArrays:
    """Convert ``graph_json`` into struct-of-arrays collections.

    Parameters
    ----------
    graph_json:
        Graph description as a dictionary following the JSON schema.

    Returns
    -------
    GraphArrays
        Struct-of-arrays representation of ``graph_json``.
    """

    nodes = graph_json.get("nodes", {})
    if isinstance(nodes, list):
        nodes = {n.get("id", str(i)): n for i, n in enumerate(nodes)}

    id_map = {nid: i for i, nid in enumerate(nodes)}
    n_vert = len(id_map)

    W0 = Config.windowing.get("W0", 0.0)
    Q = Config.windowing.get("Q", 0)
    Dq = int(Config.windowing.get("Dq", 1))
    Dp = int(Config.windowing.get("Dp", 1))

    psi_init = np.zeros((n_vert, Dq), dtype=np.complex64)
    psi_init[:, 0] = 1.0

    vertices = {
        "depth": np.zeros(n_vert, dtype=np.int32),
        "window_len": np.asarray(
            [nodes[nid].get("window_len", W0) for nid in nodes], dtype=np.float32
        ),
        "window_idx": np.zeros(n_vert, dtype=np.int32),
        "layer": np.asarray(
            [nodes[nid].get("layer", Q) for nid in nodes], dtype=np.int32
        ),
        "psi": psi_init,
        "psi_acc": np.zeros((n_vert, Dq), dtype=np.complex64),
        "EQ": np.zeros(n_vert, dtype=np.float32),
        "p": np.full((n_vert, Dp), 1.0 / Dp, dtype=np.float32),
        "bit": np.zeros(n_vert, dtype=np.int8),
        "conf": np.zeros(n_vert, dtype=np.float32),
        "fanin": np.zeros(n_vert, dtype=np.int32),
        "ancestry": np.zeros((n_vert, 4), dtype=np.uint64),
        "m": np.zeros((n_vert, 3), dtype=np.float32),
        "rho_mean": np.asarray(
            [nodes[nid].get("rho_mean", 0.0) for nid in nodes], dtype=np.float32
        ),
    }

    edges_data = graph_json.get("edges", [])
    n_edge = len(edges_data)

    edges = {
        "src": [],
        "dst": [],
        "d0": [],
        "rho": [],
        "alpha": [],
        "phi": [],
        "A": [],
        "U": [],
        "sigma": [],
    }

    default_u = _identity_matrix(Dq)
    unitary_map = getattr(Config, "unitaries", {})

    for edge in edges_data:
        src_idx = id_map.get(edge.get("from"))
        dst_idx = id_map.get(edge.get("to"))
        if src_idx is None or dst_idx is None:
            continue
        edges["src"].append(src_idx)
        edges["dst"].append(dst_idx)
        edges["d0"].append(edge.get("delay", 0.0))
        edges["rho"].append(edge.get("density", 0.0))
        edges["alpha"].append(edge.get("weight", 1.0))
        edges["phi"].append(edge.get("phase_shift", 0.0))
        edges["A"].append(edge.get("A_phase", 0.0))
        u_id = edge.get("u_id")
        if u_id is not None and isinstance(unitary_map, dict) and u_id in unitary_map:
            edges["U"].append(unitary_map[u_id])
        else:
            edges["U"].append([row[:] for row in default_u])
        edges["sigma"].append(0.0)

    # Build edge-neighbor adjacency CSR
    vertex_edges: Dict[int, List[int]] = {i: [] for i in range(n_vert)}
    for idx, (s, d) in enumerate(zip(edges["src"], edges["dst"])):
        vertex_edges[s].append(idx)
        vertex_edges[d].append(idx)

    nbr_ptr: List[int] = [0]
    nbr_idx: List[int] = []
    for idx in range(n_edge):
        s = edges["src"][idx]
        d = edges["dst"][idx]
        neighbors = set(vertex_edges[s] + vertex_edges[d])
        neighbors.discard(idx)
        nbr_idx.extend(sorted(neighbors))
        nbr_ptr.append(len(nbr_idx))

    adjacency = {
        "nbr_ptr": np.asarray(nbr_ptr, dtype=np.int32),
        "nbr_idx": np.asarray(nbr_idx, dtype=np.int32),
    }

    d0_arr = np.asarray(edges["d0"], dtype=np.float32)
    edges = {
        "src": np.asarray(edges["src"], dtype=np.int32),
        "dst": np.asarray(edges["dst"], dtype=np.int32),
        "d0": d0_arr,
        "rho": np.asarray(edges["rho"], dtype=np.float32),
        "alpha": np.asarray(edges["alpha"], dtype=np.float32),
        "phi": np.asarray(edges["phi"], dtype=np.float32),
        "A": np.asarray(edges["A"], dtype=np.float32),
        "U": np.asarray(edges["U"], dtype=np.complex64),
        "sigma": np.asarray(edges["sigma"], dtype=np.float32),
        "d_eff": np.maximum(1, np.round(d0_arr)).astype(np.int32),
    }

    return GraphArrays(
        id_map=id_map, vertices=vertices, edges=edges, adjacency=adjacency
    )
