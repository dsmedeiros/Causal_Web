"""Compatibility layer for the v2 engine prototype.

The :class:`EngineAdapter` exposes a subset of the legacy tick engine API but
drives a new depth-based scheduler and the lightweight :mod:`lccm` model.  The
adapter processes packets ordered by arrival depth and advances vertex windows
according to the local causal consistency math.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from collections import deque
import threading
import time
import random
import math

import numpy as np

from ..logging.logger import log_record
from .lccm import LCCM
from .scheduler import DepthScheduler
from .state import Packet, TelemetryFrame
from .loader import GraphArrays, load_graph_arrays
from .rho_delay import effective_delay, update_rho_delay, update_rho_delay_vec
from .qtheta_c import close_window, deliver_packet, deliver_packets_batch
from .epairs import EPairs
from .bell import BellHelpers, Ancestry
from ...config import Config


class EngineAdapter:
    """Bridge between the legacy orchestrator calls and engine v2."""

    def __init__(self) -> None:
        self._scheduler = DepthScheduler()
        self._running = False
        self._vertices: Dict[int, Dict[str, Any]] = {}
        self._arrays: GraphArrays | None = None
        self._edges_by_src: Dict[int, np.ndarray] = {}
        self._frame = 0
        self._rng = random.Random(Config.run_seed)
        eps_cfg = dict(Config.epsilon_pairs)
        eps_cfg.pop("decay_interval", None)
        eps_cfg.pop("decay_on_window_close", None)
        eps_cfg.pop("emit_per_delivery", None)
        eps_seed = eps_cfg.pop("seed", Config.run_seed)
        self._epairs = EPairs(seed=eps_seed, **eps_cfg)
        bell_seed = Config.bell.get("seed", Config.run_seed)
        self._bell = BellHelpers(seed=bell_seed)
        # Preallocated helpers to avoid per-event allocations
        self._eye_cache: Dict[int, np.ndarray] = {}
        self._psi_zero: Dict[int, np.ndarray] = {}
        self._p_zero: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    def _splitmix64(self, x: int) -> int:
        """Return a SplitMix64 hash of ``x``."""

        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = x
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        return int(z ^ (z >> 31))

    def _update_ancestry(
        self,
        dst: int,
        edge_id: int,
        depth_arr: int,
        seq: int,
        mu: float,
        kappa: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update ancestry hash and phase moment for ``dst``.

        The update uses only local information from the arriving packet and is
        deterministic given :data:`Config.run_seed`. The phase moment ``m`` is
        biased toward the destination's unitary via the supplied phase
        ``mu`` (``arg(<U ψ, ψ>)``) and coherence ``kappa`` and is
        down-weighted under heavy fan-in using ``β_m = β_m0 / (1 + Λ_v)`` where
        ``Λ_v`` counts quantum arrivals in the current window. The coefficient
        ``β_m0`` is sourced from :data:`Config.ancestry` so it can be adjusted via
        configuration.
        """

        if self._arrays is None:
            return np.zeros(4, dtype=np.uint64), np.zeros(3, dtype=float)

        v_arr = self._arrays.vertices

        # Current hash lanes and moment vector
        h0 = np.uint64(v_arr["h0"][dst])
        h1 = np.uint64(v_arr["h1"][dst])
        h2 = np.uint64(v_arr["h2"][dst])
        h3 = np.uint64(v_arr["h3"][dst])
        m = np.array(
            [v_arr["m0"][dst], v_arr["m1"][dst], v_arr["m2"][dst]],
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Local phase statistics
        u_local = np.array([np.cos(mu), np.sin(mu), kappa], dtype=np.float32)

        # Moment update with normalisation
        lambda_v = 0
        if dst in self._vertices:
            lambda_v = getattr(self._vertices[dst]["lccm"], "_lambda_q", 0)
        beta_m0 = Config.ancestry.get("beta_m0", 0.1)
        beta_m = beta_m0 / (1.0 + float(lambda_v))
        m = (1.0 - beta_m) * m + beta_m * u_local
        norm = float(np.linalg.norm(m))
        if norm > 0:
            m_norm = norm
            m /= norm
        else:
            m_norm = 0.0

        v_arr["m0"][dst] = m[0]
        v_arr["m1"][dst] = m[1]
        v_arr["m2"][dst] = m[2]
        v_arr["m_norm"][dst] = m_norm

        # --------------------------------------------------------------
        # Rolling hash update
        def f2u64(x: float) -> np.uint64:
            return np.frombuffer(np.float64(x).tobytes(), dtype=np.uint64)[0]

        h0 ^= np.uint64(dst) ^ (np.uint64(depth_arr) << np.uint64(1))
        h1 ^= np.uint64(np.int64(edge_id)) ^ (np.uint64(seq) << np.uint64(1))
        h2 ^= f2u64(mu)
        h3 ^= f2u64(kappa)
        h0 = np.uint64(self._splitmix64(int(h0)))
        h1 = np.uint64(self._splitmix64(int(h1)))
        h2 = np.uint64(self._splitmix64(int(h2)))
        h3 = np.uint64(self._splitmix64(int(h3)))

        v_arr["h0"][dst] = h0
        v_arr["h1"][dst] = h1
        v_arr["h2"][dst] = h2
        v_arr["h3"][dst] = h3

        ancestry = np.array([h0, h1, h2, h3], dtype=np.uint64)
        return ancestry, m

    # Public API -----------------------------------------------------
    def build_graph(self, graph_json_path: str | Dict[str, Any]) -> None:
        """Load a graph description and initialise per-vertex state.

        The loader exposes a CSR of incident edges which is used here to size
        windows and to compute mean incident densities for dynamic window
        growth.
        """

        if isinstance(graph_json_path, dict):
            graph = graph_json_path
        else:  # pragma: no cover - simple file IO
            import json

            with open(graph_json_path, "r", encoding="utf-8") as fh:
                graph = json.load(fh)

        params = graph.get("params", {})
        window_defaults = dict(Config.windowing)
        window_defaults.update(params)
        W0 = window_defaults.get("W0", 4)
        zeta1 = window_defaults.get("zeta1", 0.0)
        zeta2 = window_defaults.get("zeta2", 0.0)
        rho0 = params.get("rho0", 1.0)
        a = window_defaults.get("a", 1.0)
        b = window_defaults.get("b", 0.5)
        k_theta = window_defaults.get("k_theta", a)
        k_c = window_defaults.get("k_c", b)
        C_min = window_defaults.get("C_min", 0.0)
        f_min = params.get("f_min", 1.0)
        conf_min = params.get("conf_min", 0.0)
        H_max = params.get("H_max", 0.0)
        T_hold = window_defaults.get("T_hold", 1)
        T_class = params.get("T_class", 1)

        self._arrays = load_graph_arrays(graph)
        self._vertices.clear()
        edges = self._arrays.edges
        adj = self._arrays.adjacency
        n_vert = len(self._arrays.vertices["depth"])
        incident_ptr = adj.get("incident_ptr")
        incident_idx = adj.get("incident_idx")

        def incident_delay_cb(vid: int) -> Iterable[int]:
            start = int(incident_ptr[vid])
            end = int(incident_ptr[vid + 1])
            idxs = incident_idx[start:end]
            return (int(edges["d_eff"][i]) for i in idxs)

        self._epairs.set_incident_delays(incident_delay_cb)

        for vid in range(n_vert):
            out_idx = np.where(edges["src"] == vid)[0]
            self._edges_by_src[vid] = out_idx
            start = int(incident_ptr[vid])
            end = int(incident_ptr[vid + 1])
            idxs = incident_idx[start:end]
            deg = end - start
            rho_mean = float(edges["rho"][idxs].mean()) if deg > 0 else 0.0
            self._arrays.vertices["rho_mean"][vid] = rho_mean
            lccm = LCCM(
                W0,
                zeta1,
                zeta2,
                rho0,
                a,
                b,
                C_min,
                f_min,
                conf_min,
                H_max,
                T_hold,
                T_class,
                k_theta=k_theta,
                k_c=k_c,
                deg=deg,
                rho_mean=rho_mean,
            )
            vertex_state = {
                "lccm": lccm,
                "psi_acc": self._arrays.vertices["psi_acc"][vid],
                "p_v": self._arrays.vertices["p"][vid],
                "bit_deque": deque(maxlen=8),
                "base_deg": deg,
            }
            self._vertices[vid] = vertex_state

    def start(self) -> None:
        """Mark the engine as running."""

        self._running = True

    def pause(self) -> None:
        """Pause execution."""

        self._running = False

    def stop(self) -> None:
        """Stop execution and reset all state."""

        self._running = False
        self._vertices.clear()
        self._scheduler.clear()

    def step(self, max_events: int | None = None) -> TelemetryFrame:
        """Advance the simulation until a window rolls or ``max_events``."""

        return self.run_until_next_window_or(max_events)

    def run_until_next_window_or(self, limit: int | None) -> TelemetryFrame:
        """Run until the next window boundary or until ``limit`` events.

        Arrivals targeting the same destination and window are grouped so that
        field updates can be applied in batches for improved performance. The
        returned frame's ``depth`` mirrors the greatest arrival depth encountered
        during processing. If no events are handled or only events at the current
        depth are processed, the depth will remain unchanged.
        """

        if not self._running:
            self.start()

        if limit is None:
            limit = float("inf")

        start_windows = {
            vid: data["lccm"].window_idx for vid, data in self._vertices.items()
        }
        events = 0
        packets = []
        edge_logs = 0
        decay_counter = 0
        decay_interval = Config.epsilon_pairs.get("decay_interval", 32)

        while self._scheduler and events < limit:
            if decay_counter == 0:
                self._epairs.decay_all()
            decay_counter = (decay_counter + 1) % max(1, decay_interval)
            depth_arr, dst, edge_id, seq, pkt = self._scheduler.pop()
            vertex = self._vertices.get(dst)
            if vertex is None:
                continue
            lccm = vertex["lccm"]
            lccm.advance_depth(depth_arr)
            prev_layer = lccm.layer

            edges = self._arrays.edges if self._arrays else {}
            dim = len(vertex["psi_acc"])
            eye = self._eye_cache.setdefault(dim, np.eye(dim, dtype=np.complex64))
            psi_zero = self._psi_zero.setdefault(dim, np.zeros(dim, dtype=np.complex64))
            p_zero = self._p_zero.setdefault(
                len(vertex["p_v"]), np.zeros(len(vertex["p_v"]), dtype=np.float32)
            )

            packet_data = pkt.payload or {}
            if not isinstance(packet_data, dict):
                packet_data = {}
            depth_val = int(packet_data.get("depth_arr", depth_arr))
            psi_val = packet_data.get("psi", psi_zero)
            p_val = packet_data.get("p", p_zero)
            bit_val = int(packet_data.get("bit", 0))

            if edges and 0 <= edge_id < len(edges.get("alpha", [])):
                alpha_val = float(edges["alpha"][edge_id])
                phi_val = float(edges["phi"][edge_id])
                A_val = float(edges["A"][edge_id])
                U_val = edges["U"][edge_id]
            else:
                alpha_val = 1.0
                phi_val = 0.0
                A_val = 0.0
                U_val = eye

            psi_list = [psi_val]
            p_list = [p_val]
            bit_list = [bit_val]
            depth_list = [depth_val]
            alpha_list = [alpha_val]
            phi_list = [phi_val]
            A_list = [A_val]
            U_list = [U_val]
            pkt_list = [pkt]
            edge_id_list = [edge_id]
            seq_list = [seq]
            window_idx = lccm.window_idx
            requeue: list[tuple[int, int, int, Packet]] = []
            while self._scheduler and events + len(pkt_list) < limit:
                d2, dst2, edge2, seq2, pkt2 = self._scheduler.pop()
                if dst2 != dst:
                    requeue.append((d2, dst2, edge2, pkt2))
                    continue
                old_depth, old_window, old_lambda = (
                    lccm.depth,
                    lccm.window_idx,
                    lccm._lambda,
                )
                lccm.advance_depth(d2)
                if lccm.window_idx != window_idx:
                    lccm.depth, lccm.window_idx, lccm._lambda = (
                        old_depth,
                        old_window,
                        old_lambda,
                    )
                    requeue.append((d2, dst2, edge2, pkt2))
                    break
                pd2 = pkt2.payload or {}
                if not isinstance(pd2, dict):
                    pd2 = {}
                depth_list.append(int(pd2.get("depth_arr", d2)))
                psi_list.append(pd2.get("psi", psi_zero))
                p_list.append(pd2.get("p", p_zero))
                bit_list.append(int(pd2.get("bit", 0)))
                if edges and 0 <= edge2 < len(edges.get("alpha", [])):
                    alpha_list.append(float(edges["alpha"][edge2]))
                    phi_list.append(float(edges["phi"][edge2]))
                    A_list.append(float(edges["A"][edge2]))
                    U_list.append(edges["U"][edge2])
                else:
                    alpha_list.append(1.0)
                    phi_list.append(0.0)
                    A_list.append(0.0)
                    U_list.append(eye)
                pkt_list.append(pkt2)
                edge_id_list.append(edge2)
                seq_list.append(seq2)
            for item in requeue:
                self._scheduler.push(*item)

            mu_list: list[float] = []
            kappa_list: list[float] = []
            pkt_intensities: list[tuple[float, float, float]] = []
            for psi_i, U_i, phi_i, A_i, p_i, b_i in zip(
                psi_list, U_list, phi_list, A_list, p_list, bit_list
            ):
                psi_out_i = U_i @ psi_i
                psi_rot_i = np.exp(1j * (phi_i + A_i)) * psi_out_i
                z_i = np.vdot(psi_rot_i, psi_i)
                mu_list.append(float(np.angle(z_i)))
                kappa_list.append(float(abs(z_i)))
                q_int = min(1.0, float(np.linalg.norm(psi_rot_i) ** 2))
                theta_int = min(1.0, float(np.sum(np.abs(p_i))))
                c_int = float(b_i)
                pkt_intensities.append((q_int, theta_int, c_int))

            if len(pkt_list) > 1:
                (
                    depth_v,
                    psi_acc,
                    p_v,
                    (bit, conf),
                    intensities,
                ) = deliver_packets_batch(
                    lccm.depth,
                    vertex["psi_acc"],
                    vertex["p_v"],
                    vertex["bit_deque"],
                    psi_list,
                    p_list,
                    bit_list,
                    depth_list,
                    alpha_list,
                    phi_list,
                    A_list,
                    U_list,
                    max_deque=Config.max_deque,
                    update_p=lccm.layer == "Θ",
                )
            else:
                packet_struct = {
                    "psi": psi_list[0],
                    "p": p_list[0],
                    "bit": bit_list[0],
                    "depth_arr": depth_list[0],
                }
                edge_struct = {
                    "alpha": alpha_list[0],
                    "phi": phi_list[0],
                    "A": A_list[0],
                    "U": U_list[0],
                }
                (
                    depth_v,
                    psi_acc,
                    p_v,
                    (bit, conf),
                    intensities,
                    _,
                    _,
                ) = deliver_packet(
                    lccm.depth,
                    vertex["psi_acc"],
                    vertex["p_v"],
                    vertex["bit_deque"],
                    packet_struct,
                    edge_struct,
                    max_deque=Config.max_deque,
                    update_p=lccm.layer == "Θ",
                )

            vertex["psi_acc"][:] = psi_acc
            vertex["p_v"][:] = p_v
            if self._arrays is not None:
                self._arrays.vertices["psi_acc"][dst] = vertex["psi_acc"]
                self._arrays.vertices["p"][dst] = vertex["p_v"]
                self._arrays.vertices["bit"][dst] = bit
                self._arrays.vertices["conf"][dst] = conf
                self._arrays.vertices["depth"][dst] = depth_v

            bit_fraction = (
                sum(vertex["bit_deque"]) / len(vertex["bit_deque"])
                if vertex["bit_deque"]
                else 0.0
            )
            entropy = float(-(p_v * np.log2(p_v + 1e-12)).sum()) if len(p_v) else 0.0
            lccm.update_classical_metrics(bit_fraction, entropy, conf)
            is_q = lccm.layer == "Q"
            lccm.deliver(is_q)
            packets.extend(pkt_list)

            edge_intensity = {
                eid: pkt_intensities[idx] for idx, eid in enumerate(edge_id_list)
            }

            layer_idx_map = {"Q": 0, "Θ": 1, "C": 2}
            if pkt_intensities:
                if lccm.layer in layer_idx_map:
                    idx = layer_idx_map[lccm.layer]
                    values = [pi[idx] for pi in pkt_intensities]
                else:
                    values = [sum(pi) for pi in pkt_intensities]
                intensity = float(np.mean(values))
            else:
                intensity = 0.0

            edges_arr = self._arrays.edges if self._arrays else {}
            edge_ids_all = self._edges_by_src.get(dst, [])
            theta = 0.0
            if lccm.layer == "Q":
                ancestry_arr = np.zeros(4, dtype=np.uint64)
                m_arr = np.zeros(3, dtype=float)
                groups: Dict[int, list[tuple[int, int, float, float]]] = {}
                for e_id, seq_id, d_arr, mu_i, kappa_i in zip(
                    edge_id_list, seq_list, depth_list, mu_list, kappa_list
                ):
                    groups.setdefault(int(d_arr), []).append(
                        (e_id, seq_id, mu_i, kappa_i)
                    )
                depth_last = depth_arr
                for d_curr in sorted(groups):
                    if self._arrays is not None:
                        self._epairs.carry(dst, d_curr, edge_ids_all, edges_arr)
                    for e_id, seq_id, mu_i, kappa_i in groups[d_curr]:
                        ancestry_arr, m_arr = self._update_ancestry(
                            dst, e_id, d_curr, seq_id, mu_i, kappa_i
                        )
                        if (
                            Config.epsilon_pairs.get("emit_per_delivery", False)
                            and self._arrays is not None
                        ):
                            h_val_i = int(ancestry_arr[0])
                            theta_i = math.atan2(m_arr[1], m_arr[0])
                            self._epairs.emit(
                                dst, h_val_i, theta_i, d_curr, [e_id], edges_arr
                            )
                    depth_last = d_curr
                theta = math.atan2(m_arr[1], m_arr[0])
            else:
                if self._arrays is not None:
                    v_arr = self._arrays.vertices
                    ancestry_arr = np.array(
                        [
                            v_arr["h0"][dst],
                            v_arr["h1"][dst],
                            v_arr["h2"][dst],
                            v_arr["h3"][dst],
                        ],
                        dtype=np.uint64,
                    )
                    m_arr = np.array(
                        [v_arr["m0"][dst], v_arr["m1"][dst], v_arr["m2"][dst]],
                        dtype=np.float32,
                    )
                else:
                    ancestry_arr = np.zeros(4, dtype=np.uint64)
                    m_arr = np.zeros(3, dtype=float)

            bell_cfg = Config.bell
            if bell_cfg.get("enabled", False):
                mi_mode = (
                    "strict"
                    if bell_cfg.get("mi_mode", "MI_strict") == "MI_strict"
                    else "conditioned"
                )

                if "lambda_u" in packet_data:
                    detector_anc = Ancestry(ancestry_arr.copy(), m_arr.copy())
                    source_anc = Ancestry(
                        np.array(packet_data.get("ancestry", ancestry_arr)),
                        np.array(packet_data.get("m", m_arr)),
                    )
                    a_D = self._bell.setting_draw(
                        mi_mode,
                        detector_anc,
                        packet_data["lambda_u"],
                        bell_cfg.get("kappa_a", 0.0),
                    )
                    outcome, meta = self._bell.contextual_readout(
                        mi_mode,
                        a_D,
                        detector_anc,
                        packet_data["lambda_u"],
                        packet_data.get("zeta", 0),
                        bell_cfg.get("kappa_xi", 0.0),
                        source_anc,
                        bell_cfg.get("kappa_a", 0.0),
                        batch=self._frame,
                    )
                    log_record(
                        category="entangled",
                        label="measurement",
                        tick=self._frame,
                        value={
                            "setting": a_D.tolist(),
                            "outcome": int(outcome),
                            "mi_mode": mi_mode,
                            "kappa_a": bell_cfg.get("kappa_a", 0.0),
                            "kappa_xi": bell_cfg.get("kappa_xi", 0.0),
                            "batch_id": self._frame,
                            "h_prefix_len": Config.epsilon_pairs.get(
                                "ancestry_prefix_L", 0
                            ),
                        },
                        metadata={"L": meta.get("L")},
                    )
                    packet_data.setdefault("ancestry", ancestry_arr)
                    packet_data.setdefault("m", m_arr)
                else:
                    source_anc = Ancestry(ancestry_arr.copy(), m_arr.copy())
                    lam_u, zeta = self._bell.lambda_at_source(
                        source_anc,
                        bell_cfg.get("beta_m", 0.0),
                        bell_cfg.get("beta_h", 0.0),
                    )
                    packet_data["lambda_u"] = lam_u
                    packet_data["zeta"] = zeta
                    packet_data["ancestry"] = ancestry_arr
                    packet_data["m"] = m_arr

            edges = edges_arr
            if (
                lccm.layer == "Q"
                and self._arrays is not None
                and not Config.epsilon_pairs.get("emit_per_delivery", False)
            ):
                h_val = int(ancestry_arr[0])
                self._epairs.emit(dst, h_val, theta, depth_last, edge_ids_all, edges)

            if lccm.layer != prev_layer:
                if prev_layer == "Q" and lccm.layer == "Θ":
                    reason = "decoh_threshold"
                elif prev_layer == "Θ" and lccm.layer == "Q":
                    reason = "recoh_threshold"
                else:
                    reason = "layer_change"
                if self._arrays is not None:
                    p_v = vertex["p_v"]
                    H_pv = (
                        float(-(p_v * np.log2(p_v + 1e-12)).sum()) if p_v.size else 0.0
                    )
                    EQ = float(self._arrays.vertices["EQ"][dst])
                else:
                    H_pv = 0.0
                    EQ = lccm._eq
                log_record(
                    category="event",
                    label="layer_transition",
                    tick=self._frame,
                    value={
                        "v_id": dst,
                        "from_layer": prev_layer,
                        "to_layer": lccm.layer,
                        "reason": reason,
                        "window_idx": lccm.window_idx,
                        "Lambda_v": lccm._lambda,
                        "EQ": EQ,
                        "H_p": H_pv,
                    },
                )

            adj = self._arrays.adjacency if self._arrays else {}
            rho_cfg = Config.rho_delay
            mode = rho_cfg.get("inject_mode", "incoming")
            inject_edges: Iterable[int]
            if mode == "incoming":
                inject_edges = [
                    eid for eid in edge_id_list if 0 <= eid < len(edges.get("rho", []))
                ]
            elif mode == "incident" and adj:
                ptr = adj.get("incident_ptr")
                idx = adj.get("incident_idx")
                if ptr is not None and idx is not None:
                    start = ptr[dst]
                    end = ptr[dst + 1]
                    inject_edges = [int(i) for i in idx[start:end]]
                else:
                    inject_edges = []
            elif mode == "outgoing":
                inject_edges = list(self._edges_by_src.get(dst, []))
            else:
                inject_edges = [
                    eid for eid in edge_id_list if 0 <= eid < len(edges.get("rho", []))
                ]

            injected: Dict[int, Tuple[float, float, int]] = {}
            if inject_edges:
                ptr = adj.get("nbr_ptr")
                nbr = adj.get("nbr_idx")
                if ptr is not None and nbr is not None and len(ptr) > 1:
                    valid = [e for e in inject_edges if 0 <= e < len(ptr) - 1]
                    if valid:
                        inject_arr = np.asarray(valid, dtype=np.int32)
                        rho_before = edges["rho"][inject_arr]
                        neigh_sums_all = (
                            np.add.reduceat(edges["rho"][nbr], ptr[:-1])
                            if nbr.size > 0
                            else np.zeros(len(ptr) - 1, dtype=np.float32)
                        )
                        neigh_sums = neigh_sums_all[inject_arr]
                        counts = (ptr[1:] - ptr[:-1])[inject_arr]
                        mean = np.divide(
                            neigh_sums,
                            counts,
                            out=np.zeros_like(neigh_sums, dtype=np.float32),
                            where=counts > 0,
                        )
                        if mode == "incoming":
                            layer_idx = {"Q": 0, "Θ": 1, "C": 2}.get(lccm.layer, 0)
                            intensity_vec = np.array(
                                [
                                    edge_intensity.get(int(e), (0.0, 0.0, 0.0))[
                                        layer_idx
                                    ]
                                    for e in valid
                                ],
                                dtype=np.float32,
                            )
                        else:
                            intensity_vec = intensity
                        rho_after, d_eff = update_rho_delay_vec(
                            rho_before,
                            mean,
                            intensity_vec,
                            alpha_d=rho_cfg.get("alpha_d", 0.0),
                            alpha_leak=rho_cfg.get("alpha_leak", 0.0),
                            eta=rho_cfg.get("eta", 0.0),
                            d0=edges["d0"][inject_arr],
                            gamma=rho_cfg.get("gamma", 0.0),
                            rho0=rho_cfg.get("rho0", 1.0),
                        )
                        edges["rho"][inject_arr] = rho_after
                        if "d_eff" in edges:
                            edges["d_eff"][inject_arr] = d_eff
                        for idx, rb, ra, de in zip(
                            inject_arr, rho_before, rho_after, d_eff
                        ):
                            injected[int(idx)] = (float(rb), float(ra), int(de))
                else:
                    layer_idx = {"Q": 0, "Θ": 1, "C": 2}.get(lccm.layer, 0)
                    for edge_idx in inject_edges:
                        if 0 <= edge_idx < len(edges.get("rho", [])):
                            start = (
                                ptr[edge_idx]
                                if ptr is not None and edge_idx < len(ptr) - 1
                                else 0
                            )
                            end = (
                                ptr[edge_idx + 1]
                                if ptr is not None and edge_idx < len(ptr) - 1
                                else 0
                            )
                            neighbours = (
                                edges["rho"][nbr[start:end]]
                                if nbr is not None and end > start
                                else []
                            )
                            intensity_val = intensity
                            if mode == "incoming":
                                intensity_val = edge_intensity.get(
                                    int(edge_idx), (0.0, 0.0, 0.0)
                                )[layer_idx]
                            rho_before = float(edges["rho"][edge_idx])
                            rho_after, d_eff = update_rho_delay(
                                rho_before,
                                neighbours,
                                intensity_val,
                                alpha_d=rho_cfg.get("alpha_d", 0.0),
                                alpha_leak=rho_cfg.get("alpha_leak", 0.0),
                                eta=rho_cfg.get("eta", 0.0),
                                d0=float(edges["d0"][edge_idx]),
                                gamma=rho_cfg.get("gamma", 0.0),
                                rho0=rho_cfg.get("rho0", 1.0),
                            )
                            edges["rho"][edge_idx] = rho_after
                            if "d_eff" in edges:
                                edges["d_eff"][edge_idx] = d_eff
                            injected[int(edge_idx)] = (
                                float(rho_before),
                                float(rho_after),
                                int(d_eff),
                            )

            log_rho_edges = self._rng.random() < Config.logging.get(
                "sample_rho_rate", 0.0
            )
            for edge_idx in self._edges_by_src.get(dst, []):
                rho_before = float(edges["rho"][edge_idx])
                if edge_idx in injected:
                    rho_after, d_eff = injected[edge_idx][1], injected[edge_idx][2]
                else:
                    rho_after = rho_before
                    d_eff = int(
                        effective_delay(
                            rho_before,
                            d0=float(edges["d0"][edge_idx]),
                            gamma=rho_cfg.get("gamma", 0.0),
                            rho0=rho_cfg.get("rho0", 1.0),
                        )
                    )
                    if "d_eff" in edges:
                        edges["d_eff"][edge_idx] = d_eff
                depth_next = depth_arr + d_eff
                payload = {
                    "psi": self._arrays.vertices["psi"][dst],
                    "p": self._arrays.vertices["p"][dst],
                    "bit": int(self._arrays.vertices["bit"][dst]),
                    "lambda_u": packet_data.get("lambda_u"),
                    "zeta": packet_data.get("zeta"),
                    "ancestry": packet_data.get("ancestry"),
                    "m": packet_data.get("m"),
                }
                new_pkt = Packet(
                    src=dst, dst=int(edges["dst"][edge_idx]), payload=payload
                )
                edge_logs += 1
                if log_rho_edges:
                    log_record(
                        category="event",
                        label="edge_delivery",
                        tick=self._frame,
                        value={
                            "rho_before": rho_before,
                            "rho_after": rho_after,
                            "d_eff": d_eff,
                            "leak_contrib": None,
                            "is_bridge": False,
                            "sigma": float(edges["sigma"][edge_idx]),
                        },
                    )
                self._scheduler.push(
                    depth_next, int(edges["dst"][edge_idx]), edge_idx, new_pkt
                )
                self._epairs.reinforce(dst, int(edges["dst"][edge_idx]))

            for other in self._epairs.partners(dst):
                bridge = self._epairs.bridges[self._epairs._bridge_key(dst, other)]
                depth_next = depth_arr + bridge.d_bridge
                payload = {
                    "psi": self._arrays.vertices["psi"][dst],
                    "p": self._arrays.vertices["p"][dst],
                    "bit": int(self._arrays.vertices["bit"][dst]),
                    "lambda_u": packet_data.get("lambda_u"),
                    "zeta": packet_data.get("zeta"),
                    "ancestry": packet_data.get("ancestry"),
                    "m": packet_data.get("m"),
                }
                edge_logs += 1
                rate = Config.logging.get("sample_rho_rate", 0.0)
                if rate > 0.0 and self._rng.random() < rate:
                    log_record(
                        category="event",
                        label="edge_delivery",
                        tick=self._frame,
                        value={
                            "rho_before": 0.0,
                            "rho_after": 0.0,
                            "d_eff": bridge.d_bridge,
                            "leak_contrib": None,
                            "is_bridge": True,
                            "sigma": bridge.sigma,
                            "edge_id": bridge.edge_id,
                        },
                    )
                self._scheduler.push(
                    depth_next,
                    other,
                    bridge.edge_id,
                    Packet(src=dst, dst=other, payload=payload),
                )
                self._epairs.reinforce(dst, other)
            events += len(pkt_list)

            if any(
                data["lccm"].window_idx != start_windows[vid]
                for vid, data in self._vertices.items()
            ):
                break

        max_depth = 0
        max_window = 0
        for data in self._vertices.values():
            lccm = data["lccm"]
            max_depth = max(max_depth, lccm.depth)
            max_window = max(max_window, lccm.window_idx)

        frame = TelemetryFrame(
            depth=max_depth, events=events, packets=packets, window=max_window
        )

        depth_bucket = max_depth // 10
        self._frame += 1
        log_record(
            category="tick",
            label="adapter_frame",
            tick=self._frame,
            value={"depth": max_depth, "events": events},
            metadata={
                "frame": self._frame,
                "depth_bucket": depth_bucket,
                "window_idx": max_window,
            },
        )

        log_record(
            category="tick",
            label="edge_window_summary",
            tick=self._frame,
            value={"count": edge_logs},
            metadata={"window_idx": max_window},
        )

        for vid, data in self._vertices.items():
            lccm = data["lccm"]
            if lccm.window_idx != start_windows.get(vid, lccm.window_idx):
                if self._arrays is not None:
                    psi_acc = data["psi_acc"]
                    v_arr = self._arrays.vertices
                    psi, EQ = close_window(psi_acc)
                    v_arr["psi"][vid] = psi
                    psi_acc.fill(0)
                    v_arr["EQ"][vid] = EQ
                    p_v = data["p_v"]
                    H_pv = (
                        float(-(p_v * np.log2(p_v + 1e-12)).sum()) if p_v.size else 0.0
                    )
                    conf = float(v_arr["conf"][vid])
                    E_theta = lccm.k_theta * (1.0 - H_pv)
                    E_C = lccm.k_c * conf
                    if getattr(lccm, "_lambda_q_prev", 0) == 0:
                        m = np.array(
                            [v_arr["m0"][vid], v_arr["m1"][vid], v_arr["m2"][vid]],
                            dtype=np.float32,
                        )
                        delta = Config.ancestry.get("delta_m", 0.02)
                        m *= 1.0 - delta
                        norm = float(np.linalg.norm(m))
                        m /= max(norm, 1e-12)
                        v_arr["m0"][vid] = m[0]
                        v_arr["m1"][vid] = m[1]
                        v_arr["m2"][vid] = m[2]
                        v_arr["m_norm"][vid] = norm
                    match Config.theta_reset:
                        case "uniform":
                            p_v.fill(1.0 / len(p_v))
                        case "renorm":
                            total = float(p_v.sum())
                            if total > 0.0:
                                p_v /= total
                        case "hold":
                            pass
                        case _:
                            p_v.fill(1.0 / len(p_v))
                    self._arrays.vertices["p"][vid] = p_v
                    self._arrays.vertices["bit"][vid] = 0
                    self._arrays.vertices["conf"][vid] = 0.0
                    v_arr["E_theta"][vid] = E_theta
                    v_arr["E_C"][vid] = E_C
                    data["bit_deque"].clear()
                    lccm.update_eq(EQ)
                    edges_arr = self._arrays.edges
                    adj = self._arrays.adjacency
                    start = int(adj["incident_ptr"][vid])
                    end = int(adj["incident_ptr"][vid + 1])
                    if end > start:
                        idxs = adj["incident_idx"][start:end]
                        rho_mean = float(edges_arr["rho"][idxs].mean())
                    else:
                        rho_mean = 0.0
                    self._arrays.vertices["rho_mean"][vid] = rho_mean
                    lccm.rho_mean = rho_mean
                    base_deg = data.get("base_deg", lccm.deg)
                    lccm.deg = base_deg + len(self._epairs.partners(vid))
                else:
                    EQ = lccm._eq
                    E_theta = 0.0
                    E_C = 0.0
                log_record(
                    category="event",
                    label="vertex_window_close",
                    tick=self._frame,
                    value={
                        "layer": lccm.layer,
                        "Lambda_v": lccm._lambda,
                        "EQ": EQ,
                        "H_p": H_pv,
                        "E_theta": E_theta,
                        "E_C": E_C,
                        "v_id": vid,
                    },
                    metadata={"window_idx": lccm.window_idx},
                )
                if Config.epsilon_pairs.get("decay_on_window_close", True):
                    # Decay bridges when vertices advance a window.
                    self._epairs.decay_all()

        return frame

    def snapshot_for_ui(self) -> dict:
        """Return a minimal snapshot for the GUI."""

        max_depth = 0
        max_window = 0
        for data in self._vertices.values():
            lccm = data["lccm"]
            max_depth = max(max_depth, lccm.depth)
            max_window = max(max_window, lccm.window_idx)
        return {"depth": max_depth, "window": max_window}

    def current_depth(self) -> int:
        """Return the current depth of the simulation."""

        max_depth = 0
        for data in self._vertices.values():
            max_depth = max(max_depth, data["lccm"].depth)
        return max_depth

    def current_frame(self) -> int:
        """Return the number of steps executed so far."""

        return self._frame


_ENGINE = EngineAdapter()


def build_graph(graph_json_path: str | Dict[str, Any] | None = None) -> None:
    """Build the simulation graph for headless runs."""

    from ...config import Config

    path = graph_json_path or Config.graph_file
    _ENGINE.build_graph(path)


def simulation_loop() -> None:
    """Start a background loop advancing the engine while running."""

    from ...config import Config

    def _run() -> None:
        while True:
            with Config.state_lock:
                if not Config.is_running or (
                    Config.tick_limit and Config.current_tick >= Config.tick_limit
                ):
                    Config.is_running = False
                    break
            _ENGINE.step()
            with Config.state_lock:
                Config.current_tick += 1
            time.sleep(0)

    threading.Thread(target=_run, daemon=True).start()


def pause_simulation() -> None:
    """Pause execution of the simulation."""

    from ...config import Config

    _ENGINE.pause()
    with Config.state_lock:
        Config.is_running = False


def resume_simulation() -> None:
    """Resume a previously paused simulation."""

    from ...config import Config

    _ENGINE.start()
    with Config.state_lock:
        Config.is_running = True
    simulation_loop()


def stop_simulation() -> None:
    """Stop the simulation and reset state."""

    from ...config import Config

    _ENGINE.stop()
    with Config.state_lock:
        Config.is_running = False


def get_snapshot() -> dict:
    """Return a minimal snapshot for the UI."""

    return _ENGINE.snapshot_for_ui()


__all__ = [
    "EngineAdapter",
    "build_graph",
    "simulation_loop",
    "pause_simulation",
    "resume_simulation",
    "stop_simulation",
    "get_snapshot",
]
