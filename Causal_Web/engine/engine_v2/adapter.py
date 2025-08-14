"""Compatibility layer for the v2 engine prototype.

The :class:`EngineAdapter` exposes a subset of the legacy tick engine API but
drives a new depth-based scheduler and the lightweight :mod:`lccm` model.  The
adapter processes packets ordered by arrival depth and advances vertex windows
according to the local causal consistency math. The module has been refactored
to remove outdated hooks from the legacy engine.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set
from collections import deque, defaultdict
import threading
import time
import random
import math

import numpy as np

from ..logging.logger import log_record, flush_metrics
from .lccm import LCCM, WindowParams, WindowState, on_window_close
from .scheduler import DepthScheduler
from .state import Packet, TelemetryFrame
from .loader import GraphArrays, load_graph_arrays
from .rho_delay import effective_delay, update_rho_delay, update_rho_delay_vec
from .rho.variational import lambda_to_coeffs
from .qtheta_c import (
    close_window,
    deliver_packet,
    deliver_packets_batch,
    phase_stats_batch,
)
from .epairs import EPairs
from .bell import BellHelpers, Ancestry
from ...config import Config, RunConfig


class EngineAdapter:
    """Bridge between the legacy orchestrator calls and engine v2."""

    def __init__(self) -> None:
        self._scheduler = DepthScheduler()
        self._running = False
        self._vertices: Dict[int, Dict[str, Any]] = {}
        self._arrays: GraphArrays | None = None
        self._edges_by_src: Dict[int, np.ndarray] = {}
        self._frame = 0
        self._cfg: RunConfig | None = None
        self._rng: random.Random | None = None
        self._epairs: EPairs | None = None
        self._bell: BellHelpers | None = None
        # Preallocated helpers to avoid per-event allocations
        self._eye_cache: Dict[int, np.ndarray] = {}
        self._psi_zero: Dict[int, np.ndarray] = {}
        self._p_zero: Dict[int, np.ndarray] = {}
        self._packet_buf: Dict[str, Any] = {}
        self._edge_buf: Dict[str, Any] = {}
        self._payload_buf: Dict[str, Any] = {}
        self._neigh_sums_cache: np.ndarray | None = None
        self._delay_changed: Set[int] = set()
        self._lock = threading.RLock()

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
        deterministic given :data:`self._cfg.run_seed`. The phase moment ``m`` is
        biased toward the destination's unitary via the supplied phase
        ``mu`` (``arg(<U ψ, ψ>)``) and coherence ``kappa`` and is
        down-weighted under heavy fan-in using ``β_m = β_m0 / (1 + Λ_v)`` where
        ``Λ_v`` counts quantum arrivals in the current window. The coefficient
        ``β_m0`` is sourced from :data:`self._cfg.ancestry` so it can be adjusted via
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
        beta_m0 = self._cfg.ancestry.get("beta_m0", 0.0)
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

        # Freeze configuration for this run
        self._cfg = Config.snapshot()
        self._rng = random.Random(self._cfg.run_seed)
        eps_cfg = dict(self._cfg.epsilon_pairs)
        eps_cfg.pop("decay_interval", None)
        eps_cfg.pop("decay_on_window_close", None)
        eps_cfg.pop("emit_per_delivery", None)
        eps_seed = eps_cfg.pop("seed", self._cfg.run_seed)
        self._epairs = EPairs(
            seed=eps_seed,
            sample_seed_rate=self._cfg.logging.get("sample_seed_rate", 1.0),
            sample_bridge_rate=self._cfg.logging.get("sample_bridge_rate", 1.0),
            **eps_cfg,
        )
        bell_seed = self._cfg.bell.get("seed", self._cfg.run_seed)
        self._bell = BellHelpers(self._cfg.bell, seed=bell_seed)

        params = graph.get("params", {})
        window_defaults = dict(self._cfg.windowing)
        window_defaults.update(params)
        lccm_cfg = self._cfg.lccm
        mode = lccm_cfg.get("mode", "thresholds")
        fe_cfg = lccm_cfg.get("free_energy", {})
        W0 = window_defaults.get("W0", 4)
        zeta1 = window_defaults.get("zeta1", 0.0)
        zeta2 = window_defaults.get("zeta2", 0.0)
        rho0 = params.get("rho0", 1.0)
        a = window_defaults.get("a", 1.0)
        b = float(window_defaults.get("b", 0.0))
        if mode == "free_energy":
            k_theta = fe_cfg.get("k_theta", window_defaults.get("k_theta", a))
            k_c = fe_cfg.get("k_c", window_defaults.get("k_c", b))
            k_q = fe_cfg.get("k_q", 0.0)
            F_min = fe_cfg.get("F_min", 0.0)
        else:
            k_theta = window_defaults.get("k_theta", a)
            k_c = window_defaults.get("k_c", b)
            k_q = 0.0
            F_min = 0.0
        C_min = window_defaults.get("C_min", 0.0)
        f_min = params.get("f_min", 1.0)
        conf_min = params.get("conf_min", 0.0)
        H_max = params.get("H_max", 0.0)
        T_hold = window_defaults.get("T_hold", 1)
        T_class = params.get("T_class", 1)

        self._arrays = load_graph_arrays(
            graph, self._cfg.windowing, getattr(self._cfg, "unitaries", {})
        )
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
                mode=mode,
                k_q=k_q,
                F_min=F_min,
                deg=deg,
                rho_mean=rho_mean,
            )
            w_init = lccm._window_size()
            vertex_state = {
                "lccm": lccm,
                "psi_acc": self._arrays.vertices["psi_acc"][vid],
                "p_v": self._arrays.vertices["p"][vid],
                "bit_deque": deque(maxlen=8),
                "base_deg": deg,
                "win_state": WindowState(M_v=rho_mean, W_v=w_init),
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
        """Advance the simulation until a window rolls or ``max_events``.

        This call is thread-safe so the UI can request snapshots while the
        simulation advances in a background thread.
        """

        with self._lock:
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
        rho_cfg = self._cfg.rho_delay
        rho_mode = self._cfg.rho.get("update_mode", "heuristic")
        lambda_cfg = self._cfg.rho.get("variational", {})
        alpha_d = rho_cfg.get("alpha_d", 0.0)
        alpha_leak = rho_cfg.get("alpha_leak", 0.0)
        eta = rho_cfg.get("eta", 0.0)
        if rho_mode == "variational":
            alpha_d, alpha_leak, eta = lambda_to_coeffs(
                lambda_cfg.get("lambda_s", 0.0),
                lambda_cfg.get("lambda_l", 0.0),
                lambda_cfg.get("lambda_I", 0.0),
                eta,
            )
        gamma = rho_cfg.get("gamma", 0.0)
        rho0 = rho_cfg.get("rho0", 1.0)
        inject_mode = rho_cfg.get("inject_mode", "incoming")

        bell_cfg = self._cfg.bell
        bell_enabled = bell_cfg.get("enabled", False)
        mi_mode_default = (
            "strict"
            if bell_cfg.get("mi_mode", "MI_strict") == "MI_strict"
            else "conditioned"
        )
        kappa_a = bell_cfg.get("kappa_a", 0.0)
        kappa_xi = bell_cfg.get("kappa_xi", 0.0)
        beta_m = bell_cfg.get("beta_m", 0.0)
        beta_h = bell_cfg.get("beta_h", 0.0)

        layer_idx_map = {"Q": 0, "Θ": 1, "C": 2}

        start_windows = {
            vid: data["lccm"].window_idx for vid, data in self._vertices.items()
        }
        events = 0
        packets = []
        edge_logs = 0
        decay_counter = 0
        decay_interval = self._cfg.epsilon_pairs.get("decay_interval", 32)
        packet_struct = self._packet_buf
        edge_struct = self._edge_buf
        payload_buf = self._payload_buf

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
                phase_val = edges.get("phase", [1.0 + 0.0j])[edge_id]
                U_val = edges["U"][edge_id]
            else:
                alpha_val = 1.0
                phase_val = 1.0 + 0.0j
                U_val = eye

            psi_list = [psi_val]
            p_list = [p_val]
            bit_list = [bit_val]
            depth_list = [depth_val]
            alpha_list = [alpha_val]
            phase_list = [phase_val]
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
                    phase_list.append(edges.get("phase", [1.0 + 0.0j])[edge2])
                    U_list.append(edges["U"][edge2])
                else:
                    alpha_list.append(1.0)
                    phase_list.append(1.0 + 0.0j)
                    U_list.append(eye)
                pkt_list.append(pkt2)
                edge_id_list.append(edge2)
                seq_list.append(seq2)
            for item in requeue:
                self._scheduler.push(*item)

            U_arr = np.asarray(U_list, dtype=np.complex64)
            phase_arr = np.asarray(phase_list, dtype=np.complex64)
            psi_arr = np.asarray(psi_list, dtype=np.complex64)
            mu_arr, kappa_arr, psi_rot_arr = phase_stats_batch(
                U_arr, phase_arr, psi_arr
            )
            mu_list = mu_arr.tolist()
            kappa_list = kappa_arr.tolist()

            p_arr = np.asarray(p_list, dtype=np.float32)
            theta_vals = np.sum(np.abs(p_arr), axis=1)
            theta_int_arr = np.minimum(1.0, theta_vals)
            theta_mean = float(np.mean(theta_vals)) if len(theta_vals) else 0.0
            theta_mean = min(1.0, theta_mean)

            q_int_arr = np.minimum(1.0, np.linalg.norm(psi_rot_arr, axis=1) ** 2)
            c_int_arr = np.asarray(bit_list, dtype=float)
            pkt_intensities = list(
                zip(q_int_arr.tolist(), theta_int_arr.tolist(), c_int_arr.tolist())
            )

            if len(pkt_list) > 1:
                (
                    depth_v,
                    psi_acc,
                    p_v,
                    (bit, conf),
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
                    phase_list,
                    U_list,
                    max_deque=self._cfg.max_deque,
                    update_p=lccm.layer == "Θ",
                )
            else:
                packet_struct["psi"] = psi_list[0]
                packet_struct["p"] = p_list[0]
                packet_struct["bit"] = bit_list[0]
                packet_struct["depth_arr"] = depth_list[0]
                edge_struct["alpha"] = alpha_list[0]
                edge_struct["phase"] = phase_list[0]
                edge_struct["U"] = U_list[0]
                (
                    depth_v,
                    psi_acc,
                    p_v,
                    (bit, conf),
                    _,
                    _,
                    _,
                ) = deliver_packet(
                    lccm.depth,
                    vertex["psi_acc"],
                    vertex["p_v"],
                    vertex["bit_deque"],
                    packet_struct,
                    edge_struct,
                    max_deque=self._cfg.max_deque,
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
            p_v = np.where(np.isfinite(p_v), p_v, np.zeros_like(p_v))
            entropy = float(-(p_v * np.log2(p_v + 1e-12)).sum()) if len(p_v) else 0.0
            lccm.update_classical_metrics(bit_fraction, entropy, conf)
            is_q = lccm.layer == "Q"
            lccm.deliver(is_q)
            packets.extend(pkt_list)

            edge_intensity = defaultdict(lambda: (0.0, 0.0, 0.0))
            for idx, eid in enumerate(edge_id_list):
                edge_intensity[int(eid)] = pkt_intensities[idx]

            if inject_mode != "incoming" and lccm.layer == "Θ":
                intensity = theta_mean
            elif pkt_intensities:
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
                            self._cfg.epsilon_pairs.get("emit_per_delivery", False)
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

            if bell_enabled:
                mi_mode = mi_mode_default

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
                        kappa_a,
                    )
                    outcome, meta = self._bell.contextual_readout(
                        mi_mode,
                        a_D,
                        detector_anc,
                        packet_data["lambda_u"],
                        packet_data.get("zeta", 0),
                        kappa_xi,
                        source_anc,
                        kappa_a,
                        batch=self._frame,
                    )
                    log_record(
                        category="entangled",
                        label="measurement",
                        frame=self._frame,
                        tick=self._frame,
                        value={
                            "setting": a_D.tolist(),
                            "outcome": int(outcome),
                            "mi_mode": mi_mode,
                            "kappa_a": kappa_a,
                            "kappa_xi": kappa_xi,
                            "batch_id": self._frame,
                            "h_prefix_len": self._cfg.epsilon_pairs.get(
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
                        beta_m,
                        beta_h,
                    )
                    packet_data["lambda_u"] = lam_u
                    packet_data["zeta"] = zeta
                    packet_data["ancestry"] = ancestry_arr
                    packet_data["m"] = m_arr

            edges = edges_arr
            if (
                lccm.layer == "Q"
                and self._arrays is not None
                and not self._cfg.epsilon_pairs.get("emit_per_delivery", False)
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
                        frame=self._frame,
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
                if prev_layer == "C" and lccm.layer != "C":
                    if self._arrays is not None:
                        self._arrays.vertices["bit"][dst] = 0
                        self._arrays.vertices["conf"][dst] = 0.0
                    vertex["bit_deque"].clear()

            adj = self._arrays.adjacency if self._arrays else {}
            mode = inject_mode
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
                rho_arr = edges["rho"]
                # Use cached neighbour sums for vectorised ρ updates unless
                # explicitly disabled for Gauss–Seidel semantics. The cache is
                # incrementally maintained for touched edges to avoid full
                # recomputation when only a few buckets change.
                if (
                    ptr is not None
                    and nbr is not None
                    and len(ptr) > 1
                    and self._cfg.rho_delay.get("vectorized", True)
                ):
                    if (
                        self._neigh_sums_cache is None
                        or self._neigh_sums_cache.size != len(ptr) - 1
                    ):
                        if nbr.size > 0:
                            self._neigh_sums_cache = np.add.reduceat(
                                rho_arr[nbr], ptr[:-1]
                            )
                        else:
                            self._neigh_sums_cache = np.zeros(
                                len(ptr) - 1, dtype=np.float32
                            )
                    neigh_sums_all = self._neigh_sums_cache
                    valid = [e for e in inject_edges if 0 <= e < len(ptr) - 1]
                    if valid and neigh_sums_all is not None:
                        inject_arr = np.asarray(valid, dtype=np.int32)
                        rho_before = rho_arr[inject_arr]
                        neigh_sums = neigh_sums_all[inject_arr]
                        counts = (ptr[1:] - ptr[:-1])[inject_arr]
                        mean = np.divide(
                            neigh_sums,
                            counts,
                            out=np.zeros_like(neigh_sums, dtype=np.float32),
                            where=counts > 0,
                        )
                        if mode == "incoming":
                            layer_idx = layer_idx_map.get(lccm.layer, 0)
                            intensity_vec = np.empty(len(valid), dtype=np.float32)
                            for i, e in enumerate(valid):
                                intensity_vec[i] = edge_intensity[int(e)][layer_idx]
                        else:
                            intensity_vec = np.full(
                                len(valid), intensity, dtype=np.float32
                            )
                        rho_after, d_eff = update_rho_delay_vec(
                            rho_before,
                            mean,
                            intensity_vec,
                            alpha_d=alpha_d,
                            alpha_leak=alpha_leak,
                            eta=eta,
                            d0=edges["d0"][inject_arr],
                            gamma=gamma,
                            rho0=rho0,
                        )
                        rho_arr[inject_arr] = rho_after
                        if self._neigh_sums_cache is not None:
                            for idx, rb, ra in zip(inject_arr, rho_before, rho_after):
                                delta = ra - rb
                                if delta != 0 and ptr is not None and nbr is not None:
                                    start = ptr[int(idx)]
                                    end = ptr[int(idx) + 1]
                                    self._neigh_sums_cache[nbr[start:end]] += delta
                        prev_d_eff: np.ndarray | None = None
                        if "d_eff" in edges:
                            prev_d_eff = edges["d_eff"][inject_arr].astype(int).copy()
                            edges["d_eff"][inject_arr] = d_eff
                        for i, (idx, rb, ra, de) in enumerate(
                            zip(inject_arr, rho_before, rho_after, d_eff)
                        ):
                            injected[int(idx)] = (float(rb), float(ra), int(de))
                            if prev_d_eff is not None and int(prev_d_eff[i]) != int(de):
                                self._delay_changed.add(dst)
                                self._delay_changed.add(int(edges["dst"][int(idx)]))
                else:
                    layer_idx = layer_idx_map.get(lccm.layer, 0)
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
                            if mode == "incoming":
                                intensity_val = edge_intensity[int(edge_idx)][layer_idx]
                            else:
                                intensity_val = intensity
                            rho_before = float(edges["rho"][edge_idx])
                            rho_after, d_eff = update_rho_delay(
                                rho_before,
                                neighbours,
                                intensity_val,
                                alpha_d=alpha_d,
                                alpha_leak=alpha_leak,
                                eta=eta,
                                d0=float(edges["d0"][edge_idx]),
                                gamma=gamma,
                                rho0=rho0,
                            )
                            edges["rho"][edge_idx] = rho_after
                            if (
                                self._neigh_sums_cache is not None
                                and ptr is not None
                                and nbr is not None
                            ):
                                delta = rho_after - rho_before
                                if delta != 0:
                                    self._neigh_sums_cache[nbr[start:end]] += delta
                            if "d_eff" in edges:
                                old_de = int(edges["d_eff"][edge_idx])
                                edges["d_eff"][edge_idx] = d_eff
                                if old_de != int(d_eff):
                                    self._delay_changed.add(dst)
                                    self._delay_changed.add(int(edges["dst"][edge_idx]))
                            injected[int(edge_idx)] = (
                                float(rho_before),
                                float(rho_after),
                                int(d_eff),
                            )

            log_rho_edges = self._rng.random() < self._cfg.logging.get(
                "sample_rho_rate", 0.0
            )
            for edge_idx in self._edges_by_src.get(dst, []):
                rho_before = float(edges["rho"][edge_idx])
                if edge_idx in injected:
                    rho_after, d_eff = injected[edge_idx][1], injected[edge_idx][2]
                else:
                    rho_after = rho_before
                    if "d_eff" in edges:
                        d_eff = int(edges["d_eff"][edge_idx])
                    else:
                        d_eff = int(
                            effective_delay(
                                rho_before,
                                d0=float(edges["d0"][edge_idx]),
                                gamma=gamma,
                                rho0=rho0,
                            )
                        )
                depth_next = depth_arr + d_eff
                payload_buf["psi"] = self._arrays.vertices["psi"][dst]
                payload_buf["p"] = self._arrays.vertices["p"][dst]
                payload_buf["bit"] = int(self._arrays.vertices["bit"][dst])
                payload_buf["lambda_u"] = packet_data.get("lambda_u")
                payload_buf["zeta"] = packet_data.get("zeta")
                payload_buf["ancestry"] = packet_data.get("ancestry")
                payload_buf["m"] = packet_data.get("m")
                new_pkt = Packet(
                    src=dst,
                    dst=int(edges["dst"][edge_idx]),
                    payload=payload_buf.copy(),
                )
                edge_logs += 1
                if log_rho_edges:
                    log_record(
                        category="event",
                        label="edge_delivery",
                        frame=self._frame,
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
                payload_buf["psi"] = self._arrays.vertices["psi"][dst]
                payload_buf["p"] = self._arrays.vertices["p"][dst]
                payload_buf["bit"] = int(self._arrays.vertices["bit"][dst])
                payload_buf["lambda_u"] = packet_data.get("lambda_u")
                payload_buf["zeta"] = packet_data.get("zeta")
                payload_buf["ancestry"] = packet_data.get("ancestry")
                payload_buf["m"] = packet_data.get("m")
                edge_logs += 1
                rate = self._cfg.logging.get("sample_rho_rate", 0.0)
                if rate > 0.0 and self._rng.random() < rate:
                    log_record(
                        category="event",
                        label="edge_delivery",
                        frame=self._frame,
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
                    Packet(src=dst, dst=other, payload=payload_buf.copy()),
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
        flush_metrics(self._frame)
        self._frame += 1
        log_record(
            category="tick",
            label="adapter_frame",
            frame=self._frame,
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
            frame=self._frame,
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
                        delta = self._cfg.ancestry.get("delta_m", 0.0)
                        m *= 1.0 - delta
                        norm = float(np.linalg.norm(m))
                        m /= max(norm, 1e-12)
                        v_arr["m0"][vid] = m[0]
                        v_arr["m1"][vid] = m[1]
                        v_arr["m2"][vid] = m[2]
                        v_arr["m_norm"][vid] = norm
                    match self._cfg.theta_reset:
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
                    p_v = np.where(np.isfinite(p_v), p_v, np.zeros_like(p_v))
                    self._arrays.vertices["p"][vid] = p_v
                    if lccm.layer != "C":
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
                    idxs = (
                        adj["incident_idx"][start:end]
                        if end > start
                        else np.array([], dtype=int)
                    )
                    if len(idxs) > 0:
                        rhos = edges_arr["rho"][idxs]
                        weights = (
                            edges_arr["alpha"][idxs]
                            if "alpha" in edges_arr
                            else np.ones_like(rhos)
                        )
                    else:
                        rhos = np.array([], dtype=float)
                        weights = np.array([], dtype=float)
                    base_deg = data.get("base_deg", lccm.deg)
                    deg = base_deg + len(self._epairs.partners(vid))
                    st = data["win_state"]
                    alpha_rho = self._cfg.windowing.get("alpha_rho", 0.1)
                    half_life = math.log(2.0) / max(alpha_rho, 1e-9)
                    params = WindowParams(
                        W0=lccm.W0,
                        brho=lccm.zeta2,
                        rho0=lccm.rho0,
                        bdeg=lccm.zeta1,
                        deg0=self._cfg.windowing.get("deg0", 3.0),
                        half_life_windows=half_life,
                        beta=self._cfg.windowing.get("beta_W", 0.5),
                        mu=self._cfg.windowing.get("mu_W", None),
                    )
                    on_window_close(rhos, weights, params, st, k=1, deg_v=deg)
                    rho_mean = float(st.M_v)
                    W_smoothed = int(max(1, round(st.W_v)))
                    self._arrays.vertices["rho_mean"][vid] = rho_mean
                    lccm.rho_mean = rho_mean
                    lccm.window_override = W_smoothed
                    lccm.window_idx = lccm.depth // W_smoothed
                    k_rho = self._cfg.windowing.get("k_rho", 1.0)
                    E_rho = k_rho * rho_mean
                    v_arr["E_rho"][vid] = E_rho
                    lccm.deg = deg
                else:
                    EQ = lccm._eq
                    E_theta = 0.0
                    E_C = 0.0
                    E_rho = 0.0
                log_record(
                    category="event",
                    label="vertex_window_close",
                    frame=self._frame,
                    tick=self._frame,
                    value={
                        "layer": lccm.layer,
                        "Lambda_v": lccm._lambda,
                        "EQ": EQ,
                        "H_p": H_pv,
                        "E_theta": E_theta,
                        "E_C": E_C,
                        "E_rho": E_rho,
                        "v_id": vid,
                    },
                    metadata={"window_idx": lccm.window_idx},
                )
                if self._cfg.epsilon_pairs.get("decay_on_window_close", True):
                    # Decay bridges when vertices advance a window.
                    self._epairs.decay_all()

        if self._delay_changed and self._epairs is not None:
            for vid in self._delay_changed:
                self._epairs.adjust_d_bridge(vid)
            self._delay_changed.clear()

        return frame

    def snapshot_for_ui(self) -> dict:
        """Return a minimal snapshot for the GUI."""

        with self._lock:
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


# Lazily constructed engine instance; GUI code may read this handle but
# should not mutate it.  External callers must obtain the adapter via
# :func:`get_engine` to avoid import-time side effects.
_ENGINE: EngineAdapter | None = None


def get_engine() -> EngineAdapter:
    """Return the module-level :class:`EngineAdapter` instance.

    The adapter is constructed lazily on first use to avoid import-time
    side effects.  Callers should use this factory rather than relying on
    implicit creation when the module is imported.
    """

    global _ENGINE
    if _ENGINE is None:
        _ENGINE = EngineAdapter()
    return _ENGINE


def build_graph(graph_json_path: str | Dict[str, Any] | None = None) -> None:
    """Build the simulation graph for headless runs."""

    from ...config import Config

    engine = get_engine()
    path = graph_json_path or Config.graph_file
    engine.build_graph(path)


def simulation_loop() -> None:
    """Start a background loop advancing the engine while running."""

    from ...config import Config

    engine = get_engine()

    def _run() -> None:
        while True:
            with Config.state_lock:
                if not Config.is_running or (
                    Config.frame_limit and Config.current_frame >= Config.frame_limit
                ):
                    Config.is_running = False
                    break
            engine.step()
            with Config.state_lock:
                Config.current_frame += 1
                Config.current_tick = Config.current_frame
            time.sleep(0)

    threading.Thread(target=_run, daemon=True).start()


def pause_simulation() -> None:
    """Pause execution of the simulation."""

    from ...config import Config

    engine = get_engine()
    engine.pause()
    with Config.state_lock:
        Config.is_running = False


def resume_simulation() -> None:
    """Resume a previously paused simulation."""

    from ...config import Config

    engine = get_engine()
    engine.start()
    with Config.state_lock:
        Config.is_running = True
    simulation_loop()


def stop_simulation() -> None:
    """Stop the simulation and reset state."""

    from ...config import Config

    engine = get_engine()
    engine.stop()
    with Config.state_lock:
        Config.is_running = False


def get_snapshot() -> dict:
    """Return a minimal snapshot for the UI."""

    engine = get_engine()
    return engine.snapshot_for_ui()


__all__ = [
    "EngineAdapter",
    "get_engine",
    "build_graph",
    "simulation_loop",
    "pause_simulation",
    "resume_simulation",
    "stop_simulation",
    "get_snapshot",
]
