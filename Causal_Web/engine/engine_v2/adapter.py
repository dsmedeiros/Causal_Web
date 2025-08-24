"""Compatibility layer for the v2 engine prototype.

The :class:`EngineAdapter` exposes a subset of the legacy tick engine API but
drives a new depth-based scheduler and the lightweight :mod:`lccm` model.  The
adapter processes packets ordered by arrival depth and advances vertex windows
according to the local causal consistency math. The module has been refactored
to remove outdated hooks from the legacy engine.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set, Tuple
import json
import os
from pathlib import Path
from collections import deque, defaultdict, OrderedDict
import threading
import time
import random
import math

import numpy as np

from Causal_Web.view import EdgeView, NodeView, ViewSnapshot, WindowEvent

from ..logging.logger import log_record, flush_metrics, _get_aggregator
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
from experiments.gates import run_gates
from experiments.policy import ACTION_SET
from invariants import checks


EDGE_LOG_BUDGET = 100
POOL_MAX_ENTRIES = 8
EPSILON = 0.0


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
        self._psi_arr_pool: OrderedDict[tuple[int, ...], deque[np.ndarray]] = (
            OrderedDict()
        )
        self._p_arr_pool: OrderedDict[tuple[int, ...], deque[np.ndarray]] = (
            OrderedDict()
        )
        self._phase_arr_pool: OrderedDict[tuple[int, ...], deque[np.ndarray]] = (
            OrderedDict()
        )
        self._U_arr_pool: OrderedDict[tuple[int, ...], deque[np.ndarray]] = (
            OrderedDict()
        )
        self._alpha_arr_pool: OrderedDict[tuple[int, ...], deque[np.ndarray]] = (
            OrderedDict()
        )
        self._neigh_sums_cache: np.ndarray | None = None
        self._delay_changed: Set[int] = set()
        self._lock = threading.RLock()
        self._last_snapshot_time = time.time()
        self._last_frame = 0
        self._changed_nodes: Set[int] = set()
        self._changed_edges: Set[tuple[int, int]] = set()
        self._closed_windows: list[WindowEvent] = []
        self._energy_totals: Dict[int, float] = {}
        self._residuals: Dict[int, float] = {}
        self._residual: float = 0.0
        self._residual_ewma: float = 0.0
        self._residual_max: float = 0.0
        self._theta_reset: bool = False
        self._eps_emit: float = 0.0
        self._Wmax: float = 64.0
        self._MI_mode: bool = False
        self._ns_plus: int = 0
        self._ns_total: int = 0
        self._ns_delta: float = 0.0
        self._edges_traversed: int = 0
        self._windows_closed_total: int = 0
        self._experiment_status: Dict[str, Any] | None = None
        self._replay_progress: float | None = None
        self._replay_playing: bool = False
        self._target_rate: float = 1.0
        self._thread: threading.Thread | None = None
        self._current_delta: Dict[str, Any] | None = None
        self._graph_static: Dict[str, Any] | None = None
        self._replay_frames: List[Dict[str, Any]] = []
        self._replay_index = 0

    # ------------------------------------------------------------------
    @property
    def residual(self) -> float:
        """Return the latest residual metric."""

        return self._residual

    @residual.setter
    def residual(self, value: float) -> None:
        """Update the residual metric."""

        self._residual = float(value)

    # ------------------------------------------------------------------
    @property
    def theta_reset(self) -> bool:
        """Flag indicating whether theta reset is enabled."""

        return self._theta_reset

    @theta_reset.setter
    def theta_reset(self, value: bool) -> None:
        self._theta_reset = bool(value)

    # ------------------------------------------------------------------
    @property
    def eps_emit(self) -> float:
        """Magnitude of the epsilon emission boost."""

        return self._eps_emit

    @eps_emit.setter
    def eps_emit(self, value: float) -> None:
        self._eps_emit = float(value)

    # ------------------------------------------------------------------
    @property
    def Wmax(self) -> float:
        """Current clamp on window size."""

        return self._Wmax

    @Wmax.setter
    def Wmax(self, value: float) -> None:
        self._Wmax = float(value)

    # ------------------------------------------------------------------
    @property
    def MI_mode(self) -> bool:
        """Flag toggling the adversarial MI mode."""

        return self._MI_mode

    @MI_mode.setter
    def MI_mode(self, value: bool) -> None:
        self._MI_mode = bool(value)

    # ------------------------------------------------------------------
    def _apply_policy_effects(self) -> None:
        """Recompute residual based on current policy flags."""

        res = self._residual
        if self._theta_reset:
            res *= 0.5
        res = max(res - self._eps_emit, 0.0)
        res = max(res - max(0.0, 64.0 - self._Wmax), 0.0)
        if self._MI_mode:
            res += 4.0
        self._residual = res

    # ------------------------------------------------------------------
    def load_replay(self, path: str | os.PathLike[str]) -> dict[str, Any] | None:
        """Load replay data from ``path`` directory and return ``GraphStatic``.

        The directory may contain ``graph_static.json`` describing the graph and
        a ``delta_log.jsonl`` file with one JSON snapshot delta per line. Missing
        files are ignored.
        """

        p = Path(path)
        with self._lock:
            self._replay_frames.clear()
            self._replay_index = 0
            self._graph_static = None
            graph_file = p / "graph_static.json"
            if graph_file.exists():
                try:
                    self._graph_static = json.loads(graph_file.read_text())
                except json.JSONDecodeError:
                    self._graph_static = None
            delta_file = p / "delta_log.jsonl"
            if delta_file.exists():
                for line in delta_file.read_text().splitlines():
                    try:
                        self._replay_frames.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            self._replay_playing = False
            self._replay_progress = 0.0
            return self._graph_static

    # ------------------------------------------------------------------
    def _get_pool_arr(
        self,
        pool: OrderedDict[tuple[int, ...], deque[np.ndarray]],
        base_shape: tuple[int, ...],
        rows: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Return an array from ``pool`` sized for ``rows`` in bucketed groups.

        Arrays are grouped by ``rows`` rounded up to the next power-of-two so
        transient batch sizes reuse a small set of allocations. ``pool`` is keyed
        by ``(bucket, *base_shape)`` where ``bucket`` is this rounded row count.
        """

        bucket = 1 << (rows - 1).bit_length()
        key = (bucket, *base_shape)
        dq = pool.setdefault(key, deque())
        arr = dq.pop() if dq else None
        if arr is None:
            arr = np.empty((bucket, *base_shape), dtype=dtype)
        pool[key] = dq
        pool.move_to_end(key)
        while len(pool) > POOL_MAX_ENTRIES:
            pool.popitem(last=False)
        return arr

    def _recycle_pool_arr(
        self,
        pool: OrderedDict[tuple[int, ...], deque[np.ndarray]],
        base_shape: tuple[int, ...],
        arr: np.ndarray,
    ) -> None:
        """Return ``arr`` to ``pool`` grouped by its bucketed row count."""

        key = (arr.shape[0], *base_shape)
        dq = pool.setdefault(key, deque())
        dq.append(arr)
        pool.move_to_end(key)
        while len(dq) > POOL_MAX_ENTRIES:
            dq.popleft()
        while len(pool) > POOL_MAX_ENTRIES:
            pool.popitem(last=False)

    # ------------------------------------------------------------------
    def _splitmix64(self, x: int) -> int:
        """Return a SplitMix64 hash of ``x``.

        Operations use Python integers and explicit masking to emulate
        unsigned 64-bit wraparound behaviour.
        """

        z = (int(x) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

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
        def f2u64(x: float) -> int:
            return int.from_bytes(np.float64(x).tobytes(), "little")

        mask = 0xFFFFFFFFFFFFFFFF
        h0 = (int(h0) ^ int(dst) ^ ((int(depth_arr) << 1) & mask)) & mask
        h1 = (int(h1) ^ (int(edge_id) & mask) ^ ((int(seq) << 1) & mask)) & mask
        h2 = (int(h2) ^ f2u64(mu)) & mask
        h3 = (int(h3) ^ f2u64(kappa)) & mask
        h0 = self._splitmix64(h0)
        h1 = self._splitmix64(h1)
        h2 = self._splitmix64(h2)
        h3 = self._splitmix64(h3)

        v_arr["h0"][dst] = np.uint64(h0)
        v_arr["h1"][dst] = np.uint64(h1)
        v_arr["h2"][dst] = np.uint64(h2)
        v_arr["h3"][dst] = np.uint64(h3)

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
            sample_seed_rate=self._cfg.logging.get("sample_seed_rate", 0.01),
            sample_bridge_rate=self._cfg.logging.get("sample_bridge_rate", 0.01),
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
                "bit_deque": deque(maxlen=POOL_MAX_ENTRIES),
                "base_deg": deg,
                "win_state": WindowState(M_v=rho_mean, W_v=w_init),
            }
            self._vertices[vid] = vertex_state

        v_arr = self._arrays.vertices
        e_arr = self._arrays.edges
        x_vals = v_arr.get("x")
        y_vals = v_arr.get("y")
        if x_vals is not None and y_vals is not None:
            positions = [(float(x_vals[i]), float(y_vals[i])) for i in range(n_vert)]
        else:
            positions = [(0.0, 0.0) for _ in range(n_vert)]
        edge_list = [
            (int(e_arr["src"][i]), int(e_arr["dst"][i]))
            for i in range(len(e_arr["src"]))
        ]
        labels = [str(i) for i in range(n_vert)]
        colors = ["#ffffff" for _ in range(n_vert)]
        flags = [True for _ in range(n_vert)]
        self._graph_static = {
            "node_positions": positions,
            "edges": edge_list,
            "node_labels": labels,
            "node_colors": colors,
            "node_flags": flags,
        }

    def graph_static(self) -> Dict[str, Any]:
        """Return a static description of the loaded graph for the UI."""

        return self._graph_static or {
            "node_positions": [],
            "edges": [],
            "node_labels": [],
            "node_colors": [],
            "node_flags": [],
        }

    def start(self) -> None:
        """Mark the engine as running."""

        if self._running:
            return
        self._running = True
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def pause(self) -> None:
        """Pause execution."""
        self._running = False

    def stop(self) -> None:
        """Stop execution and reset all state."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.1)
            self._thread = None
        self._vertices.clear()
        self._scheduler.clear()
        self._replay_frames.clear()
        self._replay_index = 0
        self._current_delta = None

    def step(
        self, max_events: int | None = None, *, collect_packets: bool = False
    ) -> TelemetryFrame:
        """Advance the simulation until a window rolls or ``max_events``.

        This call is thread-safe so the UI can request snapshots while the
        simulation advances in a background thread.
        """

        with self._lock:
            return self.run_until_next_window_or(
                max_events, collect_packets=collect_packets
            )

    def run_until_next_window_or(
        self, limit: int | None, *, collect_packets: bool = False
    ) -> TelemetryFrame:
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
        packets: list[Packet] | None = [] if collect_packets else None
        edge_logs = 0
        diagnostic_mode = "diagnostic" in getattr(self._cfg, "logging_mode", [])
        event_logs = 0
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
            self._changed_nodes.add(dst)
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
            pkt_list = [pkt] if collect_packets else []
            pkt_count = 1
            edge_id_list = [edge_id]
            seq_list = [seq]
            window_idx = lccm.window_idx
            requeue: list[tuple[int, int, int, Packet]] = []
            while self._scheduler and events + pkt_count < limit:
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
                if collect_packets:
                    pkt_list.append(pkt2)
                edge_id_list.append(edge2)
                seq_list.append(seq2)
                pkt_count += 1
            for item in requeue:
                self._scheduler.push(*item)

            dim = U_list[0].shape[0] if U_list else 0
            U_buf = self._get_pool_arr(
                self._U_arr_pool, (dim, dim), pkt_count, np.complex64
            )
            for i, U_mat in enumerate(U_list):
                U_buf[i] = U_mat
            phase_buf = self._get_pool_arr(
                self._phase_arr_pool, (), pkt_count, np.complex64
            )
            phase_buf[:pkt_count] = phase_list
            alpha_buf = self._get_pool_arr(
                self._alpha_arr_pool, (), pkt_count, np.float32
            )
            alpha_buf[:pkt_count] = alpha_list
            psi_len = len(psi_list[0]) if psi_list else 0
            psi_buf = self._get_pool_arr(
                self._psi_arr_pool, (psi_len,), pkt_count, np.complex64
            )
            for i, psi_vec in enumerate(psi_list):
                psi_buf[i] = psi_vec
            mu_arr, kappa_arr, psi_rot_arr = phase_stats_batch(
                U_buf[:pkt_count], phase_buf[:pkt_count], psi_buf[:pkt_count]
            )
            mu_list = mu_arr.tolist()
            kappa_list = kappa_arr.tolist()

            p_len = len(p_list[0]) if p_list else 0
            p_buf = self._get_pool_arr(
                self._p_arr_pool, (p_len,), pkt_count, np.float32
            )
            for i, p_vec in enumerate(p_list):
                p_buf[i] = p_vec
            theta_vals = np.sum(np.abs(p_buf[:pkt_count]), axis=1)
            theta_int_arr = np.minimum(1.0, theta_vals)
            theta_mean = float(np.mean(theta_vals)) if len(theta_vals) else 0.0
            theta_mean = min(1.0, theta_mean)

            q_int_arr = np.minimum(1.0, np.linalg.norm(psi_rot_arr, axis=1) ** 2)
            c_int_arr = np.asarray(bit_list, dtype=float)
            pkt_intensities = list(
                zip(q_int_arr.tolist(), theta_int_arr.tolist(), c_int_arr.tolist())
            )

            if pkt_count > 1:
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
                    psi_buf[:pkt_count],
                    p_buf[:pkt_count],
                    bit_list,
                    depth_list,
                    alpha_buf[:pkt_count],
                    phase_buf[:pkt_count],
                    U_buf[:pkt_count],
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

            self._recycle_pool_arr(self._U_arr_pool, (dim, dim), U_buf)
            self._recycle_pool_arr(self._phase_arr_pool, (), phase_buf)
            self._recycle_pool_arr(self._alpha_arr_pool, (), alpha_buf)
            self._recycle_pool_arr(self._psi_arr_pool, (psi_len,), psi_buf)
            self._recycle_pool_arr(self._p_arr_pool, (p_len,), p_buf)

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
            p_v[:] = np.where(np.isfinite(p_v), p_v, 0.0)
            entropy = float(-(p_v * np.log2(p_v + 1e-12)).sum()) if len(p_v) else 0.0
            lccm.update_classical_metrics(bit_fraction, entropy, conf)
            is_q = lccm.layer == "Q"
            lccm.deliver(is_q)
            if collect_packets and packets is not None:
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
                    self._ns_total += 1
                    if outcome == 1:
                        self._ns_plus += 1
                    self._ns_delta = abs(self._ns_plus / self._ns_total - 0.5)
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
                    event_logs += 1
                    if diagnostic_mode or event_logs % EDGE_LOG_BUDGET == 0:
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

            rate = self._cfg.logging.get("sample_rho_rate", 0.01)
            edge_ids_all = self._edges_by_src.get(dst)
            if edge_ids_all is not None and len(edge_ids_all) > 0:
                edge_arr = np.asarray(edge_ids_all, dtype=np.int32)
                dst_arr = edges["dst"][edge_arr].astype(int)
                rho_before_arr = edges["rho"][edge_arr].astype(float)
                if "d_eff" in edges:
                    d_eff_arr = edges["d_eff"][edge_arr].astype(int)
                else:
                    d0_arr = edges["d0"][edge_arr]
                    d_eff_arr = np.empty_like(edge_arr, dtype=int)
                    for i, (rb, d0_val) in enumerate(zip(rho_before_arr, d0_arr)):
                        d_eff_arr[i] = int(
                            effective_delay(
                                rb, d0=float(d0_val), gamma=gamma, rho0=rho0
                            )
                        )
                rho_after_arr = rho_before_arr.copy()
                for i, eid in enumerate(edge_arr):
                    if int(eid) in injected:
                        rho_after_arr[i] = injected[int(eid)][1]
                        d_eff_arr[i] = injected[int(eid)][2]
                depth_next_arr = depth_arr + d_eff_arr

                payload_template = {
                    "psi": self._arrays.vertices["psi"][dst],
                    "p": self._arrays.vertices["p"][dst],
                    "bit": int(self._arrays.vertices["bit"][dst]),
                    "lambda_u": packet_data.get("lambda_u"),
                    "zeta": packet_data.get("zeta"),
                    "ancestry": packet_data.get("ancestry"),
                    "m": packet_data.get("m"),
                }

                sched_push = self._scheduler.push
                reinforce = self._epairs.reinforce
                rng_rand = self._rng.random
                sigma_arr = edges["sigma"]

                for i in range(len(edge_arr)):
                    eid = int(edge_arr[i])
                    dst_id = int(dst_arr[i])
                    rho_before = float(rho_before_arr[i])
                    rho_after = float(rho_after_arr[i])
                    d_eff = int(d_eff_arr[i])
                    depth_next = int(depth_next_arr[i])

                    payload = payload_template.copy()
                    new_pkt = Packet(src=dst, dst=dst_id, payload=payload)

                    edge_logs += 1
                    self._changed_nodes.add(dst_id)
                    self._changed_edges.add((dst, dst_id))
                    if (
                        (diagnostic_mode or edge_logs % EDGE_LOG_BUDGET == 0)
                        and rate > EPSILON
                        and rng_rand() < rate
                    ):
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
                                "sigma": float(sigma_arr[eid]),
                            },
                        )
                    sched_push(depth_next, dst_id, eid, new_pkt)
                    reinforce(dst, dst_id)

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
                self._changed_nodes.update({dst, other})
                self._changed_edges.add((dst, other))
                rate = self._cfg.logging.get("sample_rho_rate", 0.01)
                if (
                    (diagnostic_mode or edge_logs % EDGE_LOG_BUDGET == 0)
                    and rate > EPSILON
                    and self._rng.random() < rate
                ):
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
            events += pkt_count

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
            depth=max_depth,
            events=events,
            packets=packets if collect_packets else None,
            window=max_window,
        )
        self._edges_traversed += edge_logs
        self._windows_closed_total += len(self._closed_windows)
        bridges_active = len(self._epairs.bridges) if self._epairs is not None else 0
        agg = _get_aggregator()
        agg.counts["events_processed"] += events
        agg.counts["edges_traversed"] += edge_logs
        agg.counts["windows_closed"] += len(self._closed_windows)
        agg.counts["bridges_active"] += bridges_active

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

        residual_max_local = 0.0
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
                            if total > EPSILON:
                                p_v /= total
                        case "hold":
                            pass
                        case _:
                            p_v.fill(1.0 / len(p_v))
                    p_v[:] = np.where(np.isfinite(p_v), p_v, EPSILON)
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
                event_logs += 1
                if diagnostic_mode or event_logs % EDGE_LOG_BUDGET == 0:
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
                E_total = EQ + E_theta + E_C + E_rho
                prev = self._energy_totals.get(vid, 0.0)
                leak_coeff = self._cfg.rho_delay.get("alpha_leak", 0.0)
                leak = leak_coeff * prev
                resid = E_total - prev - leak
                alpha_res = self._cfg.windowing.get("alpha_residual", 0.1)
                prev_res = self._residuals.get(vid, 0.0)
                abs_resid = abs(resid)
                self._residuals[vid] = (
                    1 - alpha_res
                ) * prev_res + alpha_res * abs_resid
                residual_max_local = max(residual_max_local, abs_resid)
                self._energy_totals[vid] = E_total
                if self._residuals:
                    self._residual = float(np.mean(list(self._residuals.values())))
                    self._residual_ewma = (
                        1 - alpha_res
                    ) * self._residual_ewma + alpha_res * self._residual
                self._closed_windows.append(
                    WindowEvent(v_id=str(vid), window_idx=lccm.window_idx)
                )
                self._changed_nodes.add(vid)
                if self._cfg.epsilon_pairs.get("decay_on_window_close", True):
                    # Decay bridges when vertices advance a window.
                    self._epairs.decay_all()

        self._residual_max = residual_max_local

        if self._delay_changed and self._epairs is not None:
            for vid in self._delay_changed:
                self._epairs.adjust_d_bridge(vid)
            self._delay_changed.clear()

        return frame

    def snapshot_for_ui(self) -> ViewSnapshot:
        """Return a :class:`ViewSnapshot` capturing recent changes for the GUI.

        The returned counters include an EWMA of the energy conservation
        residual over recent window closures.
        """

        with self._lock:
            max_depth = 0
            max_window = 0
            for data in self._vertices.values():
                lccm = data["lccm"]
                max_depth = max(max_depth, lccm.depth)
                max_window = max(max_window, lccm.window_idx)
            now = time.time()
            elapsed = now - self._last_snapshot_time
            events_per_sec = 0.0
            if elapsed > 0:
                events_per_sec = (self._frame - self._last_frame) / elapsed
            self._last_snapshot_time = now
            self._last_frame = self._frame
            counters = {
                "window": max_window,
                "events_per_sec": events_per_sec,
                "gates_fired": getattr(self._epairs, "gates_fired", 0),
                "active_bridges": len(getattr(self._epairs, "active_bridges", [])),
                "residual": self._residual,
            }
            nodes = [NodeView(id=str(n)) for n in self._changed_nodes]
            edges = [EdgeView(src=str(s), dst=str(d)) for s, d in self._changed_edges]
            closed = list(self._closed_windows)
            self._changed_nodes.clear()
            self._changed_edges.clear()
            self._closed_windows.clear()
            return ViewSnapshot(
                frame=max_depth,
                changed_nodes=nodes,
                changed_edges=edges,
                closed_windows=closed,
                counters=counters,
                invariants={
                    "inv_conservation_residual": self._residual,
                    "inv_no_signaling_delta": self._ns_delta,
                },
            )

    def current_depth(self) -> int:
        """Return the current depth of the simulation."""

        max_depth = 0
        for data in self._vertices.values():
            max_depth = max(max_depth, data["lccm"].depth)
        return max_depth

    def current_frame(self) -> int:
        """Return the number of steps executed so far."""

        return self._frame

    # ------------------------------------------------------------------
    def _build_delta(self) -> Dict[str, Any] | None:
        """Coalesce geometry changes, metrics and window events into a delta."""

        if self._arrays is None:
            return None
        v_arr = self._arrays.vertices

        # Depth and window counters
        max_depth = 0
        max_window = 0
        for data in self._vertices.values():
            lccm = data["lccm"]
            max_depth = max(max_depth, lccm.depth)
            max_window = max(max_window, lccm.window_idx)

        delta: Dict[str, Any] = {"frame": max_depth}

        positions: Dict[int, tuple[float, float]] = {}
        x_vals = v_arr.get("x")
        y_vals = v_arr.get("y")
        for vid in self._changed_nodes:
            if x_vals is not None and y_vals is not None:
                positions[int(vid)] = (
                    float(np.float32(x_vals[vid])),
                    float(np.float32(y_vals[vid])),
                )
            else:
                positions[int(vid)] = (0.0, 0.0)
        if positions:
            delta["node_positions"] = positions
        if self._changed_edges:
            delta["edges"] = [(int(a), int(b)) for a, b in self._changed_edges]
        if self._closed_windows:
            delta["closed_windows"] = [
                (ev.v_id, ev.window_idx) for ev in self._closed_windows
            ]

        now = time.time()
        elapsed = now - self._last_snapshot_time
        events_per_sec = 0.0
        if elapsed > 0:
            events_per_sec = (self._frame - self._last_frame) / elapsed
        self._last_snapshot_time = now
        self._last_frame = self._frame
        counters = {
            "window": max_window,
            "windows_closed": self._windows_closed_total,
            "bridges_active": len(getattr(self._epairs, "bridges", {})),
            "events_processed": self._frame,
            "edges_traversed": self._edges_traversed,
            "events_per_sec": float(np.float32(events_per_sec)),
            "gates_fired": getattr(self._epairs, "gates_fired", 0),
            "residual": float(np.float32(self._residual)),
            "residual_ewma": float(np.float32(self._residual_ewma)),
            "residual_max": float(np.float32(self._residual_max)),
        }
        delta["counters"] = counters
        delta["invariants"] = {
            "inv_conservation_residual": float(np.float32(self._residual)),
            "inv_no_signaling_delta": float(np.float32(self._ns_delta)),
        }

        self._changed_nodes.clear()
        self._changed_edges.clear()
        self._closed_windows.clear()
        return delta if len(delta) > 1 else None

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        """Background thread advancing the experiment and recording deltas."""

        while self._running:
            self.step()
            delta = self._build_delta()
            if delta:
                with self._lock:
                    self._current_delta = delta
                    self._replay_frames.append(dict(delta))
                log_record("delta", "snapshot", frame=self._frame, value=delta)
            self.set_experiment_status(
                {"status": "running", "residual": self._residual}
            )
            time.sleep(0)

    # ------------------------------------------------------------------
    def snapshot_delta(self) -> Dict[str, Any] | None:
        """Return the next snapshot delta for live or replay playback."""

        with self._lock:
            if self._running:
                delta = self._current_delta
                self._current_delta = None
                return delta
            if self._replay_playing and self._replay_frames:
                if self._replay_index >= len(self._replay_frames):
                    self._replay_playing = False
                    self._replay_progress = 1.0
                    return None
                delta = self._replay_frames[self._replay_index]
                self._replay_index += 1
                self._replay_progress = self._replay_index / len(self._replay_frames)
                return delta
        return None

    # ------------------------------------------------------------------
    def handle_control(self, msg: Dict[str, Any]) -> Dict[str, Any] | None:
        """Handle experiment, replay and graph control messages from the UI."""

        if msg.get("cmd") == "load_graph":
            graph = msg.get("graph")
            if graph is not None:
                self.build_graph(graph)
                return {"type": "GraphStatic", "v": 1, **self.graph_static()}
            return None

        pol = msg.get("PolicyControl")
        if pol:
            names: list[str] = []
            action = pol.get("action")
            if isinstance(action, str):
                names.append(action)
            names.extend(pol.get("actions", []))
            for name in names:
                func = ACTION_SET.get(name)
                if func:
                    func(self)
            self._apply_policy_effects()
            self.set_experiment_status(
                {"status": "running", "residual": self._residual}
            )
            return None

        exp = msg.get("ExperimentControl")
        if exp:
            action = exp.get("action")
            if action == "start":
                self.start()
                self.set_experiment_status(
                    {"status": "running", "residual": self._residual}
                )
            elif action == "pause":
                self.pause()
                self.set_experiment_status(
                    {"status": "paused", "residual": self._residual}
                )
            elif action == "resume":
                self.start()
                self.set_experiment_status(
                    {"status": "running", "residual": self._residual}
                )
            elif action == "reset":
                self.stop()
                self.set_experiment_status({"status": "idle", "residual": 0.0})
            elif action == "step":
                self.step()
                self.set_experiment_status(
                    {"status": "paused", "residual": self._residual}
                )
            elif action == "set_rate":
                self._target_rate = float(exp.get("rate", 1.0))
            elif action == "run":
                cfg = exp.get("config") or {}
                rid = exp.get("id", 0)
                gates = exp.get("gates", [1, 2, 3, 4, 5, 6])
                try:
                    metrics = run_gates(cfg, gates)
                    inv = checks.from_metrics(metrics)
                    self.set_experiment_status(
                        {
                            "id": rid,
                            "state": "finished",
                            "metrics": metrics,
                            "invariants": inv,
                        }
                    )
                except Exception as exc:
                    self.set_experiment_status(
                        {"id": rid, "state": "failed", "error": str(exc)}
                    )
            return None

        replay = msg.get("ReplayControl")
        if replay:
            action = replay.get("action")
            if action == "load":
                path = replay.get("path")
                if isinstance(path, str):
                    gs = self.load_replay(path)
                    if gs is not None:
                        return {"type": "GraphStatic", "v": 1, **gs}
                return None
            if action == "play":
                self._replay_playing = True
                self._replay_progress = self._replay_progress or 0.0
            elif action == "pause":
                self._replay_playing = False
                self._replay_progress = self._replay_progress or 0.0
            elif action == "seek":
                prog = float(replay.get("progress", 0.0))
                self._replay_index = int(prog * len(self._replay_frames))
                self._replay_progress = prog
            return None

        return None

    # ------------------------------------------------------------------
    def replay_progress(self) -> float | None:
        """Return and clear the latest replay progress if available."""

        with self._lock:
            progress = self._replay_progress
            self._replay_progress = None
        return progress

    # ------------------------------------------------------------------
    def set_experiment_status(self, status: Dict[str, Any]) -> None:
        """Record the most recent experiment status.

        Parameters
        ----------
        status:
            Mapping containing ``id``, ``state``, ``progress`` and
            ``best_metrics`` fields.
        """

        with self._lock:
            self._experiment_status = status

    # ------------------------------------------------------------------
    def experiment_status(self) -> Dict[str, Any] | None:
        """Return and clear the latest experiment status if available."""

        with self._lock:
            status = self._experiment_status
            self._experiment_status = None
        return status


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
