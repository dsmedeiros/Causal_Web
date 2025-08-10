"""Compatibility layer for the v2 engine prototype.

The :class:`EngineAdapter` exposes a subset of the legacy tick engine API but
drives a new depth-based scheduler and the lightweight :mod:`lccm` model.  The
adapter processes packets ordered by arrival depth and advances vertex windows
according to the local causal consistency math.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from collections import deque
import threading
import time

import numpy as np

from ..logging.logger import log_record
from .lccm import LCCM
from .scheduler import DepthScheduler
from .state import Packet, TelemetryFrame
from .loader import GraphArrays, load_graph_arrays
from .rho_delay import update_rho_delay
from .qtheta_c import close_window, deliver_packet
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

    # Public API -----------------------------------------------------
    def build_graph(self, graph_json_path: str | Dict[str, Any]) -> None:
        """Load a graph description."""

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
        C_min = window_defaults.get("C_min", 0.0)
        f_min = params.get("f_min", 1.0)
        H_max = params.get("H_max", 0.0)
        T_hold = window_defaults.get("T_hold", 1)
        T_class = params.get("T_class", 1)

        self._arrays = load_graph_arrays(graph)
        self._vertices.clear()
        edges = self._arrays.edges
        n_vert = len(self._arrays.vertices["depth"])
        for vid in range(n_vert):
            out_idx = np.where(edges["src"] == vid)[0]
            self._edges_by_src[vid] = out_idx
            deg = len(out_idx)
            rho_mean = float(self._arrays.vertices.get("rho_mean")[vid])
            lccm = LCCM(
                W0,
                zeta1,
                zeta2,
                rho0,
                a,
                b,
                C_min,
                f_min,
                H_max,
                T_hold,
                T_class,
                deg=deg,
                rho_mean=rho_mean,
            )
            vertex_state = {
                "lccm": lccm,
                "psi_acc": self._arrays.vertices["psi_acc"][vid],
                "p_v": self._arrays.vertices["p"][vid],
                "bit_deque": deque(maxlen=8),
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

        The returned frame's ``depth`` mirrors the greatest arrival depth
        encountered during processing. If no events are handled or only events
        at the current depth are processed, the depth will remain unchanged.
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

        while self._scheduler and events < limit:
            depth_arr, dst, edge_id, pkt = self._scheduler.pop()
            vertex = self._vertices.get(dst)
            if vertex is None:
                continue
            lccm = vertex["lccm"]
            lccm.advance_depth(depth_arr)
            prev_layer = lccm.layer

            packet_data = pkt.payload or {}
            if not isinstance(packet_data, dict):
                packet_data = {}
            packet_data.setdefault("depth_arr", depth_arr)
            if "psi" not in packet_data:
                packet_data["psi"] = np.zeros_like(vertex["psi_acc"])
            if "p" not in packet_data:
                packet_data["p"] = np.zeros_like(vertex["p_v"])
            if "bit" not in packet_data:
                packet_data["bit"] = 0

            edges = self._arrays.edges if self._arrays else {}
            dim = len(vertex["psi_acc"])
            edge_params: Dict[str, Any] = {
                "alpha": 1.0,
                "phi": 0.0,
                "A": 0.0,
                "U": np.eye(dim, dtype=np.complex64),
            }
            if edges and edge_id < len(edges.get("alpha", [])):
                edge_params.update(
                    {
                        "alpha": float(edges["alpha"][edge_id]),
                        "phi": float(edges["phi"][edge_id]),
                        "A": float(edges["A"][edge_id]),
                        "U": edges["U"][edge_id],
                    }
                )

            depth_v, psi_acc, p_v, (bit, conf), intensity = deliver_packet(
                lccm.depth,
                vertex["psi_acc"],
                vertex["p_v"],
                vertex["bit_deque"],
                packet_data,
                edge_params,
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
            lccm.update_classical_metrics(bit_fraction, entropy)
            lccm.deliver()
            packets.append(pkt)

            if lccm.layer != prev_layer:
                reason = "fanin_threshold" if prev_layer == "Q" else "decoh_threshold"
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
                    },
                )

            edges = self._arrays.edges if self._arrays else {}
            adj = self._arrays.adjacency if self._arrays else {}
            for edge_idx in self._edges_by_src.get(dst, []):
                rho_before = float(edges["rho"][edge_idx])
                ptr = adj["nbr_ptr"]
                nbr = adj["nbr_idx"]
                n_start = ptr[edge_idx]
                n_end = ptr[edge_idx + 1]
                neighbours = edges["rho"][nbr[n_start:n_end]]
                rho_after, d_eff = update_rho_delay(
                    rho_before,
                    neighbours,
                    intensity,
                    alpha_d=Config.rho_delay.get("alpha_d", 0.0),
                    alpha_leak=Config.rho_delay.get("alpha_leak", 0.0),
                    eta=Config.rho_delay.get("eta", 0.0),
                    d0=float(edges["d0"][edge_idx]),
                    gamma=Config.rho_delay.get("gamma", 0.0),
                    rho0=Config.rho_delay.get("rho0", 1.0),
                )
                edges["rho"][edge_idx] = rho_after
                if "d_eff" in edges:
                    edges["d_eff"][edge_idx] = d_eff
                depth_next = depth_arr + d_eff
                payload = {
                    "psi": self._arrays.vertices["psi"][dst],
                    "p": self._arrays.vertices["p"][dst],
                    "bit": int(self._arrays.vertices["bit"][dst]),
                }
                new_pkt = Packet(
                    src=dst, dst=int(edges["dst"][edge_idx]), payload=payload
                )
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
            events += 1

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

        for vid, data in self._vertices.items():
            lccm = data["lccm"]
            if lccm.window_idx != start_windows.get(vid, lccm.window_idx):
                if self._arrays is not None:
                    psi_acc = data["psi_acc"]
                    psi, EQ = close_window(psi_acc)
                    self._arrays.vertices["psi"][vid] = psi
                    psi_acc.fill(0)
                    self._arrays.vertices["EQ"][vid] = EQ
                    p_v = data["p_v"]
                    entropy = (
                        float(-(p_v * np.log2(p_v + 1e-12)).sum()) if len(p_v) else 0.0
                    )
                    conf = float(self._arrays.vertices["conf"][vid])
                    E_theta = lccm.a * (1.0 - entropy)
                    E_C = lccm.b * conf
                    p_v.fill(1.0 / len(p_v))
                    self._arrays.vertices["p"][vid] = p_v
                    self._arrays.vertices["bit"][vid] = 0
                    self._arrays.vertices["conf"][vid] = 0.0
                    data["bit_deque"].clear()
                    lccm.update_eq(EQ)
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
                        "E_theta": E_theta,
                        "E_C": E_C,
                        "v_id": vid,
                    },
                    metadata={"window_idx": lccm.window_idx},
                )

        return frame

    def snapshot_for_ui(self) -> dict:
        """Return a minimal snapshot for the GUI."""

        max_depth = 0
        for data in self._vertices.values():
            max_depth = max(max_depth, data["lccm"].depth)
        return {"depth": max_depth}

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
