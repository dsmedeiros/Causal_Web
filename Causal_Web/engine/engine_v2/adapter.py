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
import random

import numpy as np

from ..logging.logger import log_record
from .lccm import LCCM
from .scheduler import DepthScheduler
from .state import Packet, TelemetryFrame
from .loader import GraphArrays, load_graph_arrays
from .rho_delay import update_rho_delay
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
        eps_seed = eps_cfg.pop("seed", Config.run_seed)
        self._epairs = EPairs(seed=eps_seed, **eps_cfg)
        bell_seed = Config.bell.get("seed", Config.run_seed)
        self._bell = BellHelpers(seed=bell_seed)

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

        while self._scheduler and events < limit:
            self._epairs.decay_all()
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
            if edges and 0 <= edge_id < len(edges.get("alpha", [])):
                edge_params.update(
                    {
                        "alpha": float(edges["alpha"][edge_id]),
                        "phi": float(edges["phi"][edge_id]),
                        "A": float(edges["A"][edge_id]),
                        "U": edges["U"][edge_id],
                    }
                )

            packet_list = [packet_data]
            edge_list = [edge_params]
            pkt_list = [pkt]
            window_idx = lccm.window_idx
            requeue: list[tuple[int, int, int, Packet]] = []
            while self._scheduler and events + len(pkt_list) < limit:
                d2, dst2, edge2, pkt2 = self._scheduler.pop()
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
                pd2.setdefault("depth_arr", d2)
                if "psi" not in pd2:
                    pd2["psi"] = np.zeros_like(vertex["psi_acc"])
                if "p" not in pd2:
                    pd2["p"] = np.zeros_like(vertex["p_v"])
                if "bit" not in pd2:
                    pd2["bit"] = 0
                ep2: Dict[str, Any] = {
                    "alpha": 1.0,
                    "phi": 0.0,
                    "A": 0.0,
                    "U": np.eye(dim, dtype=np.complex64),
                }
                if edges and 0 <= edge2 < len(edges.get("alpha", [])):
                    ep2.update(
                        {
                            "alpha": float(edges["alpha"][edge2]),
                            "phi": float(edges["phi"][edge2]),
                            "A": float(edges["A"][edge2]),
                            "U": edges["U"][edge2],
                        }
                    )
                packet_list.append(pd2)
                edge_list.append(ep2)
                pkt_list.append(pkt2)
            for item in requeue:
                self._scheduler.push(*item)

            if len(pkt_list) > 1:
                psi_list: list[Any] = []
                p_list: list[Any] = []
                bit_list: list[Any] = []
                depth_list: list[int] = []
                for pd in packet_list:
                    psi_list.append(pd["psi"])
                    p_list.append(pd["p"])
                    bit_list.append(pd["bit"])
                    depth_list.append(pd["depth_arr"])
                packets_struct = {
                    "psi": psi_list,
                    "p": p_list,
                    "bit": bit_list,
                    "depth_arr": depth_list,
                }
                alpha_list: list[float] = []
                phi_list: list[float] = []
                A_list: list[float] = []
                U_list: list[Any] = []
                for ep in edge_list:
                    alpha_list.append(ep["alpha"])
                    phi_list.append(ep["phi"])
                    A_list.append(ep["A"])
                    U_list.append(ep["U"])
                edges_struct = {
                    "alpha": alpha_list,
                    "phi": phi_list,
                    "A": A_list,
                    "U": U_list,
                }
                depth_v, psi_acc, p_v, (bit, conf), intensity = deliver_packets_batch(
                    lccm.depth,
                    vertex["psi_acc"],
                    vertex["p_v"],
                    vertex["bit_deque"],
                    packets_struct,
                    edges_struct,
                    lccm.layer,
                )
            else:
                depth_v, psi_acc, p_v, (bit, conf), intensity = deliver_packet(
                    lccm.depth,
                    vertex["psi_acc"],
                    vertex["p_v"],
                    vertex["bit_deque"],
                    packet_data,
                    edge_params,
                    lccm.layer,
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
            packets.extend(pkt_list)

            ancestry_arr = (
                self._arrays.vertices["ancestry"][dst]
                if self._arrays is not None
                else np.zeros(4, dtype=np.int32)
            )
            m_arr = (
                self._arrays.vertices["m"][dst]
                if self._arrays is not None
                else np.zeros(3, dtype=float)
            )

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

            edges = self._arrays.edges if self._arrays else {}
            if lccm.layer == "Q" and self._arrays is not None:
                h_val = int.from_bytes(ancestry_arr.tobytes(), "little")
                edge_ids = self._edges_by_src.get(dst, [])
                theta = float(np.angle(self._arrays.vertices["psi"][dst][0]))
                self._epairs.carry(dst, depth_arr, edge_ids, edges)
                self._epairs.emit(dst, h_val, theta, depth_arr, edge_ids, edges)

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
                    "lambda_u": packet_data.get("lambda_u"),
                    "zeta": packet_data.get("zeta"),
                    "ancestry": packet_data.get("ancestry"),
                    "m": packet_data.get("m"),
                }
                new_pkt = Packet(
                    src=dst, dst=int(edges["dst"][edge_idx]), payload=payload
                )
                edge_logs += 1
                rate = Config.logging.get("sample_edge_rate", 0.0)
                if rate > 0.0 and self._rng.random() < rate:
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

            for other in list(self._epairs.adjacency.get(dst, [])):
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
                rate = Config.logging.get("sample_edge_rate", 0.0)
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
                    psi, EQ = close_window(psi_acc)
                    self._arrays.vertices["psi"][vid] = psi
                    psi_acc.fill(0)
                    self._arrays.vertices["EQ"][vid] = EQ
                    p_v = data["p_v"]
                    H_pv = (
                        float(-(p_v * np.log2(p_v + 1e-12)).sum()) if p_v.size else 0.0
                    )
                    conf = float(self._arrays.vertices["conf"][vid])
                    E_theta = lccm.a * (1.0 - H_pv)
                    E_C = lccm.b * conf
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
                        "H_p": H_pv,
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
