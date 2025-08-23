from __future__ import annotations

"""Monte Carlo Tree Search optimiser for hyperparameters."""

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .api import Optimizer
from .priors import DiscretePrior, GaussianPrior, Prior


@dataclass
class Node:
    """Single node in the hyperparameter search tree."""

    x_partial: Dict[str, float]
    pending: List[str]
    N: int = 0
    Q: float = 0.0
    children: List["Node"] = field(default_factory=list)


class MCTS_H(Optimizer):
    """MCTS optimiser with progressive widening, caching and promotion.

    The optimiser samples hyperparameter configurations using UCT selection
    with progressive widening.  Partial configurations are cached in a simple
    transposition table allowing separate branches of the tree to share node
    statistics.  Candidates are first evaluated using a cheap proxy metric and
    can be promoted for a subsequent full evaluation when the proxy score
    meets a configurable threshold or falls within a promotion quantile.
    When ``multi_objective`` is enabled the optimiser draws random Dirichlet
    weights to scalarise objective vectors for backpropagation.  Supplying an
    ``hv_box`` in the configuration instead uses the expected hypervolume
    improvement within the given reference box as the node value, reverting to
    scalarisation when omitted.
    """

    def __init__(
        self,
        space: Sequence[str],
        priors: Dict[str, Prior] | None = None,
        cfg: Dict[str, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise the optimiser.

        Parameters
        ----------
        space:
            Ordered sequence of parameter names.
        priors:
            Optional mapping of parameter names to sampling priors.
        cfg:
            Optional configuration dictionary. Recognised keys include
            ``c_ucb``, ``alpha_pw``, ``k_pw``, ``max_nodes``,
            ``promote_threshold``, ``promote_quantile``, ``promote_window``,
            ``multi_objective`` and ``hv_box``.
        rng:
            Optional NumPy random generator used for deterministic sampling.
        """

        self.space = list(space)
        self.priors = priors or {}
        cfg = cfg or {}
        self.c_ucb = float(cfg.get("c_ucb", 0.7))
        self.alpha = float(cfg.get("alpha_pw", 0.4))
        self.k = float(cfg.get("k_pw", 1.0))
        self.max_nodes = int(cfg.get("max_nodes", 10000))
        self.promote_threshold = cfg.get("promote_threshold")
        if self.promote_threshold is None:
            self.promote_quantile = float(cfg.get("promote_quantile", 0.6))
        else:
            self.promote_quantile = cfg.get("promote_quantile")
        self.promote_window = int(cfg.get("promote_window", 0))
        self.multi_objective = bool(cfg.get("multi_objective", False))
        self.hv_box = cfg.get("hv_box")
        if self.hv_box is not None:
            self.hv_box = np.array(self.hv_box, dtype=float)
        self._pareto: List[np.ndarray] = []
        self._hv = 0.0
        self.rng = rng or np.random.default_rng(int(cfg.get("rng_seed", 0)))
        self.root = Node({}, self.space.copy())
        self._paths: Dict[Tuple[Tuple[str, float], ...], Dict[str, Any]] = {}
        self._pending_full: List[Dict[str, float]] = []
        self._suggest_full: set[Tuple[Tuple[str, float], ...]] = set()
        self._ttable: Dict[Tuple[Tuple[str, float], int], Node] = {
            (tuple(), 0): self.root
        }
        self._nodes = 1
        self._proxy_scores: List[float] = []
        # instrumentation
        self._expansions = 0
        self._suggestions = 0
        self._rollout_depth_total = 0
        self._proxy_evals = 0
        self._promotions = 0
        self._proxy_cache: Dict[Tuple[Tuple[str, float], ...], float] = {}
        self._proxy_full_pairs: List[Tuple[float, float]] = []
        self._frontier = 1
        self._prior_data: Dict[str, List[float]] = {}
        self._prior_bins: Dict[str, int] = {}
        self._bin_stats: Dict[str, Dict[float, List[float]]] = {}
        for k, prior in self.priors.items():
            if isinstance(prior, GaussianPrior):
                self._prior_data[k] = [prior.mu]
            elif isinstance(prior, DiscretePrior):
                self._prior_bins[k] = len(prior.values)
                self._bin_stats[k] = {v: [] for v in prior.values}
                self._prior_data[k] = list(prior.values)
            else:
                self._prior_data[k] = []
        for p in self.space:
            self._prior_data.setdefault(p, [])
        # generation tracking
        self._generation = 0
        self._promotion_history: List[float] = []
        self._best_history: List[float] = []

    # ------------------------------------------------------------------
    # public API
    def suggest(self, n: int) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for _ in range(n):
            if self._pending_full:
                cfg = self._pending_full.pop(0)
                key = tuple(sorted(cfg.items()))
                self._suggest_full.add(key)
                out.append(cfg)
                self._suggestions += 1
                continue
            node, path = self._select(self.root, [self.root])
            if node.pending:
                node = self._expand(node)
                path.append(node)
            full, path = self._rollout(node, path)
            self._rollout_depth_total += len(path) - 1
            key = tuple(sorted(full.items()))
            info: Dict[str, Any] = {"path": path}
            if self.multi_objective and self.hv_box is None:
                info["weights"] = None
            self._paths[key] = info
            out.append(full)
            self._suggestions += 1
        return out

    def observe(self, results: List[Dict[str, float]]) -> None:
        gen_proxy = 0
        gen_promote = 0
        gen_best: float | None = None
        for res in results:
            cfg = res["config"]
            key = tuple(sorted(cfg.items()))
            info = self._paths.get(key, {})
            path = info.get("path", [])
            weights = info.get("weights")
            if "objectives" in res:
                objs = res["objectives"]
                vals = np.array([objs[k] for k in sorted(objs)], dtype=float)
                if self.multi_objective and self.hv_box is not None:
                    reward = float(self._hv_improvement(vals))
                    full_score = -reward
                elif self.multi_objective:
                    if weights is None:
                        weights = self.rng.dirichlet(np.ones(len(vals)))
                    reward = -float(np.dot(weights, vals))
                    full_score = float(np.dot(weights, vals))
                else:
                    reward = -float(vals[0])
                    full_score = float(vals[0])
                gen_best = full_score if gen_best is None else min(gen_best, full_score)
                self._paths.pop(key, None)
                for node in path:
                    node.N += 1
                    node.Q += (reward - node.Q) / node.N
                if key in self._proxy_cache:
                    self._proxy_full_pairs.append(
                        (self._proxy_cache.pop(key), full_score)
                    )
                if self.multi_objective and self.hv_box is not None:
                    self._update_pareto(vals, reward)
                self._update_priors(cfg, full_score)
            elif "objectives_proxy" in res:
                objs = res["objectives_proxy"]
                vals = np.array([objs[k] for k in sorted(objs)], dtype=float)
                if self.multi_objective and self.hv_box is not None:
                    score = -float(self._hv_improvement(vals))
                elif self.multi_objective:
                    if weights is None:
                        weights = self.rng.dirichlet(np.ones(len(vals)))
                        info["weights"] = weights
                        self._paths[key] = info
                    score = float(np.dot(weights, vals))
                else:
                    score = float(vals[0])
                self._proxy_scores.append(score)
                self._proxy_cache[key] = score
                self._proxy_evals += 1
                gen_proxy += 1
                if (
                    self.promote_window > 0
                    and len(self._proxy_scores) > self.promote_window
                ):
                    del self._proxy_scores[0]
                thresh = self.promote_threshold
                if self.promote_quantile is not None and self._proxy_scores:
                    scores = (
                        self._proxy_scores[-self.promote_window :]
                        if self.promote_window > 0
                        else self._proxy_scores
                    )
                    thresh = float(np.quantile(scores, self.promote_quantile))
                promote = thresh is None or score <= float(thresh)
                reward = -score if promote else -1e9
                for node in path:
                    node.N += 1
                    node.Q += (reward - node.Q) / node.N
                if promote:
                    self._pending_full.append(cfg)
                    self._promotions += 1
                    gen_promote += 1
                else:
                    self._paths.pop(key, None)
            elif "fitness" in res or "fitness_full" in res:
                full_score = float(res.get("fitness", res.get("fitness_full", 0.0)))
                gen_best = full_score if gen_best is None else min(gen_best, full_score)
                reward = -full_score
                self._paths.pop(key, None)
                for node in path:
                    node.N += 1
                    node.Q += (reward - node.Q) / node.N
                if key in self._proxy_cache:
                    self._proxy_full_pairs.append(
                        (self._proxy_cache.pop(key), full_score)
                    )
                self._update_priors(cfg, full_score)
            elif "fitness_proxy" in res:
                score = float(res["fitness_proxy"])
                self._proxy_scores.append(score)
                self._proxy_cache[key] = score
                self._proxy_evals += 1
                gen_proxy += 1
                if (
                    self.promote_window > 0
                    and len(self._proxy_scores) > self.promote_window
                ):
                    del self._proxy_scores[0]
                thresh = self.promote_threshold
                if self.promote_quantile is not None and self._proxy_scores:
                    scores = (
                        self._proxy_scores[-self.promote_window :]
                        if self.promote_window > 0
                        else self._proxy_scores
                    )
                    thresh = float(np.quantile(scores, self.promote_quantile))
                promote = thresh is None or score <= float(thresh)
                reward = -score if promote else -1e9
                for node in path:
                    node.N += 1
                    node.Q += (reward - node.Q) / node.N
                if promote:
                    self._pending_full.append(cfg)
                    self._promotions += 1
                    gen_promote += 1
                else:
                    self._paths.pop(key, None)
            else:
                self._paths.pop(key, None)
        self._generation += 1
        rate = gen_promote / gen_proxy if gen_proxy else 0.0
        self._promotion_history.append(rate)
        prev_best = self._best_history[-1] if self._best_history else float("inf")
        best = min(prev_best, gen_best) if gen_best is not None else prev_best
        self._best_history.append(best)

    def done(self) -> bool:
        return self._nodes >= self.max_nodes

    # ------------------------------------------------------------------
    # persistence
    def state_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable snapshot of the optimiser state."""

        def encode(node: Node) -> Dict[str, object]:
            return {
                "x": node.x_partial,
                "pending": node.pending,
                "N": node.N,
                "Q": node.Q,
                "children": [encode(c) for c in node.children],
            }

        return {
            "space": self.space,
            "cfg": {
                "c_ucb": self.c_ucb,
                "alpha_pw": self.alpha,
                "k_pw": self.k,
                "max_nodes": self.max_nodes,
                "promote_threshold": self.promote_threshold,
                "promote_quantile": self.promote_quantile,
                "promote_window": self.promote_window,
                "multi_objective": self.multi_objective,
                "hv_box": self.hv_box.tolist() if self.hv_box is not None else None,
            },
            "root": encode(self.root),
            "pending_full": self._pending_full,
            "proxy_scores": self._proxy_scores,
            "rng_state": self.rng.bit_generator.state,
            "pareto": [p.tolist() for p in self._pareto],
            "hypervolume": self._hv,
        }

    def save(self, path: str | pathlib.Path) -> None:
        """Persist the optimiser state to ``path`` in JSON format."""

        pathlib.Path(path).write_text(json.dumps(self.state_dict()))

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        priors: Dict[str, Prior] | None = None,
    ) -> "MCTS_H":
        """Restore an optimiser from ``path``.

        Parameters
        ----------
        path:
            File previously written via :meth:`save`.
        priors:
            Optional priors to seed sampling. When omitted the optimiser will
            sample uniformly.
        """

        data = json.loads(pathlib.Path(path).read_text())
        cfg = data.get("cfg", {})
        rng = np.random.default_rng()
        rng.bit_generator.state = data["rng_state"]
        opt = cls(data["space"], priors or {}, cfg, rng)

        def decode(d: Dict[str, object]) -> Node:
            node = Node(dict(d["x"]), list(d["pending"]), d["N"], d["Q"])
            node.children = [decode(c) for c in d.get("children", [])]
            return node

        opt.root = decode(data["root"])
        opt._pending_full = [dict(x) for x in data.get("pending_full", [])]
        opt._proxy_scores = [float(s) for s in data.get("proxy_scores", [])]
        opt._ttable = {}
        opt._pareto = [np.array(p, dtype=float) for p in data.get("pareto", [])]
        opt._hv = float(data.get("hypervolume", 0.0))

        def rebuild(node: Node) -> None:
            key = (tuple(sorted(node.x_partial.items())), len(node.x_partial))
            opt._ttable[key] = node
            for ch in node.children:
                rebuild(ch)

        rebuild(opt.root)
        opt._nodes = len(opt._ttable)
        return opt

    # ------------------------------------------------------------------
    # MCTS internals
    def _select(self, node: Node, path: List[Node]) -> Tuple[Node, List[Node]]:
        current = node
        while current.pending:
            limit = max(1, int(self.k * (current.N**self.alpha)))
            if len(current.children) < limit:
                break
            best = None
            best_score = -float("inf")
            for child in current.children:
                uct = child.Q + self.c_ucb * math.sqrt(
                    math.log(current.N + 1) / (child.N + 1)
                )
                if uct > best_score:
                    best_score, best = uct, child
            if best is None:
                break
            path.append(best)
            current = best
        return current, path

    def _expand(self, node: Node) -> Node:
        param = node.pending[0]
        prior = self.priors.get(param)
        val = prior.sample(self.rng) if prior else float(self.rng.random())
        x = {**node.x_partial, param: val}
        key = (tuple(sorted(x.items())), len(x))
        parent_leaf = len(node.children) == 0
        child = self._ttable.get(key)
        if child is None:
            child = Node(x, node.pending[1:])
            self._ttable[key] = child
            self._nodes += 1
            self._expansions += 1
            if child.pending:
                self._frontier += 1
        node.children.append(child)
        if parent_leaf:
            self._frontier -= 1
        return child

    def _rollout(
        self, node: Node, path: List[Node]
    ) -> Tuple[Dict[str, float], List[Node]]:
        current = node
        while current.pending:
            param = current.pending[0]
            prior = self.priors.get(param)
            val = prior.sample(self.rng) if prior else float(self.rng.random())
            x = {**current.x_partial, param: val}
            key = (tuple(sorted(x.items())), len(x))
            parent_leaf = len(current.children) == 0
            child = self._ttable.get(key)
            if child is None:
                child = Node(x, current.pending[1:])
                self._ttable[key] = child
                self._nodes += 1
                self._expansions += 1
                if child.pending:
                    self._frontier += 1
            current.children.append(child)
            if parent_leaf:
                self._frontier -= 1
            path.append(child)
            current = child
        return current.x_partial, path

    # ------------------------------------------------------------------
    # hypervolume helpers
    def _hv_value(self, front: List[np.ndarray]) -> float:
        """Return dominated hypervolume for ``front`` within ``hv_box``.

        Parameters
        ----------
        front:
            Non-dominated list of objective vectors.

        Returns
        -------
        float
            Lebesgue measure of the dominated region clipped to ``hv_box``.

        Notes
        -----
        The implementation uses a simple recursive slicing algorithm and is
        intended for small Pareto fronts. Complexity grows exponentially with
        the number of objectives and points.
        """

        if not front or self.hv_box is None:
            return 0.0
        pts = np.array(front, dtype=float)
        if pts.shape[1] != len(self.hv_box):
            raise ValueError("hv_box dimension mismatch")

        lower = self.hv_box[:, 0]
        upper = self.hv_box[:, 1]
        # shift to origin for easier slicing
        pts = pts - lower
        ref = upper - lower

        pts = pts[np.all(pts <= ref, axis=1)]
        if pts.size == 0:
            return 0.0

        order = np.argsort(pts[:, 0])
        pts = pts[order]

        def hv_recursive(points: np.ndarray, ref_point: np.ndarray) -> float:
            if points.size == 0:
                return 0.0
            if points.shape[1] == 1:
                return float(ref_point[0] - np.min(points[:, 0]))
            volume = 0.0
            while points.size:
                x = points[-1, 0]
                width = ref_point[0] - x
                if width > 0:
                    volume += width * hv_recursive(points[:, 1:], ref_point[1:])
                ref_point = ref_point.copy()
                ref_point[0] = x
                points = points[points[:, 0] < x]
            return float(volume)

        return hv_recursive(pts, ref)

    def _hv_improvement(self, vals: np.ndarray) -> float:
        if self.hv_box is None:
            return 0.0
        if not np.all(vals <= self.hv_box[:, 1]) or not np.all(
            vals >= self.hv_box[:, 0]
        ):
            return 0.0
        front = []
        for p in self._pareto:
            if np.all(p <= vals) and np.any(p < vals):
                continue
            if np.all(vals <= p) and np.any(vals < p):
                return 0.0
            front.append(p)
        front.append(vals)
        hv_after = self._hv_value(front)
        return max(0.0, hv_after - self._hv)

    def _update_pareto(self, vals: np.ndarray, hv_impr: float) -> None:
        if hv_impr <= 0:
            return
        new_front = []
        for p in self._pareto:
            if np.all(p <= vals) and np.any(p < vals):
                continue
            new_front.append(p)
        new_front.append(vals)
        self._pareto = new_front
        self._hv += hv_impr

    # ------------------------------------------------------------------
    # metrics
    def metrics(self) -> Dict[str, float]:
        """Return collected search metrics.

        The dictionary includes expansion and promotion rates, average rollout
        depth, Spearman correlation between proxy and full evaluations, the
        current frontier size, and per-generation promotion and best-so-far
        improvements.
        """

        expansion_rate = self._expansions / max(1, self._suggestions)
        promotion_rate = self._promotions / max(1, self._proxy_evals)
        avg_depth = self._rollout_depth_total / max(1, self._suggestions)
        spearman = self._spearman()
        gen_rate = (
            self._promotion_history[-1] if self._promotion_history else float("nan")
        )
        rate_delta = (
            self._promotion_history[-1] - self._promotion_history[-2]
            if len(self._promotion_history) > 1
            else float("nan")
        )
        best = self._best_history[-1] if self._best_history else float("inf")
        if (
            len(self._best_history) > 1
            and math.isfinite(self._best_history[-2])
            and math.isfinite(best)
        ):
            best_improve = self._best_history[-2] - best
        else:
            best_improve = 0.0
        out = {
            "expansion_rate": expansion_rate,
            "promotion_rate": promotion_rate,
            "avg_rollout_depth": avg_depth,
            "spearman_proxy_full": spearman,
            "frontier": float(self._frontier),
            "promotion_rate_gen": gen_rate,
            "promotion_rate_improvement": rate_delta,
            "best_so_far": best,
            "best_so_far_improvement": best_improve,
        }
        if self.hv_box is not None:
            out["hypervolume"] = self._hv
        return out

    def _spearman(self) -> float:
        if len(self._proxy_full_pairs) < 2:
            return float("nan")
        proxy, full = zip(*self._proxy_full_pairs)
        rx = self._rank(proxy)
        ry = self._rank(full)
        if np.all(rx == rx[0]) or np.all(ry == ry[0]):
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])

    @staticmethod
    def _rank(vals: Sequence[float]) -> np.ndarray:
        arr = np.array(vals)
        order = np.argsort(arr)
        ranks = np.empty(len(arr), dtype=float)
        i = 0
        while i < len(arr):
            j = i
            while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
                j += 1
            rank = 0.5 * (i + j) + 1
            ranks[order[i : j + 1]] = rank
            i = j + 1
        return ranks

    # ------------------------------------------------------------------
    # adaptive priors
    def _update_priors(self, cfg: Dict[str, float], reward: float) -> None:
        for param, val in cfg.items():
            prior = self.priors.get(param)
            if prior is None:
                continue
            self._prior_data[param].append(val)
            if isinstance(prior, GaussianPrior):
                arr = np.array(self._prior_data[param], dtype=float)
                mu = float(np.median(arr))
                mad = float(np.median(np.abs(arr - mu)))
                sigma = float(mad * 1.4826) or 1.0
                self.priors[param] = GaussianPrior(mu, sigma)
            elif isinstance(prior, DiscretePrior):
                stats = self._bin_stats.setdefault(param, {})
                stats.setdefault(val, []).append(reward)
                bins = self._prior_bins.get(param, len(prior.values))
                variances = [np.var(r) for r in stats.values() if len(r) > 1]
                if variances:
                    median_var = float(np.median(variances))
                    for v, r in list(stats.items()):
                        if len(r) > 1 and np.var(r) > median_var:
                            bins += 1
                            break
                empty = [v for v, r in stats.items() if len(r) == 0]
                bins = max(1, bins - len(empty))
                arr = np.array(self._prior_data[param], dtype=float)
                qs = np.linspace(0, 1, bins + 2)[1:-1]
                centres = np.quantile(arr, qs)
                probs = np.full(len(centres), 1.0 / len(centres))
                self.priors[param] = DiscretePrior(
                    list(map(float, centres)), list(probs)
                )
                self._prior_bins[param] = bins
                self._bin_stats[param] = {v: [] for v in self.priors[param].values}
