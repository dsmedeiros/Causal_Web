from __future__ import annotations

"""Monte Carlo Tree Search optimiser for hyperparameters."""

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .api import Optimizer
from .priors import Prior


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
    weights to scalarise objective vectors for backpropagation.
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
            ``promote_threshold``, ``promote_quantile``, ``promote_window``
            and ``multi_objective``.
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
            if self.multi_objective:
                info["weights"] = None
            self._paths[key] = info
            out.append(full)
            self._suggestions += 1
        return out

    def observe(self, results: List[Dict[str, float]]) -> None:
        for res in results:
            cfg = res["config"]
            key = tuple(sorted(cfg.items()))
            info = self._paths.get(key, {})
            path = info.get("path", [])
            weights = info.get("weights")
            if "objectives" in res:
                objs = res["objectives"]
                vals = np.array([objs[k] for k in sorted(objs)], dtype=float)
                if self.multi_objective:
                    if weights is None:
                        weights = self.rng.dirichlet(np.ones(len(vals)))
                    reward = -float(np.dot(weights, vals))
                else:
                    reward = -float(vals[0])
                full_score = (
                    float(vals[0])
                    if not self.multi_objective
                    else float(np.dot(weights, vals))
                )
                self._paths.pop(key, None)
                for node in path:
                    node.N += 1
                    node.Q += (reward - node.Q) / node.N
                if key in self._proxy_cache:
                    self._proxy_full_pairs.append(
                        (self._proxy_cache.pop(key), full_score)
                    )
            elif "objectives_proxy" in res:
                objs = res["objectives_proxy"]
                vals = np.array([objs[k] for k in sorted(objs)], dtype=float)
                if self.multi_objective:
                    if weights is None:
                        weights = self.rng.dirichlet(np.ones(len(vals)))
                        info["weights"] = weights
                        self._paths[key] = info
                    score = float(np.dot(weights, vals))
                else:
                    score = float(vals[0])
                reward = -score
                for node in path:
                    node.N += 1
                    node.Q += (reward - node.Q) / node.N
                self._proxy_scores.append(score)
                self._proxy_cache[key] = score
                self._proxy_evals += 1
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
                if thresh is None or score <= float(thresh):
                    self._pending_full.append(cfg)
                    self._promotions += 1
            elif "fitness" in res or "fitness_full" in res:
                full_score = float(res.get("fitness", res.get("fitness_full", 0.0)))
                reward = -full_score
                self._paths.pop(key, None)
                for node in path:
                    node.N += 1
                    node.Q += (reward - node.Q) / node.N
                if key in self._proxy_cache:
                    self._proxy_full_pairs.append(
                        (self._proxy_cache.pop(key), full_score)
                    )
            elif "fitness_proxy" in res:
                score = float(res["fitness_proxy"])
                reward = -score
                for node in path:
                    node.N += 1
                    node.Q += (reward - node.Q) / node.N
                self._proxy_scores.append(score)
                self._proxy_cache[key] = score
                self._proxy_evals += 1
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
                if thresh is None or score <= float(thresh):
                    self._pending_full.append(cfg)
                    self._promotions += 1
            else:
                self._paths.pop(key, None)

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
            },
            "root": encode(self.root),
            "pending_full": self._pending_full,
            "proxy_scores": self._proxy_scores,
            "rng_state": self.rng.bit_generator.state,
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
        child = self._ttable.get(key)
        if child is None:
            child = Node(x, node.pending[1:])
            self._ttable[key] = child
            self._nodes += 1
            self._expansions += 1
        node.children.append(child)
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
            child = self._ttable.get(key)
            if child is None:
                child = Node(x, current.pending[1:])
                self._ttable[key] = child
                self._nodes += 1
                self._expansions += 1
            current.children.append(child)
            path.append(child)
            current = child
        return current.x_partial, path

    # ------------------------------------------------------------------
    # metrics
    def metrics(self) -> Dict[str, float]:
        """Return collected search metrics."""

        expansion_rate = self._expansions / max(1, self._suggestions)
        promotion_rate = self._promotions / max(1, self._proxy_evals)
        avg_depth = self._rollout_depth_total / max(1, self._suggestions)
        spearman = self._spearman()
        return {
            "expansion_rate": expansion_rate,
            "promotion_rate": promotion_rate,
            "avg_rollout_depth": avg_depth,
            "spearman_proxy_full": spearman,
        }

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
