from __future__ import annotations

import time
from pathlib import Path

from experiments import results_registry as rr


def test_query_filters(tmp_path: Path) -> None:
    db = tmp_path / "results.db"
    conn = rr.connect(db)
    rr.insert(
        conn,
        run_id="run1",
        optimizer="mcts_h",
        promotion_rate=0.5,
        residual=0.01,
        proxy_full_corr=0.9,
        mcts_nodes_expanded=10,
        mcts_promotions=2,
        mcts_bins_created=3,
        mcts_frontier=4,
        mcts_params={"c_ucb": 1.0},
    )
    rr.insert(
        conn,
        run_id="run2",
        optimizer="other",
        promotion_rate=0.1,
        residual=0.05,
        proxy_full_corr=0.1,
    )
    t0 = time.perf_counter()
    rows = rr.query(
        conn,
        optimizer="mcts_h",
        promotion_min=0.3,
        residual_max=0.02,
        proxy_full_corr_min=0.5,
    )
    elapsed = time.perf_counter() - t0
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run1"
    # expect query to complete quickly (<0.2s)
    assert elapsed < 0.2
