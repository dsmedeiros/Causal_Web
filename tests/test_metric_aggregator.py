import csv
from pathlib import Path

from Causal_Web.engine.logging.logger import MetricAggregator


def test_metric_aggregator_writes_tall_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    agg = MetricAggregator(csv_path)
    # frame 0 events
    agg.add(0, "A")
    agg.add(0, "B")
    agg.add(0, "A")
    agg.flush(0)
    # frame 1 events with new category
    agg.add(1, "B")
    agg.add(1, "C")
    agg.flush(1)

    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))

    assert rows == [
        {"frame": "0", "category": "A", "count": "2"},
        {"frame": "0", "category": "B", "count": "1"},
        {"frame": "1", "category": "B", "count": "1"},
        {"frame": "1", "category": "C", "count": "1"},
    ]
