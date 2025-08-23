import pytest

pytest.importorskip("PySide6")
from ui_new.state.Telemetry import TelemetryModel
import pytest


def test_record_updates_histories():
    tm = TelemetryModel(max_points=2)
    tm.record({"events": 1.0}, {"inv": True}, depth=3)
    tm.record({"events": 2.0}, {"inv": False}, depth=4)
    assert tm.counters["events"] == [1.0, 2.0]
    assert tm.invariants["inv"] == [1.0, 0.0]
    assert tm.depth == 4
    assert tm.depthLabel == "depth"
    tm.record({"events": 3.0}, {"inv": True}, depth=5, label="frame")
    assert tm.depthLabel == "frame"
    ci = tm.counterIntervals["events"]
    assert pytest.approx(ci[0]) == 2.5
    assert ci[1] <= ci[0] <= ci[2]
