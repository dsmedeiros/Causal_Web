from ui_new.state.Telemetry import TelemetryModel


def test_record_updates_histories():
    tm = TelemetryModel(max_points=2)
    tm.record({"events": 1.0}, {"inv": True}, depth=3)
    tm.record({"events": 2.0}, {"inv": False}, depth=4)
    assert tm.counters["events"] == [1.0, 2.0]
    assert tm.invariants["inv"] == [1.0, 0.0]
    assert tm.depth == 4
