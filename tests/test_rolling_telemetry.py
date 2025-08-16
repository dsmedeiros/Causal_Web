from telemetry import RollingTelemetry


def test_record_caps_length():
    rt = RollingTelemetry(max_points=3)
    for i in range(5):
        rt.record({"a": float(i)}, {"inv": i % 2 == 0})
    assert rt.get_counters()["a"] == [2.0, 3.0, 4.0]
    assert rt.get_invariants()["inv"] == [1.0, 0.0, 1.0]


def test_optional_arguments():
    rt = RollingTelemetry(max_points=2)
    rt.record({"x": 1.0})
    rt.record(invariants={"ok": True})
    assert rt.get_counters()["x"] == [1.0]
    assert rt.get_invariants()["ok"] == [1.0]
