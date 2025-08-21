import msgpack
import pytest


@pytest.fixture
def snapshot_delta_sample():
    """Sample snapshot delta message for serialization checks."""

    return {"type": "SnapshotDelta", "v": 1, "frame": 0}


def test_round_trip_snapshot_delta(snapshot_delta_sample):
    b = msgpack.packb(snapshot_delta_sample, use_bin_type=True)
    out = msgpack.unpackb(b, raw=False)
    assert out["type"] == "SnapshotDelta"
    assert out["v"] == 1
    assert isinstance(out["frame"], int)
