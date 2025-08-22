"""Ensure recorded golden logs remain distinct and consistent."""

from pathlib import Path

from tests.golden_utils import expected_from_log


def test_golden_logs_unique():
    """Verify golden logs expose distinct residual traces and length."""

    paths = sorted(Path("tests/goldens").glob("run*.jsonl"))
    assert paths, "no golden logs found"
    frames, residuals = zip(*(expected_from_log(str(p)) for p in paths))
    # each residual EWMA should be unique to guard against accidental duplicates
    assert len(set(residuals)) == len(residuals)
    # logs should share a common frame count and be reasonably long for coverage
    assert len(set(frames)) == 1
    assert frames[0] >= 100
