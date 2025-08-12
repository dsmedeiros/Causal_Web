import math

from Causal_Web.geometry.curvature import CurvatureLogger, forman_curvature


def test_forman_curvature_basic():
    f = forman_curvature(4.0, deg_u=3, deg_v=2)
    assert math.isclose(f, (1 / 3 + 1 / 2) / 4)


def test_curvature_logger_stats():
    logger = CurvatureLogger()
    logger.log_edge("A", 0.2)
    logger.log_edge("A", 0.3)
    logger.log_edge("B", 0.1)
    stats = logger.window_close()
    assert math.isclose(stats["A"]["mean"], 0.25)
    assert math.isclose(stats["A"]["var"], 0.0025)
    assert math.isclose(stats["B"]["mean"], 0.1)
    assert math.isclose(stats["B"]["var"], 0.0)
