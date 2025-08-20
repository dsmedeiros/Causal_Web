import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QGuiApplication
from PySide6.QtQuick import QSGNode, QSGGeometryNode

from ui_new.graph.GraphView import GraphView

# PySide builds without setInstanceCount still need to run tests
if not hasattr(QSGGeometryNode, "setInstanceCount"):
    QSGGeometryNode.setInstanceCount = lambda self, n: None


@pytest.mark.parametrize("n", [2000, 5000])
def test_graph_view_instancing(n):
    app = QGuiApplication.instance() or QGuiApplication([])

    gv = GraphView()
    nodes = [(float(i), float(i)) for i in range(n)]
    edges = [(i, (i + 1) % n) for i in range(n)]
    labels = [str(i) for i in range(n)]
    gv.set_graph(nodes, edges, labels=labels)

    root = QSGNode()
    gv._update_nodes(root)
    gv._update_edges(root)

    assert gv._node_geom is not None
    assert len(gv._node_material.offsets) == n

    assert gv._edge_geom is not None
    assert len(gv._edge_material.starts) == len(edges)

    gv.apply_delta({"closed_windows": [(i, 0) for i in range(n)]})
    gv._update_pulses(root)
    assert gv._pulse_geom is not None
    assert len(gv._pulse_material.offsets) == n

    gv.apply_delta({"node_positions": {300: (1.0, 2.0)}})
    gv._update_nodes(root)
    assert gv._node_offsets[300].x() == 1.0
    assert gv._node_offsets[300].y() == 2.0

    app.quit()
