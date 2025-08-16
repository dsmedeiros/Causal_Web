import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtQuick import QSGNode

from ui_new.graph.GraphView import GraphView


def test_closed_window_pulses_decay():
    app = QApplication.instance() or QApplication([])
    view = GraphView()
    view.set_graph([(0.0, 0.0)], [], labels=["a"], colors=["white"], flags=[True])
    view.apply_delta({"closed_windows": [("0", 0)]})
    assert 0 in view._pulses
    root = QSGNode()
    for _ in range(view._pulse_duration):
        view._update_pulses(root)
    assert 0 not in view._pulses
