import os

# Use offscreen platform to avoid GUI requirement
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from Causal_Web.gui_legacy.canvas_widget import CanvasWidget
from Causal_Web.graph.model import GraphModel


def test_load_model_recreates_hud_item():
    app = QApplication.instance() or QApplication([])
    canvas = CanvasWidget(editable=False)
    canvas.load_model(GraphModel.blank())
    assert canvas._hud_item is not None
    assert canvas._hud_item.scene() is canvas.scene()
    # Calling again should recreate the HUD item without errors
    canvas.load_model(GraphModel.blank())
    assert canvas._hud_item.scene() is canvas.scene()
    # Remove gui modules to avoid side effects on other tests
    import sys

    for name in [m for m in list(sys.modules) if m.startswith("Causal_Web.gui_legacy")]:
        del sys.modules[name]
