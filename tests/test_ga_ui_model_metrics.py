import os
import pytest
from PySide6.QtCore import QCoreApplication

from ui_new.state.GA import GAModel


def test_ga_model_metrics(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "experiments").mkdir()
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    model = GAModel()
    model.maxGenerations = 1
    model.step()
    assert model.nodeCount == model.populationSize
    assert model.frontier == 0
    assert 0 <= model.expansionRate <= 1
    assert 0 <= model.promotionRate <= 1
