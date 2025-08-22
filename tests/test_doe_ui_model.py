import pytest

from ui_new.state.DOE import DOEModel


def test_doe_model_metrics(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    model = DOEModel()
    model.runLhs(samples=1)
    assert model.nodeCount == 1
    assert model.frontier == 0
    assert model.expansionRate == pytest.approx(1.0)
    assert 0 <= model.promotionRate <= 1
