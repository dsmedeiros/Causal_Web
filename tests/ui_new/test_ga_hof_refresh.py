import json
import os

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

from ui_new.state import GAModel


def test_ga_panel_refreshes_on_hof_update(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    hof_path = exp_dir / "hall_of_fame.json"
    hof_path.write_text(json.dumps({"archive": []}))
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])

    model = GAModel()
    updated = []

    def on_changed() -> None:
        updated.append(True)
        loop.quit()

    model.hallOfFameChanged.connect(on_changed)
    hof_path.write_text(json.dumps({"archive": [{"origin": "mcts"}]}))

    loop = QEventLoop()
    QTimer.singleShot(500, loop.quit)
    loop.exec()

    assert updated, "hallOfFameChanged not emitted"
    assert model.hallOfFame == [{"origin": "mcts"}]
