import pathlib
from experiments.artifacts import (
    TopKEntry,
    update_top_k,
    load_top_k,
    save_hall_of_fame,
    load_hall_of_fame,
    allocate_run_dir,
    persist_run,
)


def test_top_k_merge(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "top_k.json"
    e1 = TopKEntry("run1", 0.5, {}, {"g": 1.0}, {}, 1, "")
    update_top_k([e1], path, k=2)
    e2 = TopKEntry("run2", 0.8, {}, {"g": 2.0}, {}, 2, "")
    update_top_k([e2], path, k=2)
    data = load_top_k(path)
    assert data["rows"][0]["run_id"] == "run2"
    assert data["k"] == 2


def test_hall_of_fame_roundtrip(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "hof.json"
    entries = [
        {"gen": 0, "run_id": "a", "fitness": 0.9, "objectives": {}, "path": "runs/a"}
    ]
    save_hall_of_fame(entries, path)
    data = load_hall_of_fame(path)
    assert data["archive"][0]["path"] == "runs/a"


def test_allocate_run_dir_creates_unique_path(tmp_path: pathlib.Path) -> None:
    run_id, abs_path, rel_path = allocate_run_dir(tmp_path)
    assert abs_path == tmp_path / rel_path
    assert rel_path.startswith("runs/")
    persist_run({}, {}, abs_path)
    assert abs_path.exists()


def test_persist_run_copies_delta_log(tmp_path: pathlib.Path) -> None:
    from Causal_Web.config import Config

    Config.output_dir = str(tmp_path)
    log = tmp_path / "delta_log.jsonl"
    log.write_text("{}\n")
    dest = tmp_path / "run"
    persist_run({}, {}, dest)
    assert (dest / "delta_log.jsonl").read_text() == "{}\n"
