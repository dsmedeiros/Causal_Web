import json
from Causal_Web.main import MainService, _apply_overrides
from Causal_Web.config import Config


def test_cli_flags_exposed_without_config(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text("{}")
    old_max = Config.max_ticks
    old_host = Config.database["host"]
    try:
        service = MainService(
            argv=[
                "--config",
                str(cfg),
                "--max_ticks",
                "5",
                "--database.host",
                "example.com",
            ]
        )
        args, data = service._parse_args()
        assert args.max_ticks == 5
        assert args.database_host == "example.com"
        _apply_overrides(args, data)
        assert Config.max_ticks == 5
        assert Config.database["host"] == "example.com"
    finally:
        Config.max_ticks = old_max
        Config.database["host"] = old_host
