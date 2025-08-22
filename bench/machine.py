"""Utilities for collecting machine metadata."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any, Dict


def machine_info() -> Dict[str, Any]:
    """Return basic hardware and OS details for reproducibility.

    Information includes platform string, Python version, CPU model, core count,
    memory size in gigabytes and available GPU information when ``nvidia-smi`` is
    present on the ``PATH``.
    """

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
        "cores": os.cpu_count(),
    }
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        info["memory_gb"] = round(pages * page_size / (1024**3), 2)
    except (AttributeError, ValueError, OSError):
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        gpus = []
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            name, mem = [x.strip() for x in line.split(",", 1)]
            gpus.append({"name": name, "memory": mem})
        if gpus:
            info["gpus"] = gpus
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return info
