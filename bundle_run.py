"""Archive experiment runs into a timestamped tarball.

This helper collects the ``experiments`` directory—including run artifacts
and the run index—into ``bundle_<timestamp>.tar.gz``. It is intended to be
executed after simulations so results can be easily shared or attached to
bug reports.
"""

from __future__ import annotations

import tarfile
import time
from pathlib import Path


def main() -> None:
    """Create a tarball of the ``experiments`` directory."""

    root = Path("experiments")
    if not root.exists():
        print("No experiments directory found; nothing to bundle.")
        return

    ts = time.strftime("%Y%m%dT%H%M%SZ")
    bundle_path = Path(f"bundle_{ts}.tar.gz")
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(root, arcname=str(root))
    print(f"Bundled experiments to {bundle_path}")


if __name__ == "__main__":
    main()
