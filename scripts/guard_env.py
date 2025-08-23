import sys, pkgutil, ctypes.util, numpy as np

bad_qt = [m for m in ("PyQt5", "PyQt6") if pkgutil.find_loader(m)]
if bad_qt:
    print(f"ERROR: Unexpected Qt bindings present: {bad_qt}", file=sys.stderr)
    sys.exit(1)
ver = tuple(map(int, np.__version__.split(".")[:2]))
if ver >= (2, 0):
    print(
        f"ERROR: NumPy {np.__version__} detected; require < 2.0 for now.",
        file=sys.stderr,
    )
    sys.exit(1)
if sys.platform.startswith("linux"):
    missing = [
        lib
        for lib in ("GL", "EGL", "xkbcommon")
        if ctypes.util.find_library(lib) is None
    ]
    if missing:
        print(
            f"ERROR: Missing system libraries: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)
print("env-ok")
