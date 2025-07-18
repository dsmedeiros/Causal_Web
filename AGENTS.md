# Repository Guidelines

- **Formatting**: Run `black` on the `Causal_Web` package after modifying Python files.
- **Syntax check**: Execute `python -m compileall Causal_Web` to ensure files compile.
- **Docstrings**: Add or update docstrings when modifying or adding public functions or classes.
- **README**: Update `README.md` to reflect any user-facing code changes or new features.
- **Manual run**: The simulation GUI starts with `python -m Causal_Web.main`.
- **Dependencies**: Install `numpy` with `pip install numpy` before running tests.
- **Testing**: Execute `pytest` after installing dependencies to verify core behaviour.
- **GUI dependency**: Install `dearpygui` via `pip install dearpygui` to run the simulation dashboard.
- **Packaging**: Run `python bundle_run.py` after a simulation to archive the output logs.

