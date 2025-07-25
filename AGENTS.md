# Repository Guidelines

- **Architecture**: Ensure all code complies with 'Single Responsibility Principle'.
- **Code**: Ensure all functions are no longer than 50 lines of code.
- **Formatting**: Run `black` on the `Causal_Web` package after modifying Python files.
- **Syntax check**: Execute `python -m compileall Causal_Web` to ensure files compile.
- **Docstrings**: Add or update docstrings when modifying or adding public functions or classes.
- **README**: Update `README.md` to reflect any user-facing code changes or new features.
- **Manual run**: The simulation GUI starts with `python -m Causal_Web.main`.
- **Dependencies**: Install `numpy`, `networkx` and `pytest` with `pip install numpy networkx pytest` before running tests.
- **Testing**: Execute `pytest` after installing dependencies to verify core behaviour.
- **Packaging**: Run `python bundle_run.py` after a simulation to archive the output logs.

