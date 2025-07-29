# Repository Guidelines

- **Architecture**: Ensure all code complies with 'Single Responsibility Principle'.
- **Code**: Ensure all functions are concise and focused on a single responsibility. Exceptions may be made for functions such as configuration, initialization, or those handling complex logic, provided they remain clear and maintainable.
- **Formatting**: Run `black` on the `Causal_Web` package after modifying Python files.
- **Syntax check**: Execute `python -m compileall Causal_Web` to ensure files compile.
- **Docstrings**: Add or update docstrings when modifying or adding public functions or classes.
- **README**: Update `README.md` to reflect any user-facing code changes or new features.
- **Manual run**: The simulation GUI starts with `python -m Causal_Web.main`.
- **Dependencies**: Install `numpy`, `networkx`, `pytest` and `pydantic` with `pip install numpy networkx pytest pydantic` before running tests.
- **Testing**: Execute `pytest` after installing dependencies to verify core behaviour.
- **Packaging**: Run `python bundle_run.py` after a simulation to archive the output logs.

