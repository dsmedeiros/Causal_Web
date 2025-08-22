# Contributing

## Tooling
- Install hooks with `pre-commit install`.
- Format and lint with `black`, `ruff`, and type-check with `mypy`.

## Commit Style
Use [Angular](https://angular.io/guide/styleguide#commit-message-header) conventional commit messages.

## Versioning
Releases follow semantic versioning. Increment the major version for incompatible API changes, the minor version for new functionality, and the patch version for bug fixes.

## Tests
Run unit tests and golden replays locally before submitting a change:

```bash
pytest -q
pytest tests/test_replay_golden.py -q
```

The golden replay test uses logs under `tests/goldens/` to verify deterministic playback.
