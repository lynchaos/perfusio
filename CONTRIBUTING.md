# Contributing to perfusio

Thank you for your interest in contributing!

## Ways to contribute

- **Bug reports** — Open an issue with a minimal reproducible example.
- **Feature requests** — Open an issue describing the use case.
- **Pull requests** — See the workflow below.
- **Cell-line datasets** — Share anonymised calibration data via the
  Discussions tab.

## Development setup

```bash
git clone https://github.com/lynchaos/perfusio.git
cd perfusio
pip install -e ".[dev]"
pre-commit install
```

## Running tests

```bash
pytest                        # fast tests (excludes @slow)
pytest --runslow              # include 30-day integration test
pytest --cov=perfusio --cov-report=term-missing
```

## Code style

- **Ruff** for linting and formatting (`ruff check . --fix && ruff format .`).
- **mypy --strict** for type checking (`mypy src/`).
- All new public functions must have NumPy-style docstrings.
- New modules must include `from __future__ import annotations`.

## Pull request checklist

- [ ] Tests pass (`pytest`).
- [ ] Coverage does not decrease below 92% lines / 85% branch.
- [ ] `mypy` reports zero errors.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`.
- [ ] Docstrings updated.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`.

## Licence

By contributing you agree that your contributions will be licensed under
the Apache-2.0 licence.
