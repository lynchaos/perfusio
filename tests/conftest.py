# conftest.py
"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pathlib
import sys
import sysconfig

import pytest


def _patch_pyro_stats() -> None:
    """Patch pyro/ops/stats.py for Python 3.13+ compile-time SyntaxError.

    pyro-ppl 1.9.1 contains an invalid escape sequence (``\\ge``) in a
    docstring in ``pyro/ops/stats.py``.  Python 3.13 promotes invalid escape
    sequences from a DeprecationWarning to a hard SyntaxError raised during
    bytecode compilation — before any import hook or filterwarnings filter
    can intercept it.  This function rewrites the file on disk before test
    collection triggers the transitive import chain through gpytorch→pyro.

    The replacement is idempotent: after the first patch the needle
    ``') \\ge E'`` (one backslash) is not present in the fixed text
    ``') \\\\ge E'`` (two backslashes), so subsequent runs are no-ops.
    """
    if sys.version_info < (3, 13):
        return
    site_packages = pathlib.Path(sysconfig.get_path("purelib"))
    stats_file = site_packages / "pyro" / "ops" / "stats.py"
    if not stats_file.exists():
        return
    text = stats_file.read_text(encoding="utf-8")
    # The bad sequence — one backslash before 'ge' inside a LaTeX math expr.
    # Using ') \ge E' as the needle guarantees idempotency: after patching,
    # ') \\ge E' does NOT contain ') \ge E' as a substring.
    needle = r") \ge E"
    if needle not in text:
        return  # already patched or different version
    fixed = text.replace(needle, r") \\ge E")
    stats_file.write_text(fixed, encoding="utf-8")
    # Remove any stale bytecode so Python recompiles from the patched source.
    for pyc in (stats_file.parent / "__pycache__").glob("stats.*.pyc"):
        pyc.unlink(missing_ok=True)


_patch_pyro_stats()


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="Run slow tests.")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="pass --runslow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
