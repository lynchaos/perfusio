# Configuration file for the Sphinx documentation builder.
#
# Run ``make html`` from docs/ or use the docs.yml GitHub Actions workflow.

from __future__ import annotations

import perfusio

project = "perfusio"
author = "Kemal Yaylali"
copyright = "2026, Kemal Yaylali"
version = perfusio.__version__
release = version

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# ── MyST Parser ───────────────────────────────────────────────────────────────
myst_enable_extensions = ["amsmath", "dollarmath", "colon_fence"]

# ── HTML ──────────────────────────────────────────────────────────────────────
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"{project} {release}"

# ── NBsphinx ──────────────────────────────────────────────────────────────────
nbsphinx_execute = "never"  # notebooks pre-executed; outputs committed

# ── Source ────────────────────────────────────────────────────────────────────
templates_path = ["_templates"]
exclude_patterns = ["build", "**.ipynb_checkpoints"]
