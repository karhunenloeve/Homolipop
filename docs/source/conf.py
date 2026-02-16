from __future__ import annotations

import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parents[1]
SRC = ROOT / "src"

# Make local packages importable for autodoc and sphinx-gallery.
sys.path.insert(0, str(SRC))

project = "Homolipop"
author = "Luciano Melodia"
release = "1.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "sphinx_copybutton",
]

html_theme = "furo"

# Keep module pages lean. Control what is shown via explicit directives.
autodoc_default_options = {
    "show-inheritance": True,
}

# Crucial: avoid type-hint crossref spam like R, Simplex, np.ndarray, ...
autodoc_typehints = "none"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx: second entry must be a non-empty string URL or None. Never {}.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Do not turn missing xrefs into warnings.
nitpicky = False

autosummary_generate = True
autosummary_generate_overwrite = True

sphinx_gallery_conf = {
    "examples_dirs": [str(ROOT / "examples")],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r".*\.py$",
    "ignore_pattern": r"^_",
    "download_all_examples": False,
    "run_stale_examples": True,
}