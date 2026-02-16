from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

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
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True
autosummary_generate_overwrite = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "furo"

# Intersphinx: must be (url, inventory) where inventory is None or a non-empty string.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Make the build strict, but silence known noisy cross-ref classes from examples/autogen.
suppress_warnings = [
    "autosummary.import_cycle",
    "ref.class",
    "ref.func",
    "ref.mod",
    "ref.ref",
]

# Sphinx-Gallery
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sphinx_gallery_conf = {
    "examples_dirs": [os.path.join(_ROOT, "examples")],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r".*\.py$",
    "ignore_pattern": r"^_",
}