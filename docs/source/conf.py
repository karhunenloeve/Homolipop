# pylint: disable=invalid-name,missing-module-docstring
import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))

project = "Homolipop"
author = "Luciano Melodia"
release = "1.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
]

autosummary_generate = True
autosummary_generate_overwrite = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
html_theme = "furo"