import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))

project = "Homolipop"
author = "Luciano Melodia"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]
html_theme = "furo"