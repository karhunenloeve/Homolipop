import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

html_theme = "furo"

project = "Homolipop"
author = "Luciano Melodia"
copyright = "2026, Luciano Melodia"
release = "1.0.0"