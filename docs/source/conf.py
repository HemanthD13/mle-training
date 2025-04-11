import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))  # Ensure Sphinx finds the package

# Project Information
project = "MLE Training"
author = "Hemanth D"
release = "0.1"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Supports NumPy docstring format
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc.typehints",
]

# Theme
html_theme = "sphinx_rtd_theme"

# Napoleon settings for NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
