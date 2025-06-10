# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))

project = 'pyrootmemo'
copyright = '2025, Anil Yildiz, Gerrit J. Meijer'
author = 'Anil Yildiz, Gerrit J. Meijer'

release = '0.1.0'  # Update this to your package version
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', # To parse NumPy/Google style docstrings
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',

    # You can add more, e.g. 'sphinx.ext.viewcode', 'sphinx.ext.intersphinx', etc.
]

autoclass_content = 'init'
autodoc_member_order = 'bysource'
add_module_names = False

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'author_year'



html_theme_options = {
    'display_version': True,
    'style_nav_header_background': '#2980B9',  # Example: customizing the header color
    # more theme options here...
}

templates_path = ['_templates']
exclude_patterns = []

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
