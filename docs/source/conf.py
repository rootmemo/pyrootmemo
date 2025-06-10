# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyrootmemo'
copyright = '2025, Anil Yildiz, Gerrit J. Meijer'
author = 'Anil Yildiz, Gerrit J. Meijer'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', # To parse NumPy/Google style docstrings
    'sphinx_rtd_theme', 
    # You can add more, e.g. 'sphinx.ext.viewcode', 'sphinx.ext.intersphinx', etc.
]

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
html_static_path = ['_static']
