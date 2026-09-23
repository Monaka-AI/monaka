# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinx_bootstrap_theme
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

project = 'Monaka'
copyright = '2026, Hiroaki Ozaki (Monaka.AI)'
author = 'Hiroaki Ozaki (Monaka.AI)'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'ja'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'book'
#html_static_path = ['_static']

html_theme = 'pydata_sphinx_theme'
#html_theme = 'bootstrap'
#html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

html_theme_options = {
    'bootswatch_theme': 'simplex',
    'bootstrap_version': '3',
    "icon_links": [
         {
             "name": "GitHub",
             "url": "https://github.com/Monaka-AI/monaka/",  # required
             "icon": "fab fa-github",
             "type": "fontawesome",  # Default is fontawesome
         },
         {
             "name": "X",
             "url": "https://x.com/MonakaAI123",
             "icon": "fab fa-twitter",
         },
     ],
}