# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src/jlinops'))
print(sys.path)
# sys.path.append(os.path.abspath('/sphinxext'))

project = 'jlinops'
copyright = '2023, Jonathan Lindbloom'
author = 'Jonathan Lindbloom'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ 'sphinx.ext.autodoc',
    #'sphinx.ext.mathjax', 
    'sphinx.ext.autosectionlabel',
    'sphinx_panels',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'nbsphinx',
    ]

autosummary_generate = True
autosummary_imported_members = True
autoclass_content = "both"

autodoc_type_aliases = {
    'Iterable': 'Iterable',
    'ArrayLike': 'ArrayLike',
    "DTypeLike": "DTypeLike",
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_title = 'jlinops'



# Make wider
# html_theme_options = {'body_min_width': '90%'}

## CSS
# html_css_files = [
#     'custom.css',
# ]


# autodoc_type_aliases = {
#     'Iterable': 'Iterable',
#     'ArrayLike': 'ArrayLike'
# }