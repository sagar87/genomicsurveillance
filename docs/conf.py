# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


# MOCK_MODULES = ['numpy', 'scipy', 'matplotlib', 'matplotlib.pyplot', 'tensorflow', 'h5py', 'scipy.spatial.distance', 'scipy.spatial', 'scipy.optimize', 'scipy.misc', 'scipy.special', 'scipy.sparse', 'scipy.linalg', 'scipy.stats', 'scipy.sparse.base', 'sklearn.utils.murmurhash', 'numpy.core', 'numpy.core.numeric', 'scipy.sparse.linalg']
MOCK_MODULES = ["tensorflow", "click", "h5py", "tqdm", "scipy", "scipy.interpolate"]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


# from unittest import mock

# # Mock modules because it fails to build in readthedocs
# MOCK_MODULES = ["numpy", "scipy", "scipy.interpolate", "matplotlib", "matplotlib.pyplot"]
# for mod_name in MOCK_MODULES:
#     sys.modules[mod_name] = mock.Mock()

# -- Project information -----------------------------------------------------

project = "genomicsurveillance"
copyright = "2021, Harald Vohringer, Moritz Gerstung"
author = "Harald Vohringer, Moritz Gerstung"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
