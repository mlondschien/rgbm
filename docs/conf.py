project = "rgbm"
copyright = "2026, Malte Londschien"
author = "Malte Londschien"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "nbsphinx",
]

# Render the notebook from its committed outputs; do not re-execute on build
# (the dataset isn't available on RTD).
nbsphinx_execute = "never"

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
