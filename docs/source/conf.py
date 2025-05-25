# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
# 假设 docs 目录在项目根目录下
sys.path.insert(0, os.path.abspath('../..')) # Adjusted path assuming conf.py is in docs/source
# 在文件顶部附近添加
try:
    # Need to adjust import path if conf.py is in docs/source
    from pysmatch import __version__ as version
except ImportError:
    print("Warning: Could not import pysmatch to get version. Is it installed or in sys.path?")
    version = 'unknown' # 或者从 setup.py 读取

# 设置 project release
release = version
project = 'pysmatch'
copyright = '2025, Miao Hancheng'
author = 'Miao Hancheng'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # 从 docstrings 导入文档
    'sphinx.ext.napoleon',  # 支持 Google 和 NumPy 风格 docstrings
    'sphinx.ext.viewcode',  # 添加源码链接
    'sphinx.ext.githubpages', # 支持 GitHub Pages
    'myst_parser',         # 解析 Markdown 文件
    'nbsphinx',            # 处理 Jupyter Notebooks
    'sphinx_design',       # 提供 UI 组件，如按钮、网格
    'sphinx.ext.intersphinx', # Optional: Link to other projects' docs
    'sphinx.ext.todo',      # Optional: Support for todo notes
]

# Optional: Configure intersphinx mapping if you want to link to Python/Numpy docs etc.
# intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
#                        'numpy': ('https://numpy.org/doc/stable/', None),
#                        'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None)}

# Optional: Configure Napoleon settings if needed
# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True
myst_enable_extensions = [
    "amsmath",        # For math equations
    "colon_fence",    # Allows ``` directives for code blocks etc.
    "deflist",
    "dollarmath",     # For inline math <span class="math-inline">\.\.\.</span>
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",        # Automatically convert bare URLs to links
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",   # Allows defining substitutions
    "tasklist",       # Allows checkbox lists - [ ] and - [x]
    "table",          # <-- ESSENTIAL: Enable GFM tables
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store'] # Standard excludes

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index' # Default is usually correct

# docs/source/conf.py
nbsphinx_kernel_name = 'python3' # 或者你的 notebook 使用的内核名
nbsphinx_execute = 'never' # 或 'never' 如果不想执行代码
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further. For a list of options available for each theme, see the
# documentation.
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for nbsphinx ----------------------------------------------------
# Allow errors in notebooks to prevent build failures
nbsphinx_allow_errors = True
# Execute notebooks before processing: 'always', 'never', 'auto' (default)

# Optional: Add custom CSS for nbsphinx if needed
# html_css_files = [
#     'custom_nbsphinx.css',
# ]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',  # <--- 确保有这行
}



# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# ... 其他 Napoleon 设置 ...