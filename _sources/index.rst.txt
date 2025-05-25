=================
index
=================

**pysmatch**: It is a Python package providing a robust tool for propensity score matching in Python. This package fixes known bugs from the original project and introduces new features such as parallel computing and model selection, enhancing performance and flexibility.

.. container:: button-container

   .. button-link:: https://github.com/miaohancheng/pysmatch/
      :class: btn btn-primary btn-lg

      Goto GitHub Repository

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: English README
      :link: https://github.com/miaohancheng/pysmatch/blob/main/README.md

      View project description and usage instructions on GitHub.

   .. grid-item-card:: 中文 README (Chinese README)
      :link: https://github.com/miaohancheng/pysmatch/blob/main/README_CHINESE.md

      在 GitHub 上查看项目的中文描述和使用说明.

.. toctree::
   :maxdepth: 2
   :caption: 目录 (Contents):
   :hidden:

   readme_en          <-- 添加：链接到英文 README 页面
   readme_zh          <-- 添加：链接到中文 README 页面
   installation
   usage
   example            <-- 添加：链接到 Example Notebook 页面
   api                <-- 保持你现有的 API 页面链接
   contributing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: 核心功能 (Core Features):
   :hidden:

   api/matcher
   api/modeling
   api/matching
   api/visualization
   api/utils

.. only:: html

   .. rubric:: 快速导航 (Quick Navigation)

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

.. raw:: html


       <style>
          .button-container {
             margin-top: 1rem;
             margin-bottom: 2rem;
             text-align: center; /* Center the button */
          }
          /* Basic styling for the button - requires sphinx-design or similar */
          /* If not using sphinx-design, you might need custom CSS */
          .btn {
             display: inline-block;
             font-weight: 400;
             text-align: center;
             vertical-align: middle;
             user-select: none;
             background-color: transparent;
             border: 1px solid transparent;
             padding: .375rem .75rem;
             font-size: 1rem;
             line-height: 1.5;
             border-radius: .25rem;
             transition: color .15s ease-in-out,background-color .15s ease-in-out,border-color .15s ease-in-out,box-shadow .15s ease-in-out;
             text-decoration: none; /* Remove underline from link */
          }
          .btn-primary {
             color: #fff;
             background-color: #007bff;
             border-color: #007bff;
          }
          .btn-primary:hover {
             color: #fff;
             background-color: #0056b3;
             border-color: #004085;
             text-decoration: none;
          }
          .btn-lg {
             padding: .5rem 1rem;
             font-size: 1.25rem;
             line-height: 1.5;
             border-radius: .3rem;
          }
       </style>

