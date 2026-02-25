.. _installation:

Installation
============

Add instructions on how to install pysmatch here. You can include content from your README.

.. code-block:: bash

   pip install pysmatch

If you see ``ValueError: numpy.dtype size changed``, reinstall NumPy and pandas
in one command, then restart your kernel/session:

.. code-block:: bash

   pip install --upgrade --force-reinstall "numpy>=1.26.4" "pandas>=2.1.4"

Or install from source:

.. code-block:: bash

   git clone https://github.com/miaohancheng/pysmatch.git
   cd pysmatch
   pip install .

Dependencies
------------
Core dependencies:
* Python 3.9+
* pandas
* numpy
* scipy
* statsmodels
* scikit-learn
* matplotlib
* imbalanced-learn
* seaborn

Optional dependencies:
* ``pip install "pysmatch[tree]"`` for CatBoost models
* ``pip install "pysmatch[tune]"`` for Optuna tuning
