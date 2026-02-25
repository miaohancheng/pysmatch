.. _installation:

Installation
============

Add instructions on how to install pysmatch here. You can include content from your README.

.. code-block:: bash

   pip install pysmatch

Or install from source:

.. code-block:: bash

   git clone https://github.com/mhcone/pysmatch.git
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
