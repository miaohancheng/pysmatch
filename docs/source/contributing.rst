.. _contributing:

Contributing
============

Please read the repository `contributing guide <https://github.com/miaohancheng/pysmatch/blob/main/CONTRIBUTING.md>`_ before opening a pull request.

The local validation workflow is:

.. code-block:: bash

   python -m pip install -e ".[all,dev]"
   python scripts/sync_readme_to_docs.py --check
   MPLBACKEND=Agg python -m pytest

Matching and balance changes should include a focused test, document their
statistical assumptions, and keep the English and Chinese user guides aligned.
