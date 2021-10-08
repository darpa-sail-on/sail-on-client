.. _install:

Installation
============

This part of the documentation covers the requirements and installation of sail-on client.

Requirements
------------

1. `Python`_ >= 3.7
2. `Poetry`_ >= 1.1.0
3. `poetry-dynamic-versioning`_

Installation
------------

Setup Poetry
^^^^^^^^^^^^

1. Install Poetry following the instructions available in the `installation page`_. (https://python-poetry.org/docs/#installation)

2. Install poetry-dynamic-versioning using::

      pip install poetry-dyanmic-versioning

   .. note::
      This is required before the package is built since poetry does not support plugins yet.


Installation with Poetry
^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone the repositories associated with different components in a working directory::

      git clone https://github.com/tinker-engine/tinker-engine.git
      git clone https://github.com/darpa-sail-on/sailon_tinker_launcher.git
      git clone https://github.com/darpa-sail-on/Sail_On_Evaluate.git
      git clone https://github.com/darpa-sail-on/Sail-On-API.git
      git clone https://github.com/darpa-sail-on/sail-on-client.git

   This would create tinker-engine, sailon_tinker_launcher, Sail_On_Evaluate,
   Sail-On-API and sail-on-client directories in your working directory

2. Install the different components in a virtual environment::

      cd sail-on-client
      poetry install
      poetry run pip install ../tinker-engine ../sailon_tinker_launcher ../Sail-On-API/ ../Sail_On_Evaluate/
      poetry shell


.. Appendix 1: Links

.. _Python 3.7: https://www.python.org/downloads/release/python-370/
.. _installation page: https://python-poetry.org/docs/#installation https://pipenv.pypa.io/en/latest/
.. _poetry-dynamic-versioning: https://github.com/mtkennerly/poetry-dynamic-versioning
.. _Poetry: https://github.com/python-poetry/poetry
.. _tinker-engine: https://gitlab.kitware.com/darpa_learn/tinker-engine
.. _Sail-On: https://github.com/darpa-sail-on/Sail-On-API
.. _Sail_On_Evaluate: https://github.com/darpa-sail-on/Sail_On_Evaluate
.. _sailon_tinker_launcher: https://github.com/darpa-sail-on/sailon_tinker_launcher
.. _sail-on-client: https://github.com/darpa-sail-on/sail-on-client


