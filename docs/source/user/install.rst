.. _install:

Installation
============

This part of the documentation covers the requirements and installation of sail-on client.

Requirements
------------

1. `Python 3.7`_
2. `Pipenv`_
3. `tinker-engine`_
4. `Script Config`_
5. `TA2 Agent`_
6. `Sail-On`_

Installation
------------

Installation with Pipenv ( Recommended )
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone the repositories associated with different components in a working directory::

      $ git clone https://gitlab.kitware.com/darpa_learn/tinker-engine.git
      $ git clone https://gitlab.kitware.com/darpa-sail-on/sail-on.git
      $ git clone https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector.git
      $ git clone https://gitlab.kitware.com/darpa-sail-on/sail-on-client.git

   This would create tinker-engine, sail-on, evm_based_novelty_detector and sail-on-client
   directories in your working directory
2. Install the different components in a virtual environment::

    $ cd ../sail-on-client
    $ pipenv install
    $ pipenv shell

Installation without Pipfile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The installation requires cloning and installing multiple repositories. Thus after
following the instructions for a repository, please move back to your
your working directory.

Install Tinker Engine
"""""""""""""""""""""

1. Clone the `tinker-engine`_ repository::

     $ git clone https://gitlab.kitware.com/darpa_learn/tinker-engine.git

   This would create a directory called tinker-engine in your working directory

2. Install the dependencies of the tinker-engine in a virtual environment::

     $ pipenv install
     $ pipenv shell

   This would create a virtual environment and activate the environment. Please
   use this virtual environment for installing all other repositories.

3. Install the tinker-engine in the virtual environment::

     $ pip install -e .

Install Sail-On Server
""""""""""""""""""""""

1. Clone the `Sail-On`_ repository::

     $ git clone https://gitlab.kitware.com/darpa-sail-on/sail-on.git

   This would create a directory called sail-on in your working directory

2. Go into the sail-on directory and install the dependencies for the server using::

     $ cd sail-on
     $ pip install -r requirements.txt

3. Install the server::

     $ pip install -e .

Install TA2 Agent
"""""""""""""""""

1. Clone `TA2 agent`_ repository::

     $ git clone https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector.git

   This would create a directory called evm_based_novelty_detector in your working directory

2. Go into the directory for TA2 agent and install the dependencies for the server using::

     $ cd evm_based_novelty_detector
     $ pip install -r requirements.txt
     $ pip install -e timm

3. Install agent using::

     $ pip install -e .

Install Sail-On Client
""""""""""""""""""""""

1. Clone the `sail-on-client`_ repository::

     $ git clone https://gitlab.kitware.com/darpa-sail-on/sail-on-client.git

   This would create a directory called sail-on-client in your working directory

2. Go into the sail-on-client directory and install the dependencies for the client using::

     $ cd sail-on-client
     $ pip install -r requirements.txt


3. Install the client using::

     $ pip install -e .


.. Appendix 1: Links

.. _Python 3.7: https://www.python.org/downloads/release/python-370/
.. _Pipenv: https://pipenv.pypa.io/en/latest/
.. _tinker-engine: https://gitlab.kitware.com/darpa_learn/tinker-engine
.. _Script Config: https://pypi.org/project/scriptconfig
.. _TA2 Agent: https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector
.. _Sail-On: https://gitlab.kitware.com/darpa-sail-on/sail-on
.. _sail-on-client: https://gitlab.kitware.com/darpa-sail-on/sail-on-client


