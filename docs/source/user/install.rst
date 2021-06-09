.. _install:

Installation
============

This part of the documentation covers the requirements and installation of sail-on client.

Requirements
------------

1. `Python 3.7`_
2. `Pipenv`_

Installation
------------

Installation with Pipenv ( Recommended )
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone the repositories associated with different components in a working directory::

      $ git clone https://github.com/tinker-engine/tinker-engine.git
      $ git clone https://github.com/darpa-sail-on/sailon_tinker_launcher.git
      $ git clone https://github.com/darpa-sail-on/Sail_On_Evaluate.git
      $ git clone https://github.com/darpa-sail-on/Sail-On-API.git
      $ git clone https://github.com/darpa-sail-on/sail-on-client.git

   This would create tinker-engine, sailon_tinker_launcher, Sail_On_Evaluate,
   Sail-On-API and sail-on-client directories in your working directory

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

      $ git clone https://github.com/tinker-engine/tinker-engine.git

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

      $ git clone https://github.com/darpa-sail-on/Sail-On-API.git

   This would create a directory called sail-on in your working directory

2. Go into the sail-on directory and install the dependencies for the server using::

     $ cd Sail-On-API
     $ pip install -r requirements.txt

3. Install the server::

     $ pip install -e .

Install Sail_On_Evaluate
""""""""""""""""""""""""

1. Clone `Sail_On_Evaluate`_ repository::

      $ git clone https://github.com/darpa-sail-on/Sail_On_Evaluate.git

   This would create a directory called Sail_On_Evaluate in your working directory

2. Go into the directory for metric code and install the dependencies using::

     $ cd Sail_On_Evaluate
     $ pip install -r requirements.txt

3. Install metric using::

     $ pip install -e .

Install sailon_tinker_launcher
""""""""""""""""""""""""""""""

1. Clone `sailon_tinker_launcher`_ repository::

      $ git clone https://github.com/darpa-sail-on/sailon_tinker_launcher.git

   This would create a directory called sailon_tinker_launcher in your working directory

2. Go into the directory for the launcher and install the dependencies using::

     $ cd sailon_tinker_launcher
     $ pip install -r requirements.txt

3. Install sailon_tinker_launcher using::

     $ pip install -e .


Install Sail-On Client
""""""""""""""""""""""

1. Clone the `sail-on-client`_ repository::

      $ git clone https://github.com/darpa-sail-on/sail-on-client.git

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
.. _Sail-On: https://github.com/darpa-sail-on/Sail-On-API
.. _Sail_On_Evaluate: https://github.com/darpa-sail-on/Sail_On_Evaluate
.. _sailon_tinker_launcher: https://github.com/darpa-sail-on/sailon_tinker_launcher
.. _sail-on-client: https://github.com/darpa-sail-on/sail-on-client


