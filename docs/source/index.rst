.. Sail-On Client documentation master file, created by
   sphinx-quickstart on Thu Sep  3 16:54:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Sail-On Client's documentation!
==========================================

Release v\ |version| (:ref:`Installation <install>`)

**Sail-On Client** is a library used for supporting evaluation of open world algorithms
in `DARPA SAIL-ON`_. The library is based on the work supported by DARPA under
Contract No. HR001120C0055 and is licensed under `APACHE-V2 LICENSE`_.

Sail-On Client is being developed as an open source library to support evaluating **vision centric**
open world agents. This involves communicating with evaluation server across a RESTful interface,
providing empirical protocols that are used to systematically evaluate the agents capabilities across
multiple levels of the novelty hierarchy.

Sail-On Client is built upon `tinker-engine`_, an open source library that provides the
abstractions and interface used for running the empirical protocols

Features
--------

The major features provided by Sail-On Client

- RESTful interface for communicating with the evaluation server
- Empirical protocols for evaluating open world agents
- Dynamic algorithm discovery via plugins
- Configurable components with sensible defaults
- Saving and Restoring parameter for an algorithm

User Guide
----------

.. toctree::
   :maxdepth: 2

   user/install

Hydra
-----

.. toctree::
   :maxdepth: 2

   hydra/motivation
   hydra/config
   hydra/cli
   hydra/hydra_plugins

Empirical Protocols
-------------------

.. toctree::
   :maxdepth: 2

   protocols/ond
   protocols/condda
   protocols/protocol_api

Interfaces
----------

.. toctree::
   :maxdepth: 2

   harness/harness
   harness/harness_api

Metrics
-------

.. toctree::
   :maxdepth: 2

   metrics/metric_formulation
   metrics/metric_api

Feedback
--------

.. toctree::
   :maxdepth: 2

   feedback/feedback
   feedback/feedback_api

Agents
------

.. toctree::
   :maxdepth: 2

   agents/agents
   agents/agents_api

Saving and Restoring Models
---------------------------

.. toctree::
   :maxdepth: 2

   saving/features
   saving/checkpoint
   saving/checkpoint_api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. Appendix 1: Links

.. _DARPA SAIL-ON: https://www.darpa.mil/program/science-of-artificial-intelligence-and-learning-for-open-world-novelty
.. _APACHE-V2 LICENSE: https://www.apache.org/licenses/LICENSE-2.0
.. _tinker-engine: https://github.com/tinker-engine/tinker-engine
.. _SAIL-ON Website: https://www.darpa.mil/program/science-of-artificial-intelligence-and-learning-for-open-world-novelty
