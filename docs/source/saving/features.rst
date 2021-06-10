Saving And Restoring Features for Algorithm
===========================================

Introduction
------------

We rely on configuration options to save and restore features associated with
an algorithm. Refer to `gae_nd_graph_fe.json` and `gae_nd_graph_precomputed.json`
configurations on the `config`_ directory for an example usage.


Sample Configuration
--------------------

Saving Features
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../config/gae_nd_graph_fe.json
   :language: json
   :emphasize-lines: 7-9

Restore Features
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../config/gae_nd_graph_precomputed.json
   :language: json
   :emphasize-lines: 7-10

.. Appendix 1: Links

.. _config: https://gitlab.kitware.com/darpa-sail-on/sail-on-client/-/tree/master/config

