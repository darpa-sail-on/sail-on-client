Saving And Restoring Attributes of Algorithm
============================================

Introduction
------------

We rely on :code:`Checkpoint` mixin to save and restore attribute of an algorithm.
The mixin can be added to an adaptor associated with an algorithm to provide
:code:`save_attributes` and :code:`restore_attributes` function. These functions
can be used with configuration options to save and restore states in a pickle
file. Refer to `ond_12_with_rd_nd_save.json` and `ond_12_with_rd_nd_restore.json`
configurations on the `config`_ directory for more details.


Sample Detector, Adapter and Configuration Parameters
-----------------------------------------------------

Sample Detector
^^^^^^^^^^^^^^^

.. literalinclude:: ../../sail_on_client/mock.py
   :language: python
   :lines: 120-149

Sample Adapter
^^^^^^^^^^^^^^^

.. literalinclude:: ../../sail_on_client/mock.py
   :language: python
   :lines: 152-191


Sample Configuration
^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../config/ond_12_with_rd_nd_save.json
   :language: json
   :lines: 6-16
   :emphasize-lines: 2-9

.. literalinclude:: ../../config/ond_12_with_rd_nd_restore.json
   :language: json
   :lines: 6-16
   :emphasize-lines: 2-8

.. Appendix 1: Links

.. _config: https://gitlab.kitware.com/darpa-sail-on/sail-on-client/-/tree/master/config
