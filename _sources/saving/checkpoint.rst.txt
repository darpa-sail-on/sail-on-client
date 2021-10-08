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

.. note::

    Saving features and saving attributes were created with different use cases.
    Saving features is used to save features for videos and restore them
    across multiple tests. Thus the features can be extracted for a video
    once and can be reused across multiple tests. We cannot guarantee this
    mode of operation when checkpointing since attributes are not saved with
    videos as keys. Thus checkpointing can be used to skip steps for a protocol
    but it shouldn't be used on a sequence of videos that is different from
    the original sequence. This would result in undefined behavior.


Sample Detector, Adapter and Configuration Parameters
-----------------------------------------------------

Sample Detector
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../sail_on_client/mock.py
   :language: python
   :lines: 120-152

Sample Adapter
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../sail_on_client/mock.py
   :language: python
   :lines: 153-178


Sample Configuration
^^^^^^^^^^^^^^^^^^^^

.. note::
    The configuration options used for saving attributes are highlighted in the
    examples below.

.. literalinclude:: ../../../config/ond_12_with_rd_nd_save.json
   :language: json
   :lines: 6-16
   :emphasize-lines: 1-7

.. literalinclude:: ../../../config/ond_12_with_rd_nd_restore.json
   :language: json
   :lines: 6-16
   :emphasize-lines: 1-7

.. Appendix 1: Links

.. _config: https://gitlab.kitware.com/darpa-sail-on/sail-on-client/-/tree/master/config
