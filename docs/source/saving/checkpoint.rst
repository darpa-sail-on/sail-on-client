Saving And Restoring Attributes of Algorithm
============================================

Introduction
------------

We rely on :code:`Checkpoint` mixin to save and restore attribute of an algorithm.
The mixin can be added to an adaptor associated with an algorithm to provide
:code:`save_attributes` and :code:`restore_attributes` function. These functions
can be used with configuration options to save and restore states in a pickle
file.

.. warning::

    Saving and restorying attributes is experimental, and error-prone due
    to lack of a concrete use-case in the current evaluation.

.. note::

    Saving features and saving attributes were created with different use cases.
    Saving features is used to save features for videos and restore them
    across multiple tests. Thus the features can be extracted for a video
    once and can be reused across multiple tests. We cannot guarantee this
    mode of operation when checkpointing since attributes are not saved with
    videos as keys. Thus checkpointing can be used to skip steps for a protocol
    but it shouldn't be used on a sequence of videos that is different from
    the original sequence. This would result in undefined behavior.


Sample Detector and Adapter
---------------------------

Sample Detector
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../sail_on_client/agent/mock_ond_agents.py
   :language: python
   :lines: 130-159

Sample Adapter
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../sail_on_client/agent/mock_ond_agents.py
   :language: python
