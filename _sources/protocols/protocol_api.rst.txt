Protocol API
============

Visual Protocol API
-------------------

Interfaces that are common across OND and CONDDA protocol

Visual Protocol Dataclasses
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sail_on_client.protocol.visual_dataclasses
    :members:
    :inherited-members:
    :sourcelink:


Visual Round
^^^^^^^^^^^^

.. automodule:: sail_on_client.protocol.visual_round
    :members:
    :private-members: _run_feature_extraction, _run_world_change_detection
    :inherited-members:
    :sourcelink:


Visual Test
^^^^^^^^^^^

.. automodule:: sail_on_client.protocol.visual_test
    :members:
    :private-members: _save_features, _restore_features
    :inherited-members:
    :sourcelink:

Visual Protocol
^^^^^^^^^^^^^^^

.. automodule:: sail_on_client.protocol.visual_protocol
    :members:
    :inherited-members:
    :sourcelink:


OND API
-------

OND Dataclasses
^^^^^^^^^^^^^^^

.. automodule:: sail_on_client.protocol.ond_dataclasses
    :members:
    :inherited-members:
    :sourcelink:


OND Round
^^^^^^^^^

.. automodule:: sail_on_client.protocol.ond_round
    :members:
    :private-members: _run_novelty_classification, _run_novelty_adaptation, _run_evaluate_roundwise
    :inherited-members:
    :special-members: __call__
    :sourcelink:

OND Test
^^^^^^^^

.. automodule:: sail_on_client.protocol.ond_test
    :members:
    :private-members: _create_feedback_instance, _run_novelty_characterization
    :inherited-members:
    :special-members: __call__
    :sourcelink:


OND Protocol
^^^^^^^^^^^^

.. automodule:: sail_on_client.protocol.ond_protocol
    :members:
    :inherited-members:
    :private-members: _evaluate_algorithms
    :sourcelink:


CONDDA API
----------

CONDDA Dataclasses
^^^^^^^^^^^^^^^^^^

.. automodule:: sail_on_client.protocol.condda_dataclasses
    :members:
    :inherited-members:
    :sourcelink:


CONDDA Round
^^^^^^^^^^^^

.. automodule:: sail_on_client.protocol.condda_round
    :members:
    :private-members: _run_novelty_characterization
    :inherited-members:
    :special-members: __call__
    :sourcelink:

CONDDA Test
^^^^^^^^^^^

.. automodule:: sail_on_client.protocol.condda_test
    :members:
    :inherited-members:
    :special-members: __call__
    :sourcelink:


CONDDA Protocol
^^^^^^^^^^^^^^^

.. automodule:: sail_on_client.protocol.condda_protocol
    :members:
    :inherited-members:
    :sourcelink:
