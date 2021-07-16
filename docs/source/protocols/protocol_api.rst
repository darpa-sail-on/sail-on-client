Protocol API
============

Visual Protocol API
-------------------

Interfaces that are common across OND and CONDDA protocol

Visual Protocol Dataclasses
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sail_on_client.protocol.visual_dataclasses.FeatureExtractionParams
    :members:
    :inherited-members:

.. autoclass:: sail_on_client.protocol.visual_dataclasses.WorldChangeDetectionParams
    :members:
    :inherited-members:

Visual Round
^^^^^^^^^^^^

.. autoclass:: sail_on_client.protocol.visual_round.VisualRound
    :members:
    :private-members: _run_feature_extraction, _run_world_change_detection
    :inherited-members:


Visual Test
^^^^^^^^^^^

.. autoclass:: sail_on_client.protocol.visual_test.VisualTest
    :members:
    :private-members: _save_features, _restore_features
    :inherited-members:

Visual Protocol
^^^^^^^^^^^^^^^

.. autoclass:: sail_on_client.protocol.visual_protocol.VisualProtocol
    :members:
    :inherited-members:

OND API
-------

OND Dataclasses
^^^^^^^^^^^^^^^

.. autoclass:: sail_on_client.protocol.ond_dataclasses.AlgorithmAttributes
    :members:
    :inherited-members:

.. autoclass:: sail_on_client.protocol.ond_dataclasses.InitializeParams
    :members:
    :inherited-members:

.. autoclass:: sail_on_client.protocol.ond_dataclasses.NoveltyClassificationParams
    :members:
    :inherited-members:

.. autoclass:: sail_on_client.protocol.ond_dataclasses.NoveltyCharacterizationParams
    :members:
    :inherited-members:

OND Round
^^^^^^^^^

.. autoclass:: sail_on_client.protocol.ond_round.ONDRound
    :members:
    :private-members: _run_novelty_classification, _run_novelty_adaptation, _run_evaluate_roundwise
    :inherited-members:
    :special-members: __call__

OND Test
^^^^^^^^

.. autoclass:: sail_on_client.protocol.ond_test.ONDTest
    :members:
    :private-members: _create_feedback_instance, _run_novelty_characterization
    :inherited-members:
    :special-members: __call__


OND Protocol
^^^^^^^^^^^^

.. autoclass:: sail_on_client.protocol.ond_protocol.ONDProtocol
    :members:
    :inherited-members:
    :private-members: _evaluate_algorithms


CONDDA API
----------

CONDDA Dataclasses
^^^^^^^^^^^^^^^^^^

.. autoclass:: sail_on_client.protocol.condda_dataclasses.AlgorithmAttributes
    :members:
    :inherited-members:

.. autoclass:: sail_on_client.protocol.condda_dataclasses.InitializeParams
    :members:
    :inherited-members:

.. autoclass:: sail_on_client.protocol.condda_dataclasses.NoveltyCharacterizationParams
    :members:
    :inherited-members:

CONDDA Round
^^^^^^^^^^^^

.. autoclass:: sail_on_client.protocol.condda_round.CONDDARound
    :members:
    :private-members: _run_novelty_characterization
    :inherited-members:
    :special-members: __call__

CONDDA Test
^^^^^^^^^^^

.. autoclass:: sail_on_client.protocol.condda_test.CONDDATest
    :members:
    :inherited-members:
    :special-members: __call__


CONDDA Protocol
^^^^^^^^^^^^^^^

.. autoclass:: sail_on_client.protocol.condda_protocol.Condda
    :members:
    :inherited-members:
