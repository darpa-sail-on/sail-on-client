API Documentation
=================

.. module:: sail_on_client

This part of the documentation provides reference documentation for all the modules
of the package

Interfaces
----------

We currently support HTTP rest based interface to communicate with the evaluation
server. The interface has the following functions

.. autoclass:: sail_on_client.protocol.parinterface.ParInterface
    :members:

.. autoclass:: sail_on_client.protocol.localinterface.LocalInterface
    :members:

Protocols
---------

We currently support two empirical protocols OND and CONDDA. Both the protocols
have the same interface with the following functions

.. autoclass:: sail_on_client.protocol.ond_protocol.SailOn
    :members:
    :undoc-members:


.. autoclass:: sail_on_client.protocol.condda.Condda
    :members:
    :undoc-members:

Metrics
-------

We currently support metrics for image classification, document transcription
and activity recognition domain with the following functions

.. autoclass:: sail_on_client.evaluate.image_classification.ImageClassificationMetrics
    :members:
    :undoc-members:

.. autoclass:: sail_on_client.evaluate.activity_recognition.ActivityRecognitionMetrics
    :members:
    :undoc-members:

.. autoclass:: sail_on_client.evaluate.document_transcription.DocumentTranscriptionMetrics
    :members:
    :undoc-members:

The metrics inherit from the program metric class with the following functions

.. autoclass:: sail_on_client.evaluate.metrics.ProgramMetrics
    :members:
    :undoc-members:

Errors
------

We currently support 3 types of errors

.. autoclass:: sail_on_client.errors.ServerError
    :members:
    :inherited-members:


.. autoclass:: sail_on_client.errors.RoundError
    :members:
    :inherited-members:


.. autoclass:: sail_on_client.errors.ProtocolError
    :members:
    :inherited-members:

The errors can be expanded upon by inheriting the base error class

.. autoclass:: sail_on_client.errors.ApiError
    :members:


Utils
-----

.. automodule:: sail_on_client.utils
    :members:
