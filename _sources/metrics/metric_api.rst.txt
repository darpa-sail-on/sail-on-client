Metric API
==========

Methods For Metric
------------------

.. autofunction:: sail_on_client.evaluate.metrics.m_num

.. autofunction:: sail_on_client.evaluate.metrics.m_num_stats

.. autofunction:: sail_on_client.evaluate.metrics.m_ndp

.. autofunction:: sail_on_client.evaluate.metrics.m_ndp_pre

.. autofunction:: sail_on_client.evaluate.metrics.m_ndp_post

.. autofunction:: sail_on_client.evaluate.metrics.m_ndp_failed_reaction

.. autofunction:: sail_on_client.evaluate.metrics.m_acc

.. autofunction:: sail_on_client.evaluate.metrics.m_accuracy_on_novel

Base Class For Metric
---------------------

.. autoclass:: sail_on_client.evaluate.program_metrics.ProgramMetrics
    :members:
    :inherited-members:

Activity Recognition Metric
---------------------------

.. autoclass:: sail_on_client.evaluate.activity_recognition.ActivityRecognitionMetrics
    :members:
    :inherited-members:

Document Transcription Metric
-----------------------------

.. autoclass:: sail_on_client.evaluate.document_transcription.DocumentTranscriptionMetrics
    :members:
    :inherited-members:

Image Classification Metric
---------------------------

.. autoclass:: sail_on_client.evaluate.image_classification.ImageClassificationMetrics
    :members:
    :inherited-members:

Utility Methods
---------------

.. autofunction:: sail_on_client.evaluate.utils.check_novel_validity

.. autofunction:: sail_on_client.evaluate.utils.check_class_validity

.. autofunction:: sail_on_client.evaluate.utils.topk_accuracy

.. autofunction:: sail_on_client.evaluate.utils.top1_accuracy

.. autofunction:: sail_on_client.evaluate.utils.top3_accuracy

.. autofunction:: sail_on_client.evaluate.utils.get_rolling_stats

.. autofunction:: sail_on_client.evaluate.utils.get_first_detect_novelty
