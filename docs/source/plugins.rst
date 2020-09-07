Plugins
=======

Introduction
------------

We rely on `Entry Points`_ to register an algorithm . The entrypoint specification
can be used by a python package to register an algorithm by specifying a key value
pair in setup.py for the package where the key would be the name of the algorithm
and the value is the path to the algorithm class relative to the package root. Refer
to :ref:`the registration section<Sample Detector and Registration>` for an example.

Functions Used By the Protocol
------------------------------

The protocols assumes that the algorithm have the following functions during evaluation

1. :code:`FeatureExtraction`: Primarily used for extracting features from the input
   modality so that it can be used by other stages of the algorithm.
2. :code:`WorldDetection`: Encapsulates the functionality that is used by detecting
   change in distribution during an experiment. The logic in this function is used
   to determine the introduction of novelty in an experiment.
3. :code:`NoveltyClassification`: Used for providing the probability of a sample being
   novel or known. The probability of known class is subdivided into a distribution over
   all the known classes. This results in a prediction of k+1 values for every sample,
   where k is the number of known classes and 1 value is reserved predicting that the sample
   is novel.
4. :code:`NoveltyCharacterization`: Similar to :code:`NoveltyClassification` function with
   one major difference in the output. The probability of unknown class is subdivided
   into a distribution over the unknown classes. This results in a prediction of k+d+1 values
   for every sample, where k is the number of known classes, d is the number of discovered
   classes and 1 value is reserved for the probability that the sample is novel.
5. :code:`NoveltyAdaption`: Used by the algorithm to update its internal state after
   it had provided results for :code:`WorldDetection` and :code:`NoveltyClassification`/
   :code:`NoveltyCharacterization` depending on the protocol.

Adapters And Toolset
--------------------

TBD

Sample Detector and Registration
--------------------------------

Sample Detector
^^^^^^^^^^^^^^^

.. literalinclude:: ../../sail_on_client/mock.py
   :language: python

Registering the Detector
^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../setup.py
   :language: python
   :lines: 11-30
   :emphasize-lines: 20

.. Appendix 1: Links

.. _Entry Points: https://packaging.python.org/specifications/entry-points/
