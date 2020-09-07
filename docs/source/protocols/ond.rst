Open World Novelty Detection (OND)
==================================

**K+1 classification where K represents known classes, all unknown classes are grouped together**

Introduction
------------

This is the standard classification setup with an unknown class to flag/reject
samples that are deemed novel. A TA2 agent can adapt its internal representation
in any way during testing to model novel samples, but the expected prediction
space (K+1-classes) remains the same throughout a run.

Image Classification Domain
---------------------------

Prediction Space
^^^^^^^^^^^^^^^^

1. Binary Classification Score: :math:`P_{world\_changed}`
2. K + 1 Classification Score: :math:`P_{class} = [ p_{unknown}, p_1, ..., p_k]`

WorkFlow For OND
^^^^^^^^^^^^^^^^

This section provides detailed workflow of the system in the evaluation condition

OND Workflow With Red Light
"""""""""""""""""""""""""""

.. figure:: ../images/OND-Updated.png
   :alt: Workflow for OND without red light
   :align: center
   :figclass: align-center


OND Workflow With Red Light
"""""""""""""""""""""""""""
.. figure:: ../images/OND-With-Red-Light-Updated.png
   :alt: Workflow for OND with red light
   :align: center
   :figclass: align-center
