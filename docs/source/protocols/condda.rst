Continuous Open-set Novelty Detection, Discovery, Adaptation (CONDDA)
=====================================================================

.. attention:: While sail-on-client supports CONDDA protocol, it is not used in
               DARPA evaluation.

**K+D+1 classification where K represents known classes, D are dicovered classes, 1 are all unknown classes**

Introduction
------------

This combines classification with novelty/object discovery (and is more natural
for Class Novelty). A TA2 agent starts with a K+1-way classification model (K known,
1 unknown) as in OND. During testing, the TA2 agent will discover clusters of
novelty (e.g., discover new classes) and accordingly add those as separate
classes in the prediction space (i.e., one of D discovered classes).

.. hint:: In CONDDA, once an agent is able to characterize a class, the class is
          considered to be discovered rather than unknown. Thus after an agent is
          able to consistently identify the instances coming from unknown class
          is coming from the same class, it is considered a discovered class.

Prediction Space
----------------

1. Binary Classification Score: :math:`P_{world\_changed}`
2. K + D + 1 Characterization Score: :math:`P_{class} = [ p_{unknown}, p_1, ..., p_k, p_{cluster0}, ..., p_{clusterU}]`


WorkFlow For CONDDA
-------------------

This section provides detailed workflow of the system in the evaluation condition.
CONDDA protocol evaluates the novelty Characterization capabilities of an agent
along with its ability to detect when novelty was introduced.

.. note:: The point where novelty is introduced is canonically referred to as the
   point when the red-button was pushed or when the agent saw the red-light.

CONDDA workflow is executed under 2 different experimental setting

1. Without Red Light (System Detection)
2. With Red Light (Given Detection)

CONDDA Workflow Without Red Light (System Detection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The workflow of CONDDA without red light is as follow.

#. Create a session on the server by providing a list of tests and hints
#. For a test with a fixed number of images/videos,
      #. Provide a mini-batch of image/video ids to the agent.
      #. The agent would use the mini-batch to determine if the novelty was introduced
         in the batch and provide K+D+1 Characterization score for every sample
         in the mini-batch
      #. Post the results the mini-batch to the server.
#. Declare that the test is complete.
#. Terminate the session after all tests are complete.

.. figure:: ../images/CONDDA-Updated.png
   :alt: Workflow for CONDDA without red light
   :align: center
   :figclass: align-center


CONDDA Workflow With Red Light (Given Detection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The workflow of CONDDA with red light is as follow.

.. hint:: In given detection the agent is explicitly provided the image/video id
          where novelty was introduced in the test.

#. Create a session on the server by providing a list of tests and hints
#. For a test with a fixed number of images/videos,
      #. Provide the image/video id where novelty is introduced.
      #. Provide a mini-batch of image/video ids to the agent.
      #. The agent would use the mini-batch to determine and provide K+D+1
         Characterization score for every sample in the mini-batch.
      #. Post the results the mini-batch to the server.
#. Declare that the test is complete.
#. Terminate the session after all tests are complete.

.. figure:: ../images/CONDDA-With-Red-Light-Updated.png
   :alt: Workflow for CONDDA with red light
   :align: center
   :figclass: align-center
