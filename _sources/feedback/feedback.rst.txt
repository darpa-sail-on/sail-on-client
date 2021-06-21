Feedback
========

Introduction
------------

To facilitate adaption when novelty is introduced, Sail On agents can use feedback.
The feedback is assumed to be in the same space as the predictions. Thus for OND,
feeback is provided in K+1 space and for CONDDA feedback is provided in K+D+1 space.
For example, if the space of prediction is novel vs. not novel, the feedback
for a sample would be novel/not novel; if prediction is one of K+1 classes (K + unknown),
the feedback will be correct class from K+1.

Feedback can be requested for any sample in the last batch after the TA2 declares
that novelty has been introduced and has submitted a prediction for it.
Feedback is budgeted for every round for the protocols and for evaluation 10%
feedback is provided by novelty generators.


Feedback Across Domains
-----------------------

+------------------------+----------+--------------------------------------+-----------------------------------------+
|         Domain         | Protocol |                Outputs               |                 Feedback                |
+========================+==========+======================================+=========================================+
|                        |    OND   |        Class Predictions (K+1)       | One hot class vector in [K+1] dimension |
|  Image Classification  +----------+--------------------------------------+-----------------------------------------+
|                        |  CONDDA  | Characterization Predictions (K+D+1) |             Not defined yet             |
+------------------------+----------+--------------------------------------+-----------------------------------------+
|                        |    OND   |        Class Predictions (K+1)       | One hot class vector in [K+1] dimension |
|                        +----------+--------------------------------------+-----------------------------------------+
| Document Transcription |    OND   |        Class Predictions (K+1)       |          Levenshtein Distance           |
|                        +----------+--------------------------------------+-----------------------------------------+
|                        |  CONDDA  | Characterization Predictions (K+D+1) |             Not defined yet             |
+------------------------+----------+--------------------------------------+-----------------------------------------+
|                        |    OND   |        Class Predictions (K+1)       | One hot class vector in [K+1] dimension |
|  Activity Recognition  +----------+--------------------------------------+-----------------------------------------+
|                        |  CONDDA  | Characterization Predictions (K+D+1) |             Not defined yet             |
+------------------------+----------+--------------------------------------+-----------------------------------------+
