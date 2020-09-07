PAR Interface
-------------

Introduction
^^^^^^^^^^^^

:code:`ParInterface` is primarily responsible for communicating with the evaluation
server setup by the TA1 team. The interface relies on RESTful api :ref:`(detailed in the next section)<REST API>`  to provide the
following features

1. Support batch inquiry (also called rounds) with full data set response.
2. Support batch response for evaluation.
3. Answer requests for multiple dataset types.
4. Includes option for ‘hints’ accompanying datasets.
5. Can accept additional meta-data along with annotations, labels and localization data
   (e.g time intervals for video) along with class and certainty scores.
6. Provide feedback as requested by the algorithm after the results for a batch have been
   submitted.


REST API
^^^^^^^^

This section provides a detailed description of the RESTful api used for communication.

+-------------------+--------------+---------------------------------------------------+-------------------------------------------------------+---------------------------------------------------------+
|    Request Name   | Request Type |                     Definition                    | Request Data                                          | Response Data                                           |
+===================+==============+===================================================+=======================================================+=========================================================+
|    Test Request   |      GET     |     TA2 Requests for Test Identifiers             | 1. Protocol: Empirical protocol                       | 1. CSV file containing: Test ID(s) with the             |
|                   |              |     as part of a series of individual tests.      | 2. Domain: Problem domain.                            | following naming convention: Protocol.Group.Run.Seed    |
|                   |              |                                                   | 3. Detector Seed                                      |                                                         |
|                   |              |                                                   | 4. JSON file: Test Assumptions (if any)               |                                                         |
+-------------------+--------------+---------------------------------------------------+-------------------------------------------------------+---------------------------------------------------------+
|    New Session    |     POST     |       Create a new session to evaluate the        | 1. Test ID(s) obtained from test request.             | 1. Session ID: A unique identifier that the             |
|                   |              |       detector using an empirical protocol.       | 2. Protocol                                           |    server associated with the client                    |
|                   |              |                                                   | 3. Novelty Detector Version                           |                                                         |
+-------------------+--------------+---------------------------------------------------+-------------------------------------------------------+---------------------------------------------------------+
|  Dataset Request  |      GET     |           Request data for evaluation.            | 1. Session ID                                         | 1. CSV of Dataset URIs. Each URI identifies the media   |
|                   |              |                                                   | 2. Test ID                                            |    to be used. It may be location specific (e.g. S3)    |
|                   |              |                                                   | 3. Round ID (where applicable per protocol)           |    or independent, assuming a shared repository.        |
+-------------------+--------------+---------------------------------------------------+-------------------------------------------------------+---------------------------------------------------------+
|    Get Feedback   |      GET     |       Get Feedback from the server based on       | 1. Session ID                                         | 1. CSV file for detection and characterization feedback |
|                   |              |              one or more example ids              | 2. Test ID                                            |    in accordance with the feedback space specified by   |
|                   |              |                                                   | 3. Round Number                                       |    the protocol.                                        |
|                   |              |                                                   | 4. Example IDs                                        |                                                         |
|                   |              |                                                   | 5. Feedback type: Detection,  Characterization, Label |                                                         |
+-------------------+--------------+---------------------------------------------------+-------------------------------------------------------+---------------------------------------------------------+
|    Get Metadata   |      GET     |              Get metadata for a test              | 1. Test ID                                            | 1. JSON File containing the metadata.                   |
+-------------------+--------------+---------------------------------------------------+-------------------------------------------------------+---------------------------------------------------------+
|    Post Results   |     POST     | Post client detector predictions for the dataset. | 1. Session ID                                         | 1. Result acknowledgement.                              |
|                   |              |                                                   | 2. Test ID                                            |                                                         |
|                   |              |                                                   | 3. Round ID (where applicable)                        |                                                         |
|                   |              |                                                   | 4. Result Files: (CSV)                                |                                                         |
|                   |              |                                                   | 5. Protocol constant: Characterization/Detection      |                                                         |
+-------------------+--------------+---------------------------------------------------+-------------------------------------------------------+---------------------------------------------------------+
|    Evaluation     |      GET     |              Get results for test(s)              | 1. Session ID                                         | 1. Score or None.                                       |
|                   |              |                                                   | 2. Test ID                                            |                                                         |
|                   |              |                                                   | 3. Round ID (where applicable )                       |                                                         |
+-------------------+--------------+---------------------------------------------------+-------------------------------------------------------+---------------------------------------------------------+
| Terminate Session |    DELETE    |     Terminate the session after the evaluation    | 1. Session ID                                         | 1. Acknowledgement of session termination.              |
|                   |              |            for the protocol is complete           | 2. Logs for the session                               |                                                         |
+-------------------+--------------+---------------------------------------------------+-------------------------------------------------------+---------------------------------------------------------+
