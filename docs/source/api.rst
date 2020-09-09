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
