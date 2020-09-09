Local Interface
-------------

Introduction
^^^^^^^^^^^^

:code:`LocalInterface` is primarily used for replicating the capabilities of
:code:`PARInterface` without using the server. This allows local testing without
setting up a server instance locally or via a URL. Since LocalInterface uses
the files directly, the config parameter `data_dir` needs to be specified to
use this interface.
