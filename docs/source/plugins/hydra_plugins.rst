Hydra Plugins
=============

Introduction
------------

By default the cli supports configurations present in 2 search paths

1. Default path :code:`sail_on_client.configs` present in the sail-on-client package.
2. :code:`config-dir` provided to the cli by the user.

However there might be use-cases where the configuration files might be present
in multiple packages. An example use case would be an agent that is split across
multiple python package. In this case, the user could use `hydra searchpath plugin`_
to provide additional directories where the configuration files might be available.


Searchpath Plugin
-----------------

To create add a new search path to cli, use the following steps

1. Create a `hydra_plugins` folder at the root directory of your package.
2. Create a class in it that inherits from :code:`SearchPathPlugin`. Check the
   example plugin for more details

   .. literalinclude:: ../../../hydra_plugins/additional_searchpath.py
      :language: python

3. The example plugin above registers test directory found under hydra_plugins
   directory to the search path used by the cli.
4. Add additional configuration files or overrides in the directory that was
   registered in the previous step.
5. Check if the search path has been registered using::

      sail-on-client --info searchpath

.. note::
   Using :code:`pkg://` syntax requires an `__init__.py` in the directory specified
   in the plugin. Please refer to `example plugin` for a concrete example.

.. note::
   Testing plugins locally can sometimes cause errors with default hydra plugins
   like colorlor. To resolve the issue use `pip install .`


.. Apendix 1: Links

.. _hydra searchpath plugin: https://hydra.cc/docs/advanced/search_path/#creating-a-searchpathplugin
.. _example plugin: https://github.com/darpa-sail-on/sail-on-client/tree/master/hydra_plugins
