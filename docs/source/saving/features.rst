Saving And Restoring Features for Algorithm
===========================================

Introduction
------------

We rely on configuration options to save and restore features associated with
an algorithm. The features are saved in pickle files as dictionary with the
following keys `features_dict` and `logit_dict`.


Config Options For Saving Features
----------------------------------

.. attention:: Along with the config options provided below, the configuration
               saving features also require the directory where features would be
               saved in `protocol.smqtk.config.feature_dir`. This can be
               provided in the configuration file or supplied to the cli.

Default Option For Saving Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the protocol does not save feature. This option can be used to
disable saving features in a config

.. code-block:: yaml

   defaults:
      - protocol/save_features@protocol.smqtk.config: none

Saving Features Elementwise
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In elementwise option, `features_dict` and `logit_dict` would use image/video ids
as keys with features as values. This option is used for feature extraction in
activity recognition where video features can be reused across multiple tests

.. code-block:: yaml

   defaults:
      - protocol/save_features@protocol.smqtk.config: elementwise


Saving Features Testwise
^^^^^^^^^^^^^^^^^^^^^^^^

In testwise option, the pickle file would be saved for every test that was present
in the session. This option is used for replicating an experiment with pre-computed features.

.. code-block:: yaml

   defaults:
      - protocol/save_features@protocol.smqtk.config: testwise

.. note:: When `testwise` is used, the protocol would assume that the features
        are saved in `<test_id>_<agent_name>`_features.pkl`.


Config Options For Using Features
---------------------------------

.. attention:: Along with the config options provided below, the configuration
               using features also require the directory where features are
               present in `protocol.smqtk.config.feature_dir`. This can be
               provided in the configuration file or supplied to the cli.

Default Option For Using Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the protocol does not use saved feature. This option can be used to
disable using features specified in a config

.. code-block:: yaml

   defaults:
      - protocol/use_features@protocol.smqtk.config: none

Using Consolidated Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In consolidated option, `features_dict` and `logit_dict` would use image/video ids
as keys with features as values. This option is the complement of `elementwise`
option in the previous section and is used to re-use features to save time during
evaluation.

.. code-block:: yaml

   defaults:
      - protocol/use_features@protocol.smqtk.config: consolidated

.. note:: When `consolidated` is used, the protocol would assume that the features
          are present in `<agent_name>`_features.pkl`.

Using Non Consolidated Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In non consolidated option, `features_dict` and `logit_dict` are assumed to be
coming from a single test. This option is used for replicating an experiment
with pre-computed features.

.. code-block:: yaml

   defaults:
      - protocol/use_features@protocol.smqtk.config: non_consolidated

.. note:: When `non_consolidated` is used, the protocol would assume that the features
        are present in `<test_id>_<agent_name>`_features.pkl`.
