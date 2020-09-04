Running The Algorithms
----------------------

This part of the documentation goes over the steps for executing different algorithms using different protocols.

Running OND_5_14_A1 Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Go to sail-on server directory and start the server using::

      $ cd sail-on
      $ sail_on_server --data-directory data/ --results-directory ond_5_14_a1
2. Go to the sail on client repository and make a copy of the configuration file for running OND_5_14_A1::

      $ cd sail-on-client
      $ cp config/ond_5_14_a1_nd.json config/local_ond_5_14_a1_nd.json
3. Download the `EVM Model`_.
4. Change :code:`model_path` for :code:`evm_params` in :code:`local_ond_5_14_a1_nd.json` to
   point the model downloaded in the previous step.
5. Download the `Efficientnet Model`_.
6. Change :code:`model_path` for :code:`efficientnet_params` in :code:`local_ond_5_14_a1_nd.json`
   to point the model downloaded in the previous step.
7. Change :code:`dataset_root` in :code:`local_ond_5_14_a1_nd.json`
   to point to :code:`sail-on/images` directory.
8. Run the client::

      $ tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_ond_5_14_a1_nd.json


Running OND_5_14_A2 Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Go to sail-on server directory and start the server using::

      $ cd sail-on
      $ sail_on_server --data-directory data/ --results-directory ond_5_14_a2

2. Go to the sail on client repository and make a copy of the configuration file for running OND_5_14_A2::

      $ cd sail-on-client
      $ cp config/ond_5_14_a2_nd.json config/local_ond_5_14_a2_nd.json

3. Download the `EVM Model`_.
4. Change :code:`model_path` for :code:`evm_params` in :code:`local_ond_5_14_a2_nd.json`
   to point the model downloaded in the previous step.
5. Download the `Efficientnet Model`_.
6. Change :code:`model_path` for :code:`efficientnet_params` in :code:`local_ond_5_14_a2_nd.json`
   to point the model downloaded in the previous step.
7. Change :code:`dataset_root` in :code:`local_ond_5_14_a1_nd.json`  to point to
   :code:`sail-on/images` directory.
8. Run the client::

      $ tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_ond_5_14_a2_nd.json


Running CONDDA_5_14_A1 Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Go to sail-on server directory and start the server using::

      $ cd sail-on
      $ sail_on_server --data-directory data/ --results-directory condda_5_14_a1

2. Go to the sail on client repository and make a copy of the configuration file for running CONDDA_5_14_A1::

      $ cd sail-on-client
      $ cp config/condda_5_14_a1_nd.json config/local_condda_5_14_a1_nd.json

3. Download the `EVM Model`_.
4. Change :code:`model_path` for :code:`evm_params` in :code:`local_condda_5_14_a1_nd.json`.
   to point the model downloaded in the previous step.
5. Download the `Efficientnet Model`_.
6. Change :code:`model_path` for :code:`efficientnet_params` in :code:`local_condda_5_14_a1_nd.json`
   to point the model downloaded in the previous step.
7. Download the `Precomputed Features`_.
8. Update the :code:`known_feature_path` in :code:`evm_params` in
   :code:`local_condda_5_14_a1_nd.json` to point to the features
   downloaded in the previous step
9. Download the `Training Images`_.
10. Change :code:`dataset_root` in :code:`local_condda_5_14_a1_nd.json`
    to point to directory where the images were installed in the previous step.
11. Run the client::

      $ tinker sail_on_client/protocol/condda.py -i ParInterface -p config/local_condda_5_14_a1_nd.json


Running CONDDA_5_14_A2 Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Go to sail-on server directory and start the server using::

      $ cd sail-on
      $ sail_on_server --data-directory data/ --results-directory condda_5_14_a2

2. Go to the sail on client repository and make a copy of the configuration
   file for running CONDDA_5_14_A2::

      $ cd sail-on-client
      $ cp config/condda_5_14_a2_nd.json config/local_condda_5_14_a2_nd.json

3. Download the `EVM Model`_.
4. Change :code:`model_path` for :code:`evm_params` in :code:`local_ond_5_14_a2_nd.json` to point
   the model downloaded in the previous step.
5. Download the `Efficientnet Model`_.
6. Change :code:`model_path` for :code:`efficientnet_params` in :code:`local_condda_5_14_a1_nd.json`
   to point the model downloaded in the previous step.
7. Download the `Precomputed Features`_.
8. Update the :code:`known_feature_path` in :code:`evm_params` in :code:`local_condda_5_14_a1_nd.json`
   to point to the features downloaded in the previous step.
9. Download the `Training Images`_.
10. Change :code:`dataset_root` in :code:`local_condda_5_14_a1_nd.json` to point to directory
    where the images were installed in the previous step.
11. Run the client::

      $ tinker sail_on_client/protocol/condda.py -i ParInterface -p config/local_condda_5_14_a2_nd.json


.. Appendix 1: Links

.. _EVM Model: https: https://drive.google.com/file/d/1XrSWQWJsF-iPkvGM4AWkMNqvhFTb0yfk/view?usp=sharing
.. _Efficientnet Model: https://drive.google.com/file/d/1esL1W7pDHrsTmLpSFxWdzOg6oP-p8IDi/view?usp=sharing
.. _Precomputed Features: https://drive.google.com/file/d/1fzRv-8ngv89YB0J91SNejEvCuVnJK_e7/view?usp=sharing
.. _Training Images: https://drive.google.com/file/d/1QU_wD-erA1ijMZ29B1NT9ubjxF5HbImo/view?usp=sharing
