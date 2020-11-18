## Instructions For Running Experiments With M6 Algorithms

### Running Image Classification Experiments

#### Running OND_5_14_A1 Algorithm
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory ond_5_14_a1
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running OND_5_14_A1
    ```
      cd sail-on-client
      cp config/ond_5_14_a1_nd.json config/local_ond_5_14_a1_nd.json
    ```
3. Download the evm model from following [link](https://drive.google.com/file/d/1XrSWQWJsF-iPkvGM4AWkMNqvhFTb0yfk/view?usp=sharing)
4. Change `model_path` for `evm_params` in `local_ond_5_14_a1_nd.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://drive.google.com/file/d/1esL1W7pDHrsTmLpSFxWdzOg6oP-p8IDi/view?usp=sharing)
5. Change `model_path` for `efficientnet_params` in `local_ond_5_14_a1_nd.json` to point the model downloaded in the previous step
6. Change `dataset_root` in `local_ond_5_14_a1_nd.json`  to point to `sail-on/images` directory
7. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_ond_5_14_a1_nd.json
    ```

#### Running OND_5_14_A2 Algorithm
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory ond_5_14_a2
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running OND_5_14_A2
    ```
      cd sail-on-client
      cp config/ond_5_14_a2_nd.json config/local_ond_5_14_a2_nd.json
    ```
3. Download the evm model from following [link](https://drive.google.com/file/d/1XrSWQWJsF-iPkvGM4AWkMNqvhFTb0yfk/view?usp=sharing)
4. Change `model_path` for `evm_params` in `local_ond_5_14_a2_nd.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://drive.google.com/file/d/1esL1W7pDHrsTmLpSFxWdzOg6oP-p8IDi/view?usp=sharing)
5. Change `model_path` for `efficientnet_params` in `local_ond_5_14_a2_nd.json` to point the model downloaded in the previous step
6. Change `dataset_root` in `local_ond_5_14_a1_nd.json`  to point to `sail-on/images` directory
7. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_ond_5_14_a2_nd.json
    ```

#### Running CONDDA_5_14_A1 Algorithm
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory condda_5_14_a1
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running CONDDA_5_14_A1
    ```
      cd sail-on-client
      cp config/condda_5_14_a1_nd.json config/local_condda_5_14_a1_nd.json
    ```
3. Download the evm model from following [link](https://drive.google.com/file/d/1XrSWQWJsF-iPkvGM4AWkMNqvhFTb0yfk/view?usp=sharing)
4. Change `model_path` for `evm_params` in `local_condda_5_14_a1_nd.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://drive.google.com/file/d/1esL1W7pDHrsTmLpSFxWdzOg6oP-p8IDi/view?usp=sharing)
5. Change `model_path` for `efficientnet_params` in `local_condda_5_14_a1_nd.json` to point the model downloaded in the previous step
6. Download the precomputed features from following [link](https://drive.google.com/file/d/1fzRv-8ngv89YB0J91SNejEvCuVnJK_e7/view?usp=sharing)
7. Update the `known_feature_path` in `evm_params` in `local_condda_5_14_a1_nd.json` to point to the features downloaded in the previous step
8. Download the training images using the following [link](https://drive.google.com/file/d/1QU_wD-erA1ijMZ29B1NT9ubjxF5HbImo/view?usp=sharing)
9. Change `dataset_root` in `local_condda_5_14_a1_nd.json`  to point to directory where the images were installed in the previous step
10. Run the client
    ```
      tinker sail_on_client/protocol/condda.py -i ParInterface -p config/local_condda_5_14_a1_nd.json
    ```

#### Running CONDDA_5_14_A2 Algorithm
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory condda_5_14_a2
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running CONDDA_5_14_A2
    ```
      cd sail-on-client
      cp config/condda_5_14_a2_nd.json config/local_condda_5_14_a2_nd.json
    ```
3. Download the evm model from following [link](https://drive.google.com/file/d/1XrSWQWJsF-iPkvGM4AWkMNqvhFTb0yfk/view?usp=sharing)
4. Change `model_path` for `evm_params` in `local_ond_5_14_a2_nd.json` to point the model downloaded in the previous step
5. Download the efficientnet model from following [link](https://drive.google.com/file/d/1esL1W7pDHrsTmLpSFxWdzOg6oP-p8IDi/view?usp=sharing)
5. Change `model_path` for `efficientnet_params` in `local_condda_5_14_a1_nd.json` to point the model downloaded in the previous step
6. Download the precomputed features from following [link](https://drive.google.com/file/d/1fzRv-8ngv89YB0J91SNejEvCuVnJK_e7/view?usp=sharing)
7. Update the `known_feature_path` in `evm_params` in `local_condda_5_14_a1_nd.json` to point to the features downloaded in the previous step
8. Download the training images using the following [link](https://drive.google.com/file/d/1QU_wD-erA1ijMZ29B1NT9ubjxF5HbImo/view?usp=sharing)
9. Change `dataset_root` in `local_condda_5_14_a1_nd.json`  to point to directory where the images were installed in the previous step
10. Run the client
    ```
      tinker sail_on_client/protocol/condda.py -i ParInterface -p config/local_condda_5_14_a2_nd.json
    ```

