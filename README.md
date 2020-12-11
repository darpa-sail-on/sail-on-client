# Sail-On Client and Protocols
[![pipeline status](https://gitlab.kitware.com/darpa-sail-on/sail-on-client/badges/master/pipeline.svg)](https://gitlab.kitware.com/darpa-sail-on/sail-on-client/-/commits/master) [![coverage report](https://gitlab.kitware.com/darpa-sail-on/sail-on-client/badges/master/coverage.svg)](https://gitlab.kitware.com/darpa-sail-on/sail-on-client/-/commits/master)


Client and Protocols for DARPA sail-on

### Protocols present in the repository
1. [OND](https://drive.google.com/file/d/1W2Ex4-eQl1CrAbv67fAN0OJL8kmRtvt2/view?usp=sharing)
2. [CONDDA](https://drive.google.com/file/d/1sIDLTgXivaguVfSp3g1qfe7sqiUcvFLA/view?usp=sharing)

### Client present in the repository
1. [Par Interface](https://gitlab.kitware.com/darpa-sail-on/sail-on-client/-/blob/master/sail_on_client/protocol/parinterface.py)


## Requirements

1. [Python 3.7](https://www.python.org/downloads/release/python-370/)
2. [pipenv](https://pipenv.pypa.io/en/latest/)
3. [tinker-engine](https://gitlab.kitware.com/darpa_learn/tinker-engine)
4. [Script Config](https://pypi.org/project/scriptconfig)
5. [TA-2 Agent](https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector)
6. [Sail-On API](https://gitlab.kitware.com/darpa-sail-on/sail-on-api)

## Installation

### Installation with pipenv ( Recommended )
1. Clone the repositories associated with different components in a working directory
    ```
      git clone https://gitlab.kitware.com/darpa_learn/tinker-engine.git
      git clone https://gitlab.kitware.com/darpa-sail-on/sail-on-api.git
      git clone https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector.git
      git clone https://gitlab.kitware.com/darpa-sail-on/graph-autoencoder.git
      git clone https://gitlab.kitware.com/darpa-sail-on/hwr_novelty_detector.git
      git clone https://gitlab.kitware.com/darpa-sail-on/sail-on-client.git
    ```
   This would create tinker-engine, sail-on-api, evm_based_novelty_detector, graph_autoencoder,
   hwr_novelty_detector and sail-on-client directories in your working directory

2. Checkout older version of tinker-engine using
   ```
      cd tinker-engine
      git checkout old-tinker
   ```

2. Install the different components in a virtual environment
   ```
   cd ../sail-on-client
   pipenv install
   pipenv shell
   ```

### Installation without pipenv
The installation requires cloning and installing multiple repositories. Thus after
following the instructions for a repository, please move back to your
your working directory.

#### Install Tinker Engine
1. Clone the [tinker-engine](https://gitlab.kitware.com/darpa_learn/tinker-engine) repository
   ```
   git clone https://gitlab.kitware.com/darpa_learn/tinker-engine.git
   ```
   This would create a directory called tinker-engine in your working directory

2. Install the dependencies of the tinker-engine in a virtual environment
   ```
   pipenv install
   pipenv shell
   ```
   This would create a virtual environment and activate the environment. Please use this virtual environment for installing all other repositories.

3. Install the tinker-engine in the virtual environment
   ```
   pip install -e .
   ```

#### Install Sail-On Server
1. Clone the [sail-on-api](https://gitlab.kitware.com/darpa-sail-on/sail-on-api) repository
   ```
    git clone https://gitlab.kitware.com/darpa-sail-on/sail-on-api.git
   ```
   This would create a directory called sail-on in your working directory

2. Go into the sail-on directory and install the dependencies for the server using
   ```
    cd sail-on-api
    pip install -r requirements.txt
   ```

3. Install the server
   ```
    pip install -e .
   ```

#### Install TA2 Agent
1. Clone [image classification algorithm](https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector) and [activity recognition] repository
   ```
    git clone https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector.git
    git clone https://gitlab.kitware.com/darpa-sail-on/graph-autoencoder.git
    git clone https://gitlab.kitware.com/darpa-sail-on/hwr_novelty_detector.git
   ```
   This would create a directory called evm_based_novelty_detector and graph-autoencoder your
   working directory

2. Go into the directory for image classifier repository and install the dependencies using
   ```
    cd evm_based_novelty_detector
    pip install -r requirements.txt
    pip install -e timm
   ```

3. Install the algorithm using
    ```
      pip install -e .
    ```

4. Go into the directory for activity recognition and install the dependencies using
   ```
    cd ../graph-autoencoder
    pip install -r requirements.txt
   ```

5. Install the algorithm using
    ```
     pip install -e .
    ```

6. Go into the directory for transcription and install the dependencies using
   ```
    cd ../hwr_novelty_detector
    pip install -r requirements.txt
   ```

7. Install the algorithm using
    ```
     pip install -e .
    ```

#### Install Sail-On Client
1. Clone the [sail-on-client](https://gitlab.kitware.com/darpa-sail-on/sail-on-client) repository
   ```
    git clone https://gitlab.kitware.com/darpa-sail-on/sail-on-client.git
   ```
   This would create a directory called sail-on-client in your working directory

2. Go into the sail-on-client directory and install the dependencies for the client using
   ```
    cd sail-on-client
    pip install -r requirements.txt
   ```

3. Install the client using
   ```
    pip install -e .
   ```

## Running Client and Server with Different Algorithms

Note: If you are using the server setup by PAR, update the `url` to `http://3.32.8.161:5000`
in sail_on_client/protocol/configuration.json and skip step 1 of running the server.

Note: The instructions assume that under given detection, the red-light information is
provided via metadata and under system detection, this information is determined by the algorithm.

### Running Image Classification Experiments

#### Running OND 12 With Red Light and Feedback
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory ond_12_with_rd
    ```
2. Go to the sail on client repository and make a copy of the configuration file for the algorithm
    ```
      cd sail-on-client
      cp config/ond_12_with_rd_nd.json config/local_ond_12_with_rd_nd.json
    ```
3. Download the evm model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/EVM_cosine_model_umd_b3_CC_tail40000_ct7_dm55.hdf5)
4. Change `model_path` for `evm_params` in `local_ond_12_with_rd_nd.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/trained_efficientnet_b3_CC.pth.tar)
5. Change `model_path` for `efficientnet_params` in `local_ond_12_with_rd_nd.json` to point the model downloaded in the previous step
6. Change `dataset_root` in `local_ond_12_with_rd_nd.json`  to point to `sail-on/images` directory
7. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_ond_12_with_rd_nd.json
    ```

#### Running OND 12 Without Red Light and with Feedback
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory ond_12_without_rd
    ```
2. Go to the sail on client repository and make a copy of the configuration file for the algorithm
    ```
      cd sail-on-client
      cp config/ond_12_wo_rd_nd.json config/local_ond_12_wo_rd_nd.json
    ```
3. Download the evm model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/EVM_cosine_model_umd_b3_CC_tail40000_ct7_dm55.hdf5)
4. Change `model_path` for `evm_params` in `local_ond_12_wo_rd_nd.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/trained_efficientnet_b3_CC.pth.tar)
5. Change `model_path` for `efficientnet_params` in `local_ond_12_wo_rd_nd.json` to point the model downloaded in the previous step
6. Change `dataset_root` in `local_ond_12_wo_rd_nd.json`  to point to `sail-on/images` directory
7. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_ond_12_wo_rd_nd.json
    ```

#### Running OND 12 With Red Light and Without Feedback
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory ond_12_with_rd
    ```
2. Go to the sail on client repository and make a copy of the configuration file for the algorithm
    ```
      cd sail-on-client
      cp config/ond_12_with_rd_nd_wo_feedback.json config/local_ond_12_with_rd_nd_wo_feedback.json
    ```
3. Download the evm model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/EVM_cosine_model_umd_b3_CC_tail40000_ct7_dm55.hdf5)
4. Change `model_path` for `evm_params` in `local_ond_12_with_rd_nd_wo_feedback.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/trained_efficientnet_b3_CC.pth.tar)
5. Change `model_path` for `efficientnet_params` in `local_ond_12_with_rd_nd_wo_feedback.json` to point the model downloaded in the previous step
6. Change `dataset_root` in `local_ond_12_with_rd_nd_wo_feedback.json`  to point to `sail-on/images` directory
7. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_ond_12_with_rd_nd_wo_feedback.json
    ```

#### Running OND 12 Without Red Light and Without Feedback
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory ond_12_without_rd
    ```
2. Go to the sail on client repository and make a copy of the configuration file for the algorithm
    ```
      cd sail-on-client
      cp config/ond_12_wo_rd_nd_wo_feedback.json config/local_ond_12_wo_rd_nd_wo_feedback.json
    ```
3. Download the evm model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/EVM_cosine_model_umd_b3_CC_tail40000_ct7_dm55.hdf5)
4. Change `model_path` for `evm_params` in `local_ond_12_wo_rd_nd_wo_feedback.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/trained_efficientnet_b3_CC.pth.tar)
5. Change `model_path` for `efficientnet_params` in `local_ond_12_wo_rd_nd_wo_feedback.json` to point the model downloaded in the previous step
6. Change `dataset_root` in `local_ond_12_wo_rd_nd_wo_feedback.json`  to point to `sail-on/images` directory
7. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_ond_12_wo_rd_nd_wo_feedback.json
    ```

#### Running CONDDA 12 With Red Light and Without Feedback
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory condda_12_with_rd
    ```
2. Go to the sail on client repository and make a copy of the configuration file for the algorithm
    ```
      cd sail-on-client
      cp config/condda_12_with_rd_nd.json config/local_condda_12_with_rd_nd.json
    ```
3. Download the evm model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/EVM_cosine_model_umd_b3_CC_tail40000_ct7_dm55.hdf5)
4. Change `model_path` for `evm_params` in `local_condda_12_with_rd_nd.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/trained_efficientnet_b3_CC.pth.tar)
5. Change `model_path` for `efficientnet_params` in `local_condda_12_with_rd_nd.json` to point the model downloaded in the previous step
6. Change `dataset_root` in `local_condda_12_with_rd_nd.json`  to point to directory with images for testing
7. Run the client
    ```
      tinker sail_on_client/protocol/condda.py -i ParInterface -p config/local_condda_12_with_rd_nd.json
    ```

#### Running CONDDA 12 Without Red Light and Without Feedback
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory condda_12_wo_rd
    ```
2. Go to the sail on client repository and make a copy of the configuration file for the algorithm
    ```
      cd sail-on-client
      cp config/condda_12_wo_rd_nd.json config/local_condda_12_wo_rd_nd.json
    ```
3. Download the evm model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/EVM_cosine_model_umd_b3_CC_tail40000_ct7_dm55.hdf5)
4. Change `model_path` for `evm_params` in `local_condda_12_wo_rd_nd.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/trained_efficientnet_b3_CC.pth.tar)
5. Change `model_path` for `efficientnet_params` in `local_condda_12_wo_rd_nd.json` to point the model downloaded in the previous step
6. Change `dataset_root` in `local_condda_12_wo_rd_nd.json`  to point to directory with images for testing
7. Run the client
    ```
      tinker sail_on_client/protocol/condda.py -i ParInterface -p config/local_condda_12_wo_rd_nd.json
    ```

#### Save Attributes for OND 12 Without Red Light and Without Feedback
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory ond_12_without_rd
    ```
2. Go to the sail on client repository and make a copy of the configuration file for the algorithm
    ```
      cd sail-on-client
      cp config/ond_12_without_rd_nd_save.json config/local_ond_12_without_rd_nd_save.json
    ```
3. Download the evm model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/EVM_cosine_model_umd_b3_CC_tail40000_ct7_dm55.hdf5)
4. Change `model_path` for `evm_params` in `local_ond_12_without_rd_nd_save.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/trained_efficientnet_b3_CC.pth.tar)
5. Change `model_path` for `efficientnet_params` in `local_ond_12_without_rd_nd_save.json` to point the model downloaded in the previous step
6. Change `dataset_root` in `local_ond_12_without_rd_nd_save.json`  to point to `sail-on/images` directory
7. Change `save_dir` in `local_ond_12_without_rd_nd_save.json` to a directory where attributes are saved
8. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_ond_12_without_rd_nd_save.json
    ```

#### Restore Attributes for OND 12 Without Red Light and Without Feedback
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory ond_12_without_rd
    ```
2. Go to the sail on client repository and make a copy of the configuration file for the algorithm
    ```
      cd sail-on-client
      cp config/ond_12_without_rd_nd_restore.json config/local_ond_12_without_rd_nd_restore.json
    ```
3. Download the evm model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/EVM_cosine_model_umd_b3_CC_tail40000_ct7_dm55.hdf5)
4. Change `model_path` for `evm_params` in `local_ond_12_without_rd_nd_restore.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://vast.uccs.edu/~adhamija/4kitware/file_to_share/trained_efficientnet_b3_CC.pth.tar)
5. Change `model_path` for `efficientnet_params` in `local_ond_12_without_rd_nd_restore.json` to point the model downloaded in the previous step
6. Change `dataset_root` in `local_ond_12_without_rd_nd_restore.json`  to point to `sail-on/images` directory
7. Change `save_dir` in `local_ond_12_without_rd_nd_restore.json` to a directory where attributes are saved
8. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_ond_12_without_rd_nd_restore.json
    ```

Note: The instructions for running the old algorithms are available in M6-ALGO.md

### Running Transcription Experiments

#### Running HWR based Novelty Detector

1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory hwr_nd_results
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running the algorithm
    ```
      cd sail-on-client
      cp config/hwr_nd.json config/local_hwr_nd.json
      cp config/hwr_config.yaml config/local_hwr_config.yaml
    ```
3. Download the models from following [link](https://drive.google.com/file/d/1Jecja5wYKE-1Pj68KN2P0EStFzjVS2lD/view?usp=sharing
)
4. Extract the model file in the root directory of sail-on-client
   ```
     cd sail-on-client
     tar xvf hwr_novelty_detector_states_dry_run.tar.gz
   ```
5. Change `config_file_path` in `local_hwr_nd.json` to point `local_hwr_config.yaml`.
6. Download the image for dry run using the following [link](https://drive.google.com/file/d/1nUxCqxbr46gnAFSKztnjEj0fYufWRE2f/view)
8. Change `dataset_root` in `local_hwr_nd.json` to point to directory where the images are stored in the previous step
9. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_hwr_nd.json
    ```

### Running Activity Recognition Experiments

#### Running Graph Autoencoder based Novelty Detector Without Red Light

1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory gae_nd_results
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running the algorithm
    ```
      cd sail-on-client
      cp config/gae_nd.json config/local_gae_nd.json
    ```
3. Download the backbone model from following [link](https://drive.google.com/drive/u/0/folders/1ad8gny6Dqvp6hqTRwvTNvhW30lHmf6D2)
4. Change `backbone_weight_path`  and `graph_weight_path` for `feature_extractor_params` in `local_gae_nd.json` to point `rgb_imagenet.pth` and `HMDB51_new_1_model_best.pth.tar`.
5. Download the EVM model from following [link](https://drive.google.com/file/d/1MDV0nFYNYaC19DCBNmDmUaUXiM-amGNs/view?usp=sharing)
6. Change `weight_path` for `evm_params` in `local_gae_nd.json` to point the model downloaded in previous step.
7. Download the HMDB dataset using the following [link](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar)
8. Change `dataset_root` in `local_gae_nd.json` to point to directory where the videos are stored in the previous step
9. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_gae_nd.json
    ```

#### Running Feature Extraction For Graph Autoencoder based Novelty Detector Using Graph Features

1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory gae_nd_results
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running the algorithm
    ```
      cd sail-on-client
      cp config/gae_nd_graph_fe.json.json config/local_gae_nd_graph_fe.json
    ```
3. Download the backbone model from following [link](https://drive.google.com/drive/u/0/folders/1ad8gny6Dqvp6hqTRwvTNvhW30lHmf6D2)
4. Change `backbone_weight_path`  and `graph_weight_path` for `feature_extractor_params` in `local_gae_nd.json` to point `rgb_imagenet.pth` and `TA2_model_best.pth.tar`.
5. Download the EVM model from following [link](https://drive.google.com/file/d/1C1V9bk8NTxSCqncG6yPus3-iuI8kABtp/view?usp=sharing)
6. Change `weight_path` for `evm_params` in `local_gae_nd.json` to point the model downloaded in previous step.
7. Change `dataset_root` in `local_gae_nd.json` to point to directory where the videos are stored
8. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_gae_nd_graph_fe.json
    ```
    This generarates a pickle file for every test present in the config in `GAE-features` directory

#### Running Graph Autoencoder with KL divergence Using Pre-Computed Features Without Red Light

1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory gae_kl_nd_results
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running the algorithm
    ```
      cd sail-on-client
      cp config/gae_nd_graph_precomputed.json config/local_gae_kl_nd_precomputed.json
    ```
3. Download the backbone model from following [link](https://drive.google.com/drive/u/0/folders/1ad8gny6Dqvp6hqTRwvTNvhW30lHmf6D2)
4. Change `backbone_weight_path`  and `graph_weight_path` for `feature_extractor_params` in `local_gae_kl_nd_precomputed.json` to point `rgb_imagenet.pth` and `TA2_model_best.pth.tar`.
5. Download the EVM model from following [link](https://drive.google.com/file/d/1C1V9bk8NTxSCqncG6yPus3-iuI8kABtp/view?usp=sharing)
6. Change `weight_path` for `evm_params` in `local_gae_kl_nd_precomputed.json` to point the model downloaded in previous step.
7. Change `dataset_root` in `local_gae_kl_nd_precomputed.json` to point to directory where the videos are stored.
8. Change `save_dir` to the path where features are stored.
9. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_gae_kl_nd_precomputed.json
    ```

#### Running Graph Autoencoder based Novelty Detector Using Pre-Computed Features Without Red Light

1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory gae_nd_results
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running the algorithm
    ```
      cd sail-on-client
      cp config/gae_nd_graph_precomputed.json config/local_gae_nd_precomputed.json
    ```
3. Download the backbone model from following [link](https://drive.google.com/drive/u/0/folders/1ad8gny6Dqvp6hqTRwvTNvhW30lHmf6D2)
4. Change `backbone_weight_path`  and `graph_weight_path` for `feature_extractor_params` in `local_gae_nd.json` to point `rgb_imagenet.pth` and `TA2_model_best.pth.tar`.
5. Download the EVM model from following [link](https://drive.google.com/file/d/1C1V9bk8NTxSCqncG6yPus3-iuI8kABtp/view?usp=sharing)
6. Change `weight_path` for `evm_params` in `local_gae_nd.json` to point the model downloaded in previous step.
7. Change `dataset_root` in `local_gae_nd.json` to point to directory where the videos are stored.
8. Change `save_dir` to the path where features are stored.
9. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_gae_nd_precomputed.json
    ```

#### Running Feature Extraction For Graph Autoencoder based Novelty Detector Using I3D Features

1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory gae_nd_results
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running the algorithm
    ```
      cd sail-on-client
      cp config/gae_nd_graph_i3d_fe.json.json config/local_gae_nd_i3d_graph_fe.json
    ```
3. Download the backbone model from the following [link](https://drive.google.com/file/d/1P4hhDVf3WFffg0fxQSDLlh3-ToVdhDgD/view?usp=sharing) and graph model from following [link](https://drive.google.com/file/d/1A4khFuZTHAGmQENynVd6ZpDsSrLAEnfl/view?usp=sharing)
4. Change `backbone_weight_path`  and `graph_weight_path` for `feature_extractor_params` in `local_gae_nd.json` to point `rgb_imagenet.pth` and `TA2_model_best.pth.tar`.
5. Download the EVM model from following [link](https://drive.google.com/file/d/1tn4Nb_kNbqQJ3aoKypoJo48yuGR715Ma/view?usp=sharing)
6. Change `weight_path` for `evm_params` in `local_gae_nd.json` to point the model downloaded in previous step.
7. Change `dataset_root` in `local_gae_nd.json` to point to directory where the videos are stored
8. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_gae_nd_i3d_fe.json
    ```
    This generarates a pickle file for every test present in the config in `GAE-features` directory

#### Running Graph Autoencoder based Novelty Detector With Red Light

1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory gae_nd_rd_results
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running the algorithm
    ```
      cd sail-on-client
      cp config/gae_nd_rd.json config/local_gae_nd_rd.json
    ```
3. Download the backbone model from following [link](https://drive.google.com/drive/u/0/folders/1ad8gny6Dqvp6hqTRwvTNvhW30lHmf6D2)
4. Change `backbone_weight_path`  and `graph_weight_path` for `feature_extractor_params` in `local_gae_nd_rd.json` to point `rgb_imagenet.pth` and `HMDB51_new_1_model_best.pth.tar`.
5. Download the EVM model from following [link](https://drive.google.com/file/d/1MDV0nFYNYaC19DCBNmDmUaUXiM-amGNs/view?usp=sharing)
6. Change `weight_path` for `evm_params` in `local_gae_nd_rd.json` to point the model downloaded in previous step.
7. Download the HMDB dataset using the following [link](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar)
8. Change `dataset_root` in `local_gae_nd_rd.json` to point to directory where the videos are stored in the previous step
9. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py.py -i ParInterface -p config/local_gae_nd_rd.json
    ```

#### Running Graph Autoencoder based Novelty Detector Using Pre-Computed Features With Red Light

1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory gae_nd_rd_results
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running the algorithm
    ```
      cd sail-on-client
      cp config/gae_nd_rd_graph_precomputed.json config/local_gae_nd_rd_precomputed.json
    ```
3. Download the backbone model from following [link](https://drive.google.com/drive/u/0/folders/1ad8gny6Dqvp6hqTRwvTNvhW30lHmf6D2)
4. Change `backbone_weight_path`  and `graph_weight_path` for `feature_extractor_params` in `local_gae_nd_rd_precomputed.json` to point `rgb_imagenet.pth` and `TA2_model_best.pth.tar`.
5. Download the EVM model from following [link](https://drive.google.com/file/d/1C1V9bk8NTxSCqncG6yPus3-iuI8kABtp/view?usp=sharing)
6. Change `weight_path` for `evm_params` in `local_gae_nd_rd_precomputed.json` to point the model downloaded in previous step.
7. Change `dataset_root` in `local_gae_nd_rd_precomputed.json` to point to directory where the videos are stored.
8. Change `save_dir` to the path where features are stored.
9. Run the client
    ```
      tinker sail_on_client/protocol/ond_protocol.py -i ParInterface -p config/local_gae_nd_rd_precomputed.json
    ```

## Interpreting Results for Algorithms
The results for the algorithm are stored in in `<results_directory>/<protocol_name>/image_classification`,
where `<results_directory>` is specified by `--results-directory` parameter when the server is executed and `protocol`
name would be either OND or CONDDA.

For OND, every test would have three files. The files follow the following convention `session id.test name.novelty operation.csv`, where
`session_id` is provided by the server, `test name` is specified in the json file and `novelty operation` would have the value detection,
classification and characterization.

## Acknowledgement of Support and Disclaimer

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001120C0055. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the DARPA.
