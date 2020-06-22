# Sail-On Client and Protocols

Client and Protocols for DARPA sail-on

### Protocols present in the repository
1. [OND](https://drive.google.com/file/d/1W2Ex4-eQl1CrAbv67fAN0OJL8kmRtvt2/view?usp=sharing)
2. [CONDDA](https://drive.google.com/file/d/1sIDLTgXivaguVfSp3g1qfe7sqiUcvFLA/view?usp=sharing)

### Client present in the repository
1. [Par Interface](https://gitlab.kitware.com/darpa-sail-on/merge_framework/-/blob/master/merge_framework/protocol/parinterface.py)


## Requirements

1. [Python 3.7](https://www.python.org/downloads/release/python-370/)
2. [pipenv](https://pipenv.pypa.io/en/latest/)
3. [Framework](https://gitlab.kitware.com/darpa_learn/framework)
4. [Script Config](https://pypi.org/project/scriptconfig)
5. [TA-2 Agent](https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector)
6. [Sail-On](https://gitlab.kitware.com/darpa-sail-on/sail-on)

## Installation
The installation requires cloning and installing multiple repositories. Thus after
following the instructions for a repository, please move back to your
your working directory.

### Install Framework
1. Clone the [framework](https://gitlab.kitware.com/darpa_learn/framework) repository
   ```
   git clone https://gitlab.kitware.com/darpa_learn/framework.git
   ```
   This would create a directory called framework in your working directory

2. Use the development version of [framework](https://gitlab.kitware.com/darpa_learn/framework)
   ```
   cd framework
   git checkout development
   ```

3. Install the dependencies of the framework in a virtual environment
   ```
   pipenv install
   pipenv shell
   ```
   This would create a virtual environment and activate the environment. Please use this virtual environment for installing all other repositories.

4. Install the framework in the virtual environment
   ```
   pip install -e .
   ```

### Install Sail-On Server
1. Clone the [sail_on](https://gitlab.kitware.com/darpa-sail-on/sail-on) repository
   ```
    git clone https://gitlab.kitware.com/darpa-sail-on/sail-on.git
   ```
   This would create a directory called sail-on in your working directory

2. Go into the sail-on directory and install the dependencies for the server using
   ```
    cd sail-on
    pip install -r requirements.txt
   ```

3. Install the server
   ```
    pip install -e .
   ```

### Install TA2 Agent
1. Clone [TA2 agent](https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector) repository
   ```
    git clone https://gitlab.kitware.com/darpa-sail-on/evm_based_novelty_detector.git
   ```
   This would create a directory called evm_based_novelty_detector in your working directory

2. Go into the directory for TA2 agent and install the dependencies for the server using
   ```
    cd evm_based_novelty_detector
    pip install -r requirements.txt
    pip install -e timm
   ```
3. Install agent using
    ```
      pip install -e .
    ```

### Install Sail-On Client
1. Clone the [merge_framework](https://gitlab.kitware.com/darpa-sail-on/merge_framework) repository
   ```
    git clone https://gitlab.kitware.com/darpa-sail-on/merge_framework.git
   ```
   This would create a directory called merge_framework in your working directory

2. Go into the merge_framework directory and install the dependencies for the client using
   ```
    cd merge_framework
    pip install -r requirements.txt
   ```

3. Install the client using
   ```
    pip install -e .
   ```

## Running Client and Server with Different Algorithms
Before running the client or server, activate the virtual environment from the framework repository using
```
cd framework
pipenv shell
```

### Running OND_5_14_A1 Algorithm
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory ond_5_14_a1
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running OND_5_14_A1
    ```
      cd merge_framework
      cp config/ond_5_14_a1_nd.json config/local_ond_5_14_a1_nd.json
    ```
3. Download the evm model from following [link](https://drive.google.com/file/d/1XrSWQWJsF-iPkvGM4AWkMNqvhFTb0yfk/view?usp=sharing)
4. Change `model_path` for `evm_params` in `local_ond_5_14_a1_nd.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://drive.google.com/file/d/1esL1W7pDHrsTmLpSFxWdzOg6oP-p8IDi/view?usp=sharing)
5. Change `model_path` for `efficientnet_params` in `local_ond_5_14_a1_nd.json` to point the model downloaded in the previous step
6. Change `dataset_root` in `local_ond_5_14_a1_nd.json`  to point to `sail-on/images` directory
7. Run the client
    ```
      framework merge_framework/protocol/ond_protocol.py -i ParInterface -p config/local_ond_5_14_a1_nd.json
    ```

### Running OND_5_14_A2 Algorithm
1. Go to sail-on server directory and start the server using
    ```
      cd sail-on
      sail_on_server --data-directory data/ --results-directory ond_5_14_a2
    ```
2. Go to the sail on client repository and make a copy of the configuration file for running OND_5_14_A2
    ```
      cd merge_framework
      cp config/ond_5_14_a2_nd.json config/local_ond_5_14_a2_nd.json
    ```
3. Download the evm model from following [link](https://drive.google.com/file/d/1XrSWQWJsF-iPkvGM4AWkMNqvhFTb0yfk/view?usp=sharing)
4. Change `model_path` for `evm_params` in `local_ond_5_14_a2_nd.json` to point the model downloaded in the previous step
3. Download the efficientnet model from following [link](https://drive.google.com/file/d/1esL1W7pDHrsTmLpSFxWdzOg6oP-p8IDi/view?usp=sharing)
5. Change `model_path` for `efficientnet_params` in `local_ond_5_14_a2_nd.json` to point the model downloaded in the previous step
6. Change `dataset_root` in `local_ond_5_14_a1_nd.json`  to point to `sail-on/images` directory
7. Run the client
    ```
      framework merge_framework/protocol/ond_protocol.py -i ParInterface -p config/local_ond_5_14_a2_nd.json
    ```

