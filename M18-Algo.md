## Instructions For Running Experiments With M18 Algorithms

Note: Installation instructions in all sections assumes that you have cloned and
installed sail-on-client along with its dependencies

### Image Classification Experiment

#### Installation
1. Clone the following repository
  ```
    git clone https://github.com/darpa-sail-on/evm_based_novelty_detector.git
  ```
   This would create evm_based_novelty_detector in your working directory.

2. Activate the sail-on-client environment
  ```
    cd sail-on-client
    poetry shell
  ```

3. Install evm_based_novelty_detector
  ```
    cd ../evm_based_novelty_detector
    pip install -r requirements.txt
    pip install -e timm
    pip install -e .
  ```

#### Running Algorithms

1. Download the feature extractor and evm model using
  ```
    wget https://vast.uccs.edu/~mjafarzadeh/trained_f1_umd_464.pth
    wget https://vast.uccs.edu/~mjafarzadeh/evm_cosine_F1_umd_464_tail_40000_ct_0.8_dm_0.6.pkl
  ```

2. Modify problem configs
  Set the following variables in config/m18/image_classification/given_detection/ic_ond18.yaml
  and config/m18/image_classification/system_detection/ic_ond18.yaml
    1. save_dir: path to directory where all artifacts for the run are stored
    3. test_ids: List of tests
    4. dataset_root: Root directory for images
    5. model_path (cnn_params): Path to the feature extractor model
    6. model_path (evm_params): Path to the evm model

3. Run the detectors
  ```
    tinker -c config/m18/image_classification/given_detection/ic_ond18.yaml sail_on_client/protocol/ond_protocol.py
    tinker -c config/m18/image_classification/system_detection/ic_ond18.yaml sail_on_client/protocol/ond_protocol.py
  ```


### Document Transcription Experiment

#### Installation
1. Clone the following repository
  ```
    git clone https://github.com/darpa-sail-on/evm_based_novelty_detector.git
      git clone https://github.com/darpa-sail-on/hwr_novelty_detector.git
  ```
   This would create evm_based_novelty_detector, hwr_novelty_detector in your working directory.

2. Activate the sail-on-client environment
  ```
    cd sail-on-client
    poetry shell
  ```

3. Install evm_based_novelty_detector
  ```
    cd ../evm_based_novelty_detector
    pip install -r requirements.txt
    pip install -e timm
    pip install -e .
  ```

4. Install hwr_novelty_detector
  ```
    cd ../hwr_novelty_detector
    pip install -r requirements.txt
    pip install -e .
  ```

#### Running Algorithms

1. Download the model from [this link](https://drive.google.com/file/d/1YJMAGS97zHC0cBkNirEcNLdJK0YhTGU0/view?usp=sharing)

2. Untar and move the folder containing the models to config/m18/document_transcription

3. Modify problem configs config/m18/document_transcription/m18_hwr_nd.yaml
    1. save_dir: path to directory where all artifacts for the run are stored
    2. test_ids: List of tests
    3. dataset_root: Root directory for images

3. Run the detectors
  ```
    tinker -c config/m18/document_transcription/m18_hwr_nd.yaml sail_on_client/protocol/ond_protocol.py
  ```


### Activity Recognition Experiment

#### Installation
1. Clone the following repository
  ```
    git clone https://github.com/darpa-sail-on/evm_based_novelty_detector.git
    git clone https://github.com/darpa-sail-on/activity-recognition-models.git
  ```
   This would create evm_based_novelty_detector and activity-recognition-models in your working directory.

2. Activate the sail-on-client environment
  ```
    cd sail-on-client
    poetry shell
  ```

3. Install evm_based_novelty_detector
  ```
    cd ../evm_based_novelty_detector
    pip install -r requirements.txt
    pip install -e timm
    pip install -e .
  ```

4. Install activity-recognition-models
  ```
    cd ../activity-recognition-models
    pip install -r requirements.txt
    pip install -e .
  ```

#### Running Algorithms

1. Download the models from [this link](https://drive.google.com/drive/folders/1JDPNO7m4BbDZZk7VOZY01-VZYjXFCT66?usp=sharing)

2. Modify problem configs
  Set the following variables in config/m18/activity_recognition/given_detection/ar_x3d_rd.yaml
  and config/m18/activity_recognition/system_detection/ar_adaptive_x3d.yaml
    1. save_dir: path to directory where all artifacts for the run are stored
    3. test_ids: List of tests
    4. dataset_root: Root directory for images
    5. model_path (cnn_params): Path to the feature extractor model
    6. model_path (evm_params): Path to the evm model

3. Run the detectors
  ```
    tinker -c config/m18/activity_recognition/given_detection/ar_x3d_rd.yaml sail_on_client/protocol/ond_protocol.py
    tinker -c config/m18/activity_recognition/system_detection/ar_adaptive_x3d.yaml sail_on_client/protocol/ond_protocol.py
  ```


