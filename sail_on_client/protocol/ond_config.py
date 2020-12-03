"""Configuration module for OND protocol."""

import scriptconfig as scfg


class OndConfig(scfg.Config):
    """
    Default configuration for Ond protocol.

    Example:
        >>> from learn.protocol.learn_config import LearnConfig
        >>> config = LearnConfig()
        >>> print('config = {!r}'.format(config))
    """

    default = {
        "domain": scfg.Value("image_classification"),
        "test_ids": ["OND.1.1.293"],
        "novelty_detector_class": scfg.Value("OND_5_14_A1"),
        "seed": scfg.Value("seed"),
        "dataset_root": "/home/eric/sail-on/images",
        "feature_extraction_only": scfg.Value(
            False, help="Quit after feature extraction"
        ),
        "use_feedback": scfg.Value(False, help="Use feedback for the run"),
        "save_features": scfg.Value(False, help="Save features as pkl file"),
        "use_saved_features": scfg.Value(False, help="Use features saved the pkl file"),
        "save_dir": scfg.Value("", help="Directory where features are saved"),
        "save_attributes": scfg.Value(False, help="Flag to attributes in save dir"),
        "use_saved_attributes": scfg.Value(
            False, help="Use attributes saved in save dir"
        ),
        "save_elementwise": scfg.Value(False, help="Save attributes elementwise"),
        "saved_attributes": {},
        "skip_stage": [],
        "hints": [],
        "detector_config": {
            "efficientnet_params": {
                "model_path": "/home/eric/merge_framework/sail_on/protocol/trained_efficientnet_b3_fp16.pth.tar",
                "known_classes": 413,
            },
            "evm_params": {
                "model_path": "/home/eric/merge_framework/sail_on/protocol/efficientb3_EVM_model_tail33998_ct7_dm55.hdf5",  # noqa: E501
                "tailsize": 33998,
                "cover_threshold": 0.7,
                "distance_multiplier": 0.55,
            },
            "known_kmeans_params": {},
            "dataloader_params": {"batch_size": 64, "num_workers": 3},
            "csv_folder": "",
            "cores": 4,
            "detection_threshold": 0.3,
        },
    }
