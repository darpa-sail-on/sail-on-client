"""Configuration module for CONDDA."""

import scriptconfig as scfg


class ConddaConfig(scfg.Config):
    """
    Default configuration for CONDDA protocol.

    Example:
        >>> from sail_on_client.protocol.condda_config import ConddaConfig
        >>> config = ConddaConfig()
        >>> print('config = {!r}'.format(config))
    """

    default = {
        "domain": scfg.Value("image_classification"),
        "test_ids": ["CONDDA.2.1.293"],
        "novelty_detector_class": scfg.Value("CONDDA_5_14_A1"),
        "seed": scfg.Value("seed"),
        "dataset_root": "",
        "feature_extraction_only": scfg.Value(
            False, help="Quit after feature extraction"
        ),
        "save_features": scfg.Value(False, help="Save features as pkl file"),
        "use_saved_features": scfg.Value(False, help="Use features saved the pkl file"),
        "save_dir": scfg.Value("", help="Directory where features are saved"),
        "save_attributes": scfg.Value(False, help="Flag to attributes in save dir"),
        "use_saved_attributes": scfg.Value(
            False, help="Use attributes saved in save dir"
        ),
        "saved_attributes": {},
        "skip_stage": [],
        "hints": [],
        "detector_config": {
            "efficientnet_params": {"model_path": "", "known_classes": 413},
            "evm_params": {
                "model_path": "",
                "known_feature_path": "",
                "tailsize": 33998,
                "cover_threshold": 0.7,
                "distance_multiplier": 0.55,
                "number_of_unknown_to_crate_evm": 3,
            },
            "dataloader_params": {"batch_size": 128, "num_workers": 3},
            "characterization_param": {
                "clustering_type": "FINCH",
                "number_of_unknown_to_strat_clustering": 20,
            },
            "csv_folder": "",
            "cores": 4,
            "detection_threshold": 0.5,
        },
    }
