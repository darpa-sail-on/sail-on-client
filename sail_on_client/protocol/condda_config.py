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
        "detectors": {
            "has_baseline": False,
            "has_reaction_baseline": False,
            "baseline_class": None,
            "csv_folder": "",
            "cores": 6,
            "detection_threshold": 0.1,
            "detector_configs": {
                "CONDDA_5_14_A1": {
                    "feature_extractor_params": {
                        "backbone_weight_path": "",
                        "name": "i3d",
                        "arch": "i3d-50",
                        "graph_weight_path": "",
                        "model_name": "i3d",
                        "n_classes": 400,
                        "no_cuda": "False",
                        "hidden_dims": [512, 128],
                        "hidden": "True",
                        "in_dim": 1024,
                        "num_heads": [4, 1],
                        "sample_duration": 64,
                        "graph_classes": 88,
                        "mode": "feature",
                        "feature_type": "graph",
                    },
                    "evm_params": {
                        "weight_path": "",
                        "number_of_unknown_to_crate_evm": 7,
                    },
                    "characterization_params": {
                        "clustering_type": "FINCH",
                        "number_of_unknown_to_strat_clustering": 50,
                    },
                    "dataloader_params": {
                        "sample_size": 224,
                        "mean": [114.7748, 107.7354, 99.4750],
                        "sample_duration": 64,
                        "batch_size": 1,
                        "n_threads": 6,
                        "n_classes": 88,
                    },
                },
            },
        },
        "harness_config": {
            "url": "http://3.32.8.161:5001/",
            "data_location": "",
            "data_dir": "",
            "gt_dir": "",
            "gt_config": "",
        },
    }
