"""Configuration module for OND protocol."""

import scriptconfig as scfg


class OndConfig(scfg.Config):
    """
    Default configuration for OND protocol.

    Example:
        >>> from sail_on_client.protocol.ond_config import OndConfig
        >>> config = OndConfig()
        >>> print('config = {!r}'.format(config))
    """

    default = {
        "domain": scfg.Value("image_classification"),
        "test_ids": ["OND.1.1.293"],
        "seed": scfg.Value("seed"),
        "dataset_root": "/home/eric/sail-on/images",
        "feature_extraction_only": scfg.Value(
            False, help="Quit after feature extraction"
        ),
        "use_feedback": scfg.Value(False, help="Use feedback for the run"),
        "feedback_type": scfg.Value("classification", help="Type of feedback"),
        "save_features": scfg.Value(False, help="Save features as pkl file"),
        "use_saved_features": scfg.Value(False, help="Use features saved the pkl file"),
        "use_consolidated_features": scfg.Value(
            False, help="Use features consolidated over multiple tests"
        ),
        "save_dir": scfg.Value("", help="Directory where features are saved"),
        "save_attributes": scfg.Value(False, help="Flag to attributes in save dir"),
        "use_saved_attributes": scfg.Value(
            False, help="Use attributes saved in save dir"
        ),
        "save_elementwise": scfg.Value(False, help="Save attributes elementwise"),
        "is_eval_enabled": scfg.Value(False, help="Flag to enable evaluate"),
        "is_eval_roundwise_enabled": scfg.Value(
            False, help="Flag to enable roundwise evaluate"
        ),
        "saved_attributes": {},
        "skip_stages": [],
        "hints": [],
        "resume_session": scfg.Value(False, help="Flag to resume session"),
        "resumed_session_ids": {},
        "detectors": {
            "has_baseline": False,
            "has_reaction_baseline": False,
            "baseline_class": None,
            "csv_folder": "",
            "cores": 6,
            "detection_threshold": 0.1,
            "detector_configs": {
                "gae_kl_nd": {
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
                    "kl_params": {
                        "window_size": 100,
                        "mu_train": 1.0,
                        "sigma_train": 0.0057400320777888664,
                        "KL_threshold": 5.830880886275709,
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
                "baseline_i3d": {
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
