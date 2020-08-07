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
        "detector_config": {
            "resnet_params": {},
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
