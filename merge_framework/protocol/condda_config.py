import scriptconfig as scfg


class ConddaConfig(scfg.Config):
    """
    Default configuration for CONDDA protocol

    Example:
        >>> from merge_framework.protocol.condda_config import ConddaConfig
        >>> config = ConddaConfig()
        >>> print('config = {!r}'.format(config))
    """
    default = {
        'domain':  scfg.Value("image_classification"),
        'test_ids': [
                        'CONDDA.2.1.293',
        ],
        'novelty_detector_class': scfg.Value("CONDDA_5_14_A1"),
        'seed': scfg.Value("seed"),
        'dataset_root': "",
        'detector_config':
        {
            'efficientnet_params':
            {
                "model_path": "",
                "known_classes": 413
            },
            'evm_params':
            {
                "model_path": "",
                "known_feature_path": "",
                "tailsize": 33998,
                "cover_threshold": 0.7,
                "distance_multiplier": 0.55,
                "number_of_unknown_to_strat_clustering": 15,
                "number_of_unknown_to_crate_evm": 3
            },
            'dataloader_params':
            {
                "batch_size": 128,
                "num_workers": 3
            },
            'finch_params': {},
            'pfact_params': {},
            'hdbscan_params': {},
            'csv_folder': '',
            'cores': 4,
            'detection_threshold': 0.3
        }
    }




