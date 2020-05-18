import scriptconfig as scfg


class OndConfig(scfg.Config):
    """
    Default configuration for Ond protocol

    Example:
        >>> from learn.protocol.learn_config import LearnConfig
        >>> config = LearnConfig()
        >>> print('config = {!r}'.format(config))
    """
    default = {
        'domain':  scfg.Value("image_classification"),

        'test_ids': [
            'OND.1.1.1234',
            ],

        'novelty_detector_class': scfg.Value("EVMBasedNoveltyDetector"),

        'seed': scfg.Value("seed"),

        'detector_config': {

            'resnet_params': {},

            'efficientnet_params': {
                "model_path": "",
                "known_classes": 413
                },

            'evm_params': {
                "model_path": "",
                "tailsize": 33998,
                "cover_threshold": 0.7,
                "distance_multiplier": 0.55
                },
    
            'known_kmeans_params': {},

            'dataloader_params': {
                "batch_size": 1024,
                "num_workers": 8
                },

            'csv_folder': scfg.Path(''),

            'cores': scfg.Value(6),

            'detection_threshold': scfg.Value(0.3),
            }

    }




