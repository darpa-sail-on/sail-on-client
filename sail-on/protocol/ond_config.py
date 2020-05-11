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

        'novelty_detector_class': scfg.Value("RandomNoveltyAdapter"),

        'seed': scfg.Value("seed"),

        'detector_config': {

            'resnet_params': {},

            'evm_params': {},
    
            'known_kmeans_params': {},

            'dataloader_params': {},

            'csv_folder': scfg.Path('dummy_folder'),

            'cores': scfg.Value(1),

            'detection_threshold': scfg.Value(50.3), }

    }




