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

        'dataset_root': "/home/eric/sail-on/images",

        'detector_config': {

            'resnet_params': {},

            'efficientnet_params': {
                "model_path": "/home/eric/merge_framework/sail-on/protocol/trained_efficientnet_b3_fp16.pth.tar",
                "known_classes": 413
                },

            'evm_params': {
                "model_path": "/home/eric/merge_framework/sail-on/protocol/efficientb3_EVM_model_tail33998_ct7_dm55.hdf5",
                "tailsize": 33998,
                "cover_threshold": 0.7,
                "distance_multiplier": 0.55
                },
    
            'known_kmeans_params': {},

            'dataloader_params': {
                "batch_size": 128,
                "num_workers": 8
                },

            'csv_folder': '',

            'cores': 6,

            'detection_threshold': 0.3,
            }

    }




