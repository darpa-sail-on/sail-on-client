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
            'OND.47392.000000.8901',
            'OND.47392.000002.8901',
            'OND.47392.000003.8901',
            'OND.47392.000004.8901',
            'OND.47392.001000.8901',
            'OND.47392.001001.8901',
            'OND.47392.001002.8901',
            'OND.47392.001003.8901',
            'OND.47392.001004.8901',
            'OND.47392.002000.8901',
            'OND.47392.002001.8901',
            'OND.47392.002002.8901',
            'OND.47392.002003.8901',
            'OND.47392.002004.8901',
            'OND.47392.003000.8901',
            'OND.47392.003001.8901',
            'OND.47392.003002.8901',
            'OND.47392.003003.8901',
            'OND.47392.003004.8901',
            'OND.47392.004000.8901',
            'OND.47392.004001.8901',
            'OND.47392.004002.8901',
            'OND.47392.004003.8901',
            'OND.47392.004004.8901',
            ],

        'novelty_detector_class': scfg.Value("OND_5_14_A1"),

        'seed': scfg.Value("seed"),

        'dataset_root': "/data/datasets/TA1-image-classification/dataset_v1",

        'detector_config': {

            'resnet_params': {},

            'efficientnet_params': {
                "model_path": "/home/khq.kitware.com/ameya.shringi/models/efficientnet-models/trained_efficientnet_b3_fp16.pth.tar",
                "known_classes": 413
                },

            'evm_params': {
                "model_path": "/home/khq.kitware.com/ameya.shringi/models/evm-gpu/efficientb3_EVM_model_tail33998_ct7_dm55.hdf5",
                "tailsize": 33998,
                "cover_threshold": 0.7,
                "distance_multiplier": 0.55
                },
    
            'known_kmeans_params': {},

            'dataloader_params': {
                "batch_size": 128,
                "num_workers": 3
                },

            'csv_folder': '',

            'cores': 5,

            'detection_threshold': 0.3,
            }

    }




