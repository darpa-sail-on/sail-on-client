from framework.baseprotocol import BaseProtocol
from condda_config import ConddaConfig
from merge_framework.errors import RoundError
from itertools import count
import os
import json
import sys


class Condda(BaseProtocol):
    def __init__(self, discovered_plugins, algorithmsdirectory, harness, config_file):
        BaseProtocol.__init__(self, discovered_plugins, algorithmsdirectory, harness, config_file)
        if not os.path.exists(config_file):
            print(f"{config_file} does not exist", file=sys.stderr)
            sys.exit(1)

        with open(config_file, 'r') as f:
            overriden_config = json.load(f)
        self.config = ConddaConfig(overriden_config)

    def run_protocol(self):
        # provide all of the configuration information in the toolset
        self.toolset.update(self.config)
        novelty_algorithm = self.get_algorithm(self.config['novelty_detector_class'], self.toolset)
        # TODO: fix the version below
        # novelty_detector_version = version(novelty_algorithm.__module__)
        novelty_detector_version = '1.0.0'
        novelty_detector_cv = (f"{self.config['novelty_detector_class']}"
                               f"{novelty_detector_version}")
        self.toolset['session_id'] = \
                self.test_harness.session_request(self.config['test_ids'],
                                                  "CONDDA",
                                                  novelty_detector_cv)
        session_id = self.toolset['session_id']
        print("New session:", self.toolset['session_id'])
        for test_id in self.config['test_ids']:
            self.metadata = self.test_harness.get_test_metadata(test_id)
            self.toolset['test_id'] = test_id
            self.toolset['test_type'] = ""
            self.toolset['metadata'] = self.metadata
            if "red_light" in self.metadata:
                self.toolset['red_light_image'] = \
                        self.toolset['metadata']["red_light"]
            else:
                self.toolset['red_light_image'] = ""
            novelty_algorithm.execute(self.toolset, "Initialize")
            self.toolset['image_features'] = {}
            self.toolset['dataset_root'] = self.config['dataset_root']
            self.toolset['dataset_ids'] = list()
            print("Start test:", self.toolset['test_id'])
            for round_id in count(0):
                self.toolset['round_id'] = round_id
                print("Start round:", self.toolset['round_id'])
                # see if there is another round available
                try:
                    self.toolset['dataset'] = \
                            self.test_harness.dataset_request(test_id,
                                                              round_id,
                                                              session_id )
                except RoundError:
                    # no more rounds available, this test is done.
                    break
                self.toolset['features_dict'], self.toolset['logit_dict'] = \
                        novelty_algorithm.execute(self.toolset,
                                                  "FeatureExtraction")
                results = dict()
                results['detection'] = \
                        novelty_algorithm.execute(self.toolset,
                                                  "WorldDetection")
                results['characterization'] = \
                        novelty_algorithm.execute(self.toolset,
                                                  "NoveltyCharacterization")
                self.test_harness.post_results(results,
                                               test_id,
                                               round_id,
                                               session_id )
                print("Round complete:", self.toolset['round_id'])
                novelty_algorithm.execute(self.toolset,
                                          "NoveltyAdaption")
                #cleanup the round files
                os.remove(results['detection'])
                os.remove(results['characterization'])
            print( "Test complete:", self.toolset['test_id'] )
        print( "Session ended:", self.toolset['session_id'] )
        self.test_harness.terminate_session(session_id)
