from framework.baseprotocol import BaseProtocol

from ond_config import OndConfig
from sail_on_client.errors import RoundError
from importlib_metadata import version
from itertools import count
import os
import json
import sys
import logging

class SailOn( BaseProtocol ):




    def __init__( self, discovered_plugins, algorithmsdirectory, harness, config_file ):
        BaseProtocol.__init__(self, discovered_plugins, algorithmsdirectory, harness, config_file)
        if not os.path.exists(config_file):
            logging.error(f"{config_file} does not exist")
            sys.exit(1)

        with open(config_file, 'r') as f:
            overriden_config = json.load(f)
        self.config = OndConfig(overriden_config)

    def run_protocol(self):

        # provide all of the configuration information in the toolset
        self.toolset.update(self.config)

        novelty_algorithm = self.get_algorithm(self.config['novelty_detector_class'], self.toolset)

        #TODO: fix the version below
        #novelty_detector_version = version(novelty_algorithm.__module__)
        novelty_detector_version ='1.0.0'

        self.toolset['session_id'] = self.test_harness.session_request(
                self.config['test_ids'], "OND",
                "%s.%s" % (self.config['novelty_detector_class'], novelty_detector_version ) )
        session_id = self.toolset['session_id']

        logging.info( f"New session: {self.toolset['session_id']}" )

        for test in self.config['test_ids']:
            self.metadata = self.test_harness.get_test_metadata(test)
            self.toolset['test_id'] = test
            self.toolset['test_type'] = ""
            self.toolset['metadata'] = self.test_harness.get_test_metadata( test )
            if "red_light" in self.toolset['metadata']:
                self.toolset['redlight_image'] = os.path.join(
                            self.toolset['dataset_root'],
                            self.toolset['metadata']["red_light"] )
            else:
                self.toolset['redlight_image'] = ""
            novelty_algorithm.execute(self.toolset, "Initialize")
            self.toolset['image_features'] = {}
            self.toolset['dataset_root'] = self.config['dataset_root']
            self.toolset['dataset_ids'] = list()

            logging.info( f"Start test: {self.toolset['test_id']}" )

            for round_id in count(0):
                self.toolset['round_id'] = round_id

                logging.info( f"Start round: {self.toolset['round_id']}" )
                # see if there is another round available
                try:
                    self.toolset['dataset'] = self.test_harness.dataset_request( test, round_id,
                                                                                 session_id )
                except RoundError:
                    # no more rounds available, this test is done.
                    break

                self.toolset['features_dict'], self.toolset['logit_dict'] = \
                        novelty_algorithm.execute(self.toolset, "FeatureExtraction")

                results = dict()

                results['detection'] = \
                        novelty_algorithm.execute(self.toolset, "WorldDetection")

                results['classification'] = \
                        novelty_algorithm.execute(self.toolset, "NoveltyClassification")

                self.test_harness.post_results( results, test, round_id, session_id )
                with open(self.toolset['dataset'], "r") as dataset:
                    self.toolset['dataset_ids'].extend( dataset.readlines() )
                logging.info( f"Round complete: {self.toolset['round_id']}")

                #cleanup the round files
                os.remove(self.toolset['dataset'])
                os.remove(results['detection'])
                os.remove(results['classification'])
                    

            results = dict()

            self.toolset['dataset_ids'] = [image_id.strip() for image_id in self.toolset['dataset_ids']]

            results['characterization'] = novelty_algorithm.execute(self.toolset, "NoveltyCharacterization")
            if results['characterization'] is not None and os.path.exists(results['characterization']):
                self.test_harness.post_results( results, test, 0, session_id )
            logging.info( f"Test complete: {self.toolset['test_id']}" )

            #cleanup the characterization file
            os.remove(results['characterization'])

        logging.info( f"Session ended: {self.toolset['session_id']}" )
        self.test_harness.terminate_session(session_id)
