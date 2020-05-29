from framework.localinterface import LocalInterface
from framework.baseprotocol import BaseProtocol

from ond_config import OndConfig
from importlib_metadata import version
from itertools import count
import os

class SailOn( BaseProtocol ):




    def __init__( self, discovered_plugins, algorithmsdirectory, harness ):
        BaseProtocol.__init__(self, discovered_plugins, algorithmsdirectory, harness)

        self.config = OndConfig()

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

        for test in self.config['test_ids']:
            self.metadata = self.test_harness.get_test_metadata(test)
            self.toolset['test_id'] = test
            self.toolset['test_type'] = ""
            novelty_algorithm.execute(self.toolset, "Initialize")
            self.toolset['image_features'] = {}
            self.toolset['dataset_root'] = self.config['dataset_root']
            self.toolset['dataset_ids'] = list()

            for round_id in count(0):
                self.toolset['round_id'] = round_id

                # see if there is another round available
                try:
                    self.toolset['dataset'] = self.test_harness.dataset_request( test, round_id)
                except:
                    # no more rounds available, this test is done.
                    break

                self.toolset['features_dict'], self.toolset['logit_dict'] = \
                        novelty_algorithm.execute(self.toolset, "FeatureExtraction")

                results = dict()

                results['detection'] = \
                        novelty_algorithm.execute(self.toolset, "WorldDetection")

                results['classification'] = \
                        novelty_algorithm.execute(self.toolset, "NoveltyClassification")
                
                self.test_harness.post_results( results, test, round_id )
                with open(self.toolset['dataset'], "r") as dataset:
                    self.toolset['dataset_ids'].extend( dataset.readlines() )

            results = dict()
            self.toolset['dataset_ids'] = [image_id.strip() for image_id in self.toolset['dataset_ids']]

            results['characterization'] = novelty_algorithm.execute(self.toolset, "NoveltyCharacterization")
            if results['characterization'] is not None and os.path.exists(results['characterization']):
                self.test_harness.post_results( results, test, 0 )

        self.test_harness.terminate_session()
        

