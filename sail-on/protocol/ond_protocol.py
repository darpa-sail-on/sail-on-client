from framework.localinterface import LocalInterface
from framework.baseprotocol import BaseProtocol

from ond_config import OndConfig
from importlib_metadata import version
import inspect
import datetime

class SailOn( BaseProtocol ):




    def __init__( self, algorithmsdirectory, harness ):
        BaseProtocol.__init__(self, algorithmsdirectory, harness)

        self.config = OndConfig()

    def run_protocol(self):

        # provide all of the configuration information in the toolset 
        self.toolset.update(self.config)

        novelty_algorithm = self.get_algorithm(self.config['novelty_detector_class'], self.toolset)

        #TODO: fix the version below
        #novelty_detector_version = version(novelty_algorithm.__module__)
        novelty_detector_version ='1.0.0'


        import pdb
        pdb.set_trace()

        self.toolset['session_id'] = self.test_harness.session_request( 
                self.config['test_ids'], "OND",
                "%s.%s" % (self.config['novelty_detector_class'], novelty_detector_version ) )

        for test in self.config['test_ids']:
            self.toolset['test_id'] = test
            self.toolset['round_id'] = None
            self.toolset['dataset'] = self.test_harness.dataset_request( test, 0)

            novelty_algorithm.execute(self.toolset, "Initialize")

            self.toolset['image_features'] = novelty_algorithm.execute(self.toolset, "FeatureExtraction")

            results = dict()

            results['novelty_detections'] = novelty_algorithm.execute(self.toolset, "NoveltyDetection")

            results['novelty_characterization'] = novelty_algorithm.execute(self.toolset, "NoveltyCharacterization")

            self.test_harness.post_results( results, test, 0 )

        self.test_harness.terminate_session()
        

            




