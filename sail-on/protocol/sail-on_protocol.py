from framework.localinterface import LocalInterface
from framework.baseprotocol import BaseProtocol

from sail-on_config import SailOnConfig

class SailOn( LocalInterface, BaseProtocol ):




    def __init__( self, algorithmsdirectory ):
        BaseProtocol.__init__(self, algorithmsdirectory)
        LocalInterface.__init__(self, 'localinterface.json' )

        self.config = SailOnConfig()

    def run_protocol(self):
        task_ids = self.get_task_ids()


        for test in task_ids:
            self.initialize_session(test)
            #TODO: setup the toolset to construct the algo
            novelty_algorithm = self.get_algorithm(self.config['novelty_detector_class'], self.toolset)

            #TODO: setup the toolset to initialize the algorithm
            novelty_algorithm.execute(self.toolset, "Initialize")

            #TODO: select the dataset to use
            self.toolset['dataset'] = self.get_dataset('sailon', self.config['dataset'])
            
            novelty_algorithm.execute(self.toolset, "Initialize")

            self.toolset['image_features'] = novelty_algorithm.execute(self.toolset, "FeatureExtraction")

            results = dict()

            results['novelty_detections'] = novelty_algorithm.execute(self.toolset, "NoveltyDetection")

            results['novelty_characterization'] = novelty_algorithm.execute(self.toolset, "NoveltyCharacterization")

            self.post_results( toolset['dataset'], results )

            self.terminate_session()

            




