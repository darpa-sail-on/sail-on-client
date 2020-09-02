from framework.basealgorithm import BaseAlgorithm
import logging

class MockDetector(BaseAlgorithm):
    def __init__(self, toolset):
        BaseAlgorithm.__init__(self, toolset)
        self.step_dict = {
                "Initialize": self._initialize,
                "FeatureExtraction": self._feature_extraction,
                "WorldDetection": self._world_detection,
                "NoveltyClassification": self._novelty_classification,
                "NoveltyAdaption": self._novelty_adaption,
                "NoveltyCharacterization": self._novelty_characterization
                }

    def execute(self, toolset, step_descriptor):
        logging.info(f"Executing {step_descriptor}")
        return self.step_dict[step_descriptor](toolset)

    def _initialize(self, toolset):
        pass

    def _feature_extraction(self, toolset):
        self.dataset = toolset["dataset"]
        return {}, {}

    def _world_detection(self, toolset):
        return self.dataset

    def _novelty_classification(self, toolset):
        return self.dataset

    def _novelty_adaption(self, toolset):
        pass

    def _novelty_characterization(self, toolset):
        return self.dataset
