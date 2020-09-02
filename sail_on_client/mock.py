from tinker.basealgorithm import BaseAlgorithm
import logging


class MockDetector(BaseAlgorithm):
    """
    Mock detector for testing
    """

    def __init__(self, toolset):
        """
        Constructor for the detector

        Args:
            toolset (dict): Dictionary containing parameters for the constructor
        """
        BaseAlgorithm.__init__(self, toolset)
        self.step_dict = {
            "Initialize": self._initialize,
            "FeatureExtraction": self._feature_extraction,
            "WorldDetection": self._world_detection,
            "NoveltyClassification": self._novelty_classification,
            "NoveltyAdaption": self._novelty_adaption,
            "NoveltyCharacterization": self._novelty_characterization,
        }

    def execute(self, toolset, step_descriptor):
        """
        Execute method used by the protocol to run different steps associated with
        the algorithm

        Args:
            toolset (dict): Dictionary containing parameters for different steps
            step_descriptor (str): Name of the step

        Return:

        """
        logging.info(f"Executing {step_descriptor}")
        return self.step_dict[step_descriptor](toolset)

    def _initialize(self, toolset):
        """
        Initialization for the algorithm

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            None
        """
        pass

    def _feature_extraction(self, toolset):
        """
        Feature extraction step for the algorithm

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            Tuple of dictionary
        """
        self.dataset = toolset["dataset"]
        return {}, {}

    def _world_detection(self, toolset):
        """
        Detect change in world ( Novelty has been introduced )
        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for change in world
        """
        return self.dataset

    def _novelty_classification(self, toolset):
        """
        Classify data provided in known classes and unknown class

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty classification step
        """
        return self.dataset

    def _novelty_adaption(self, toolset):
        """
        Update models based on novelty classification and characterization

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            None
        """
        pass

    def _novelty_characterization(self, toolset):
        """
        Characterize novelty by clustering different novel samples

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty characterization step
        """
        return self.dataset
