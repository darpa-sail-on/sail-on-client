"""Mocks mainly used for testing protocols."""

from sailon_tinker_launcher.deprecated_tinker.basealgorithm import BaseAlgorithm
from typing import Dict, Any, Tuple, Callable

import logging
import os
import pandas as pd

log = logging.getLogger(__name__)


class PreComputedDetector(BaseAlgorithm):
    """Detector for submitting precomputed results."""

    def __init__(self, toolset: Dict) -> None:
        """
        Detector constructor.

        Args:
            toolset (dict): Dictionary containing parameters for the constructor
        """
        BaseAlgorithm.__init__(self, toolset)
        self.cache_dir = toolset["cache_dir"]
        self.has_roundwise_file = toolset["has_roundwise_file"]
        self.algorithm_name = toolset["algorithm_name"]
        self.step_dict: Dict[str, Callable] = {
            "Initialize": self._initialize,
            "FeatureExtraction": self._feature_extraction,
            "WorldDetection": self._world_detection,
            "NoveltyClassification": self._novelty_classification,
            "NoveltyAdaption": self._novelty_adaption,
            "NoveltyCharacterization": self._novelty_characterization,
        }

    def execute(self, toolset: Dict, step_descriptor: str) -> Any:
        """
        Execute method used by the protocol to run different steps associated with the algorithm.

        Args:
            toolset (dict): Dictionary containing parameters for different steps
            step_descriptor (str): Name of the step
        """
        log.info(f"Executing {step_descriptor}")
        return self.step_dict[step_descriptor](toolset)

    def _initialize(self, toolset: Dict) -> None:
        """
        Algorithm Initialization.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            None
        """
        self.round_idx = {"detection": 0, "classification": 0}
        self.test_id = toolset["test_id"]

    def _get_test_info_from_toolset(self, toolset: Dict) -> Tuple:
        """
        Private function for getting test id and round id (optionally) from toolset.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            tuple containing test id and round id (optionally)
        """
        if self.has_roundwise_file:
            return (self.test_id, toolset["round_id"])
        else:
            return self.test_id

    def _get_result_path(self, toolset: Dict, step_descriptor: str) -> str:
        """
        Private function for getting path to results.

        Args:
            toolset (dict): Dictionary containing parameters for different steps
            step_descriptor (str): Name of the step

        Return:
            Path to the result file
        """
        if self.has_roundwise_file:
            test_id, round_id = self._get_test_info_from_toolset(toolset)
            return os.path.join(
                self.cache_dir,
                f"{test_id}.{round_id}_{self.algorithm_name}_{step_descriptor}.csv",
            )
        else:
            test_id = self._get_test_info_from_toolset(toolset)
            return os.path.join(
                self.cache_dir, f"{test_id}_{self.algorithm_name}_{step_descriptor}.csv"
            )

    def _generate_step_result(self, toolset: Dict, step_descriptor: str) -> str:
        result_path = self._get_result_path(toolset, step_descriptor)
        if self.has_roundwise_file:
            return result_path
        else:
            round_file_path = os.path.join(
                self.cache_dir, f"{self.algorithm_name}_{step_descriptor}.csv"
            )
            round_idx = self.round_idx[step_descriptor]
            test_df = pd.read_csv(result_path, header=None)
            round_df = test_df.iloc[round_idx : round_idx + self.round_size]
            self.round_idx[step_descriptor] += self.round_size
            round_df.to_csv(round_file_path, index=False, header=False)
            return round_file_path

    def _feature_extraction(
        self, toolset: Dict
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Feature extraction step for the algorithm.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            Tuple of dictionary
        """
        self.dataset = toolset["dataset"]
        self.round_size = pd.read_csv(self.dataset, header=None).shape[0]
        return {}, {}

    def _world_detection(self, toolset: Dict) -> str:
        """
        Detect change in world ( Novelty has been introduced ).

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for change in world
        """
        return self._generate_step_result(toolset, "detection")

    def _novelty_classification(self, toolset: Dict) -> str:
        """
        Classify data provided in known classes and unknown class.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty classification step
        """
        return self._generate_step_result(toolset, "classification")

    def _novelty_adaption(self, toolset: Dict) -> None:
        """
        Update models based on novelty classification and characterization.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            None
        """
        pass

    def _novelty_characterization(self, toolset: Dict) -> str:
        """
        Characterize novelty by clustering different novel samples.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty characterization step
        """
        return self._get_result_path(toolset, "characterization")
