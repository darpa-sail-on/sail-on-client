"""Agents that use precomputed values in the protocol."""

from sail_on_client.agent.visual_agent import VisualAgent
from sail_on_client.agent.ond_agent import ONDAgent
from sail_on_client.agent.condda_agent import CONDDAAgent
from typing import Dict, Any, Tuple, Callable

import logging
import os
import pandas as pd

log = logging.getLogger(__name__)


class PreComputedAgent(VisualAgent):
    """Detector for submitting precomputed results."""

    def __init__(
        self,
        algorithm_name: str,
        cache_dir: str,
        has_roundwise_file: bool,
        round_size: int,
    ) -> None:
        """
        Construct agent with precomputed results.

        Args:
            algorithm_name: Name of the algorithm
            cache_dir: Path to cache directory
            has_roundwise_file: Flag to determine if the cache has files for rounds
            round_size: Size of a round
        """
        ONDAgent.__init__(self)
        self.algorithm_name = algorithm_name
        self.cache_dir = cache_dir
        self.has_roundwise_file = has_roundwise_file
        self.round_size = round_size
        self.step_dict: Dict[str, Callable] = {
            "Initialize": self.initialize,
            "FeatureExtraction": self.feature_extraction,
            "WorldDetection": self.world_detection,
        }

    def get_config(self) -> Dict:
        """Return a default configuration dictionary."""
        return {
            "algorithm_name": self.algorithm_name,
            "cache_dir": self.cache_dir,
            "has_roundwise_file": self.has_roundwise_file,
            "round_size": self.round_size,
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

    def initialize(self, toolset: Dict) -> None:
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

    def feature_extraction(
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
        pd.read_csv(self.dataset, header=None).shape[0]
        return {}, {}

    def world_detection(self, toolset: Dict) -> str:
        """
        Detect change in world ( Novelty has been introduced ).

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for change in world
        """
        return self._generate_step_result(toolset, "detection")


class PreComputedONDAgent(PreComputedAgent, ONDAgent):
    """Detector for submitting precomputed results in OND."""

    def __init__(
        self,
        algorithm_name: str,
        cache_dir: str,
        has_roundwise_file: bool,
        round_size: int,
    ) -> None:
        """
        Construct agent with precomputed results for OND.

        Args:
            algorithm_name: Name of the algorithm
            cache_dir: Path to cache directory
            has_roundwise_file: Flag to determine if the cache has files for rounds
            round_size: Size of a round
        """
        PreComputedAgent.__init__(
            self, algorithm_name, cache_dir, has_roundwise_file, round_size
        )
        ONDAgent.__init__(self)
        self.step_dict.update(
            {
                "NoveltyClassification": self.novelty_classification,
                "NoveltyAdaptation": self.novelty_adaptation,
                "NoveltyCharacterization": self.novelty_characterization,
            }
        )

    def novelty_classification(self, toolset: Dict) -> str:
        """
        Classify data provided in known classes and unknown class.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty classification step
        """
        return self._generate_step_result(toolset, "classification")

    def novelty_adaptation(self, toolset: Dict) -> None:
        """
        Update models based on novelty classification and characterization.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            None
        """
        pass

    def novelty_characterization(self, toolset: Dict) -> str:
        """
        Characterize novelty by clustering different novel samples.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty characterization step
        """
        return self._get_result_path(toolset, "characterization")


class PreComputedCONDDAAgent(PreComputedAgent, CONDDAAgent):
    """Detector for submitting precomputed results in CONDDA."""

    def __init__(
        self,
        algorithm_name: str,
        cache_dir: str,
        has_roundwise_file: bool,
        round_size: int,
    ) -> None:
        """
        Construct agent with precomputed results for CONDDA.

        Args:
            algorithm_name: Name of the algorithm
            cache_dir: Path to cache directory
            has_roundwise_file: Flag to determine if the cache has files for rounds
            round_size: Size of a round
        """
        PreComputedAgent.__init__(
            self, algorithm_name, cache_dir, has_roundwise_file, round_size
        )
        ONDAgent.__init__(self)
        self.step_dict.update(
            {"NoveltyCharacterization": self.novelty_characterization}
        )

    def novelty_characterization(self, toolset: Dict) -> str:
        """
        Characterize novelty by clustering different novel samples.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty characterization step
        """
        return self._get_result_path(toolset, "characterization")
