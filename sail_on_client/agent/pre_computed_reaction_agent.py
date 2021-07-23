"""Reaction agent that use precomputed values in the OND protocol."""

from sail_on_client.agent.ond_reaction_agent import ONDReactionAgent
from sail_on_client.agent.pre_computed_detector import PreComputedONDAgent

import logging
from typing import Dict, Any, Tuple

log = logging.getLogger(__name__)


class PreComputedONDReactionAgent(ONDReactionAgent):
    """Detector for submitting precomputed results for computing reaction performance."""

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
        self.detector = PreComputedONDAgent(
            algorithm_name, cache_dir, has_roundwise_file, round_size
        )

    def get_config(self) -> Dict:
        """Return a default configuration dictionary."""
        return {
            "algorithm_name": self.detector.algorithm_name,
            "cache_dir": self.detector.cache_dir,
            "has_roundwise_file": self.detector.has_roundwise_file,
            "round_size": self.detector.round_size,
        }

    def execute(self, toolset: Dict, step_descriptor: str) -> Any:
        """
        Execute method used by the protocol to run different steps associated with the algorithm.

        Args:
            toolset (dict): Dictionary containing parameters for different steps
            step_descriptor (str): Name of the step
        """
        log.info(f"Executing {step_descriptor}")
        return self.detector.step_dict[step_descriptor](toolset)

    def initialize(self, toolset: Dict) -> None:
        """
        Algorithm Initialization.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            None
        """
        self.detector.initialize(toolset)

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
        return self.detector.feature_extraction(toolset)

    def novelty_classification(self, toolset: Dict) -> str:
        """
        Classify data provided in known classes and unknown class.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty classification step
        """
        return self.detector.novelty_classification(toolset)
