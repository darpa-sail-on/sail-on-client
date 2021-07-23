"""Abstract interface for a baseline that is used to compute reaction performance."""

from smqtk_core import Configurable, Pluggable

from typing import Dict, Tuple, Any
from abc import abstractmethod


class ONDReactionAgent(Configurable, Pluggable):
    """Abstract class for OND Reaction Agent."""

    @classmethod
    def is_usable(cls) -> bool:
        """Determine if this class with be detected by SMQTK's plugin."""
        return True

    def get_config(self) -> Dict:
        """Return a default configuration dictionary."""
        return {}

    @abstractmethod
    def execute(self, toolset: Dict, step_descriptor: str) -> Any:
        """
        Execute method used by the protocol to run different steps associated with the algorithm.

        Args:
            toolset (dict): Dictionary containing parameters for different steps
            step_descriptor (str): Name of the step
        """
        pass

    @abstractmethod
    def feature_extraction(self, fe_toolset: Dict) -> Tuple[Dict, Dict]:
        """
        Abstract method for feature extraction.

        Args:
            fe_toolset: Parameters for feature extraction

        Returns:
            Tuple of dictionary with features and logits.
        """
        pass

    @abstractmethod
    def novelty_classification(self, ncl_toolset: Dict) -> str:
        """
        Abstract method for detecting that the world has changed.

        Args:
            ncl_toolset: Parameters for novelty classification

        Returns:
            Path to results for sample wise classification.
        """
        pass
