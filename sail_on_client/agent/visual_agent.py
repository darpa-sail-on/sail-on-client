"""Abstract interface for an agent that works for visual agent."""

from tinker.algorithm import Algorithm
from typing import Dict, Tuple
from abc import abstractmethod


class VisualAgent(Algorithm):
    """Abstract class for Visual Agent."""

    def get_config(self):
        """Return a default configuration dictionary."""
        return {}

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
    def world_detection(self, wd_toolset: Dict) -> str:
        """
        Abstract method for detecting that the world has changed.

        Args:
            wd_toolset: Parameters for feature extraction

        Returns:
            Path to results for detecting that the world has changed.
        """
        pass
