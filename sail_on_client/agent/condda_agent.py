"""Abstract interface for an agent that works under CONDDA."""

from sail_on_client.agent.visual_agent import VisualAgent
from typing import Dict
from abc import abstractmethod


class CONDDAAgent(VisualAgent):
    """Abstract class for OND agent."""

    def get_config(self) -> Dict:
        """Return a default configuration dictionary."""
        return {}

    @abstractmethod
    def novelty_characterization(self, nc_toolset: Dict) -> str:
        """
        Abstract method for novelty novelty_characterization.

        Args:
            nc_toolset: Parameters for feature extraction

        Returns:
            Path to results for novelty characterization.
        """
        pass
