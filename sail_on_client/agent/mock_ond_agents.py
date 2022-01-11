"""Mocks mainly used for testing protocols."""

from sail_on_client.checkpointer import Checkpointer
from sail_on_client.agent.ond_agent import ONDAgent

from typing import Dict, Any, Tuple, Callable

import logging
import os
import shutil
import torch

log = logging.getLogger(__name__)


class MockONDAgent(ONDAgent):
    """Mock Detector for OND Protocol."""

    def __init__(self) -> None:
        """Construct Mock OND Detector."""
        super().__init__()
        self.step_dict: Dict[str, Callable] = {
            "Initialize": self.initialize,
            "FeatureExtraction": self.feature_extraction,
            "WorldDetection": self.world_detection,
            "NoveltyClassification": self.novelty_classification,
            "NoveltyAdaption": self.novelty_adaptation,
            "NoveltyCharacterization": self.novelty_characterization,
        }

    def initialize(self, toolset: Dict) -> None:
        """
        Algorithm Initialization.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            None
        """
        pass

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
        return {}, {}

    def world_detection(self, toolset: Dict) -> str:
        """
        Detect change in world ( Novelty has been introduced ).

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for change in world
        """
        dataset_dir = os.path.dirname(self.dataset)
        dst_file = os.path.join(dataset_dir, "wc.csv")
        shutil.copyfile(self.dataset, dst_file)
        return dst_file

    def novelty_classification(self, toolset: Dict) -> str:
        """
        Classify data provided in known classes and unknown class.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty classification step
        """
        dataset_dir = os.path.dirname(self.dataset)
        dst_file = os.path.join(dataset_dir, "ncl.csv")
        shutil.copyfile(self.dataset, dst_file)
        return dst_file

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
        dataset_dir = os.path.dirname(self.dataset)
        dst_file = os.path.join(dataset_dir, "nc.csv")
        shutil.copyfile(self.dataset, dst_file)
        return dst_file

    def execute(self, toolset: Dict, step_descriptor: str) -> Any:
        """
        Execute method used by the protocol to run different steps.

        Args:
            toolset (dict): Dictionary containing parameters for different steps
            step_descriptor (str): Name of the step
        """
        log.info(f"Executing {step_descriptor}")
        return self.step_dict[step_descriptor](toolset)


class MockONDAgentWithAttributes(MockONDAgent):
    """Mock Detector for testing checkpointing."""

    def __init__(self) -> None:
        """
        Detector constructor.

        Args:
            toolset (dict): Dictionary containing parameters for the constructor
        """
        MockONDAgent.__init__(self)

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
        self.dummy_dict = toolset["dummy_dict"]
        self.dummy_list = toolset["dummy_list"]
        self.dummy_tuple = toolset["dummy_tuple"]
        self.dummy_tensor = toolset["dummy_tensor"]
        self.dummy_val = toolset["dummy_val"]
        return {}, {}


class MockONDAdapterWithCheckpoint(Checkpointer, MockONDAgent):
    """Mock Adapter for testing checkpointing."""

    def __init__(self, toolset: Dict) -> None:
        """
        Detector constructor.

        Args:
            toolset (dict): Dictionary containing parameters for the constructor
        """
        MockONDAgent.__init__(self)
        Checkpointer.__init__(self, toolset)
        self.detector = MockONDAgentWithAttributes()

    def get_config(self) -> Dict:
        """
        Get config for the plugin.

        Returns:
            Parameters for the agent
        """
        config = super().get_config()
        config.update(self.toolset)
        return config

    def execute(self, toolset: Dict, step_descriptor: str) -> Any:
        """
        Execute method used by the protocol to run different steps.

        Args:
            toolset (dict): Dictionary containing parameters for different steps
            step_descriptor (str): Name of the step
        """
        log.info(f"Executing {step_descriptor}")
        return self.detector.step_dict[step_descriptor](toolset)

    def __eq__(self, other: object) -> bool:
        """
        Overriden method to compare two mock adapters.

        Args:
            other (MockONDAdapterWithCheckpoint): Another instance of mock adapter

        Return:
            True if both instances have same attributes
        """
        if not isinstance(other, MockONDAdapterWithCheckpoint):
            return NotImplemented

        return (
            self.detector.dummy_dict == other.detector.dummy_dict
            and self.detector.dummy_list == other.detector.dummy_list
            and self.detector.dummy_tuple == other.detector.dummy_tuple
            and bool(
                torch.all(
                    torch.eq(self.detector.dummy_tensor, other.detector.dummy_tensor)
                )
            )
            and self.detector.dummy_val == other.detector.dummy_val
        )
