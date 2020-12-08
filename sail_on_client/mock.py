"""Mocks mainly used for testing protocols."""

from __future__ import annotations

from tinker.basealgorithm import BaseAlgorithm
from sail_on_client.checkpointer import Checkpointer
from typing import Dict, Any, Tuple, Callable

import logging
import torch


class MockDetector(BaseAlgorithm):
    """Mock Detector for testing image classification protocols."""

    def __init__(self, toolset: Dict) -> None:
        """
        Detector constructor.

        Args:
            toolset (dict): Dictionary containing parameters for the constructor
        """
        BaseAlgorithm.__init__(self, toolset)
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
        logging.info(f"Executing {step_descriptor}")
        return self.step_dict[step_descriptor](toolset)

    def _initialize(self, toolset: Dict) -> None:
        """
        Algorithm Initialization.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            None
        """
        pass

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
        return {}, {}

    def _world_detection(self, toolset: str) -> str:
        """
        Detect change in world ( Novelty has been introduced ).

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for change in world
        """
        return self.dataset

    def _novelty_classification(self, toolset: str) -> str:
        """
        Classify data provided in known classes and unknown class.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty classification step
        """
        return self.dataset

    def _novelty_adaption(self, toolset: str) -> None:
        """
        Update models based on novelty classification and characterization.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            None
        """
        pass

    def _novelty_characterization(self, toolset: str) -> str:
        """
        Characterize novelty by clustering different novel samples.

        Args:
            toolset (dict): Dictionary containing parameters for different steps

        Return:
            path to csv file containing the results for novelty characterization step
        """
        return self.dataset


class MockDetectorWithAttributes(MockDetector):
    """Mock Detector for testing checkpointing."""

    def __init__(self, toolset: Dict) -> None:
        """
        Detector constructor.

        Args:
            toolset (dict): Dictionary containing parameters for the constructor
        """
        MockDetector.__init__(self, toolset)

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
        self.dummy_dict = toolset["dummy_dict"]
        self.dummy_list = toolset["dummy_list"]
        self.dummy_tuple = toolset["dummy_tuple"]
        self.dummy_tensor = toolset["dummy_tensor"]
        self.dummy_val = toolset["dummy_val"]
        return {}, {}


class MockAdapterWithCheckpoint(BaseAlgorithm, Checkpointer):
    """Mock Adapter for testing checkpointing."""

    def __init__(self, toolset: Dict) -> None:
        """
        Detector constructor.

        Args:
            toolset (dict): Dictionary containing parameters for the constructor
        """
        BaseAlgorithm.__init__(self, toolset)
        Checkpointer.__init__(self, toolset)
        self.detector = MockDetectorWithAttributes(toolset)

    def execute(self, toolset: Dict, step_descriptor: str) -> Any:
        """
        Execute method used by the protocol to run different steps.

        Args:
            toolset (dict): Dictionary containing parameters for different steps
            step_descriptor (str): Name of the step
        """
        logging.info(f"Executing {step_descriptor}")
        return self.detector.step_dict[step_descriptor](toolset)

    def __eq__(self, other: object) -> bool:
        """
        Overriden method to compare two mock adapters.

        Args:
            other (MockAdapterWithCheckpoint): Another instance of mock adapter

        Return:
            True if both instances have same attributes
        """
        if not isinstance(other, MockAdapterWithCheckpoint):
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
