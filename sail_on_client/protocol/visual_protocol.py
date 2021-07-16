"""Visual Protocol."""

import logging
from typing import Dict, Any, TypeVar, Type

from sail_on_client.harness.test_and_evaluation_harness import TestAndEvaluationHarness
from sail_on_client.agent.visual_agent import VisualAgent
from tinker.protocol import Protocol
from tinker.configuration import smqtk_generator


log = logging.getLogger(__name__)
VisualProtocolType = TypeVar("VisualProtocolType", bound="VisualProtocol")


class VisualProtocol(Protocol):
    """Protocol for visual tasks."""

    def __init__(
        self, algorithms: Dict[str, VisualAgent], harness: TestAndEvaluationHarness
    ) -> None:
        """
        Construct visual protocol.

        Args:
            algorithms: Dictionary of algorithms that are used run based on the protocol
            harness: A harness for test and evaluation

        Returns:
            None
        """
        super().__init__()
        self.algorithms = algorithms
        self.harness = harness

    @classmethod
    def from_config(
        cls: Type[VisualProtocolType], config_dict: Dict, merge_default: bool = True
    ) -> VisualProtocolType:
        """
        Construct protocol from config.

        Args:
            config_dict: dictionary with parameters
            merge_default: Merge dictionary with default values

        Returns:
            An instance of visual protocol
        """
        config_dict = dict(config_dict)
        algorithm_configs = config_dict.get("algorithms", None)
        if not algorithm_configs:
            raise ValueError("No algorithms provided in the config.")
        for algorithm_name, algorithm_config in algorithm_configs.items():
            config_dict[algorithm_name] = smqtk_generator(algorithm_config)

        harness_config = config_dict.get("harness", None)
        if not harness_config:
            raise ValueError("Harness parameters not provided in the config.")
        config_dict["harness"] = smqtk_generator(algorithm_config)
        return super().from_config(config_dict, merge_default=merge_default)

    def get_config(self) -> Dict:
        """
        Get json compliant representation of the protocol.

        Returns:
            Dictionary with json compliant representation of the protocol
        """
        cfg = {}
        for algorithm_name, algorithm in self.algorithms.items():
            cfg[algorithm_name] = algorithm.get_config()
        cfg["harness"] = self.harness.get_config()
        return cfg
