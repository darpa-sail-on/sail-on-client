"""Dataclasses for CONDDA."""

from dataclasses import dataclass
import logging
from typing import Dict, List, Any
from sail_on_client.utils.utils import merge_dictionaries

try:
    from importlib.metadata import version, PackageNotFoundError  # type:ignore
except ModuleNotFoundError:
    from importlib_metadata import version, PackageNotFoundError

log = logging.getLogger(__name__)


@dataclass
class AlgorithmAttributes:
    """Class for storing attributes of algorithm present in the protocol."""

    name: str
    detection_threshold: float
    instance: Any
    package_name: str
    parameters: Dict
    session_id: str
    test_ids: List[str]

    def named_version(self) -> str:
        """
        Compute version of an algorithm.

        Returns:
            A string containing name and the version number
        """
        try:
            if self.package_name:
                version_number = version(self.package_name)
            else:
                log.warn("No package_name provided. Using 0.0.1 as stand in.")
                version_number = "0.0.1"
        except PackageNotFoundError:
            log.warn(
                "Failed to detect the version of the algorithm. Using 0.0.1 as stand in."
            )
            version_number = "0.0.1"
        return f"{self.name}-{version_number}"

    def remove_completed_tests(self, finished_tests: List[str]) -> None:
        """
        Remove finished tests from test_ids.

        Args:
            finished_tests: List of tests that are complete

        Returns:
            None
        """
        test_set = set(self.test_ids)
        ftest_set = set(finished_tests)
        self.test_ids = list(test_set ^ ftest_set)

    def merge_detector_params(
        self, detector_params: Dict, exclude_keys: List = None
    ) -> None:
        """
        Merge common parameters with algorithm specific parameters with exclusions.

        Args:
            detector_params: Dictionary of common parameters
            exclude_keys: List of keys that should be excluded in the merge

        Returns:
            None
        """
        if not exclude_keys:
            exclude_keys = []
        self.parameters = merge_dictionaries(
            self.parameters, detector_params, exclude_keys
        )


@dataclass
class InitializeParams:
    """Class for storing parameters that are used to initialize the algorithm."""

    parameters: Dict
    session_id: str
    test_id: str

    def get_toolset(self) -> Dict:
        """
        Convert the data present in the class into a dictionary.

        Returns
            A dictionary with data associated with the class
        """
        algorithm_toolset = self.parameters.copy()
        algorithm_toolset["session_id"] = self.session_id
        algorithm_toolset["test_id"] = self.test_id
        algorithm_toolset["test_type"] = ""
        return algorithm_toolset


@dataclass
class NoveltyCharacterizationParams:
    """Class for storing parameters associated novelty characterization with an algorithm."""

    features_dict: Dict
    logit_dict: Dict
    round_id: int

    def get_toolset(self) -> Dict:
        """
        Convert the data present in the class into a dictionary.

        Returns
            A dictionary with data associated with the class
        """
        return {
            "features_dict": self.features_dict,
            "logit_dict": self.logit_dict,
            "round_id": self.round_id,
        }
