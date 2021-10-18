"""Abstract harness for T&E."""

from smqtk_core import Configurable, Pluggable
from abc import abstractmethod
from typing import List, Dict, Any, TypeVar

TestAndEvaluationHarnessType = TypeVar(
    "TestAndEvaluationHarnessType", bound="TestAndEvaluationHarness"
)


class TestAndEvaluationHarness(Configurable, Pluggable):
    """Abstract interface for test and evaluation harness."""

    @classmethod
    def is_usable(cls) -> bool:
        """Determine if this class with be detected by SMQTK's plugin."""
        return True

    def get_config(self) -> Dict:
        """Return a default configuration dictionary."""
        return {}

    @abstractmethod
    def test_ids_request(
        self,
        protocol: str,
        domain: str,
        detector_seed: str,
        test_assumptions: str = "{}",
    ) -> str:
        """
        Request Test Identifiers as part of a series of individual tests.

        Args:
            protocol : string indicating which protocol is being evaluated
            domain : problem domain for the tests
            detector_seed : A seed provided by the novelty detector
            test_assumptions : Assumptions used by the detector

        Returns:
            Filename of file containing test ids
        """
        pass

    @abstractmethod
    def session_request(
        self,
        test_ids: list,
        protocol: str,
        domain: str,
        novelty_detector_version: str,
        hints: list,
        detection_threshold: float,
    ) -> str:
        """
        Create a new session to evaluate the detector using an empirical protocol.

        Args:
            test_ids: List of tests being evaluated in this session
            protocol: String indicating which protocol is being evaluated
            domain: String indicating which domain is being evaluated
            novelty_detector_version: The novelty detector being evaluated
            hints: Hints used for the session
            detection_threshold: Detection threshold for the session

        Returns:
            A session identifier provided by the server
        """
        pass

    @abstractmethod
    def resume_session(self, session_id: str) -> List[str]:
        """
        Get finished test from an existing session.

        Args:
            session id : Session id that was started but not terminated

        Returns:
            List of tests finished in the session
        """
        pass

    @abstractmethod
    def dataset_request(self, test_id: str, round_id: int, session_id: str) -> str:
        """
        Request data for evaluation.

        Args:
            test_id: The test being evaluated at this moment.
            round_id: The sequential number of the round being evaluated
            session_id: The identifier provided by the server for a single experiment

        Returns:
            Filename of a file containing a list of image files (including full path for each)
        """
        pass

    @abstractmethod
    def get_feedback_request(
        self,
        feedback_ids: list,
        feedback_type: str,
        test_id: str,
        round_id: int,
        session_id: str,
    ) -> str:
        """
        Get Labels from the server based provided one or more example ids.

        Args:
            feedback_ids: List of media ids for which feedback is required
            feedback_type: Protocols constants with the values: label, detection, characterization
            test_id: The id of the test currently being evaluated
            round_id: The sequential number of the round being evaluated
            session_id: The id provided by a server denoting a session

        Returns:
            Path to a file containing containing requested feedback
        """
        pass

    @abstractmethod
    def post_results(
        self, result_files: Dict[str, str], test_id: str, round_id: int, session_id: str
    ) -> None:
        """
        Post client detector predictions for the dataset.

        Args:
            result_files: A dictionary of results with protocol constant as key and file path as value
            test_id: The id of the test currently being evaluated
            round_id: The sequential number of the round being evaluated
            session_id: The id provided by a server denoting a session

        Returns:
            None
        """
        pass

    @abstractmethod
    def evaluate_round_wise(
        self, test_id: str, round_id: int, session_id: str,
    ) -> Dict[str, Any]:
        """
        Get results for round(s).

        Args:
            test_id: The id of the test currently being evaluated
            round_id: The sequential number of the round being evaluated
            session_id: The id provided by a server denoting a session

        Returns:
            Path to a file with the results
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        test_id: str,
        round_id: int,
        session_id: str,
        baseline_session_id: str = None,
    ) -> Dict:
        """
        Get results for test(s).

        Args:
            test_id: The id of the test currently being evaluated
            round_id: The sequential number of the round being evaluated
            session_id: The id provided by a server denoting a session

        Returns:
            Path to a file with the results
        """
        pass

    @abstractmethod
    def get_test_metadata(self, session_id: str, test_id: str) -> Dict[str, Any]:
        """
        Retrieve the metadata json for the specified test.

        Args:
            session_id: The id of the session currently being evaluated
            test_id: The id of the test currently being evaluated

        Returns:
            A dictionary containing metadata
        """
        pass

    @abstractmethod
    def complete_test(self, session_id: str, test_id: str) -> None:
        """
        Mark test as completed.

        Args:
            session_id: The id of the session currently being evaluated
            test_id: The id of the test currently being evaluated

        Returns:
            None
        """
        pass

    @abstractmethod
    def terminate_session(self, session_id: str) -> None:
        """
        Terminate the session after the evaluation for the protocol is complete.

        Args:
            session_id: The id provided by a server denoting a session

        Returns: None
        """
        pass
