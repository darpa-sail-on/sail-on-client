"""Client implementation for local interface."""

from sail_on.api.file_provider import FileProvider
from sail_on.api.file_provider import get_session_info
from sail_on.api.errors import RoundError
from sail_on_client.errors import RoundError as ClientRoundError
from tinker.harness import Harness

from tempfile import TemporaryDirectory
from typing import Any, Dict
import os
import shutil


class LocalInterface(Harness):
    """Interface without any server communication."""

    def __init__(self, config_file: str, config_folder: str) -> None:
        """
        Initialize an object of local interface.

        Args:
            config_file: Name of the config file that provides parameter
                         to the interface
            config_folder: The directory where configfile is present
        Returns:
            None
        """
        Harness.__init__(self, config_file, config_folder)
        self.temp_dir = TemporaryDirectory()
        self.data_dir = self.configuration_data["data_dir"]
        self.result_directory = self.temp_dir.name
        self.file_provider = FileProvider(self.data_dir, self.result_directory)

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
        result_dict = self.file_provider.test_ids_request(
            protocol, domain, detector_seed, test_assumptions
        )
        return result_dict["test_ids"]

    def session_request(
        self,
        test_ids: list,
        protocol: str,
        domain: str,
        novelty_detector_version: str,
        hints: list,
    ) -> str:
        """
        Create a new session to evaluate the detector using an empirical protocol.

        Args:
            test_ids   : list of tests being evaluated in this session
            protocol   : string indicating which protocol is being evaluated
            domain     : string indicating which domain is being evaluated
            novelty_detector_version : string indicating the version of the novelty detector being evaluated
            hints      : Hints used for the session
        Returns:
            A session identifier provided by the server
        """
        return self.file_provider.new_session(
            test_ids, protocol, domain, novelty_detector_version, hints
        )

    def dataset_request(self, test_id: str, round_id: int, session_id: str) -> str:
        """
        Request data for evaluation.

        Args:
            test_id    : the test being evaluated at this moment.
            round_id   : the sequential number of the round being evaluated
            session_id : the identifier provided by the server for a single experiment

        Returns:
            Filename of a file containing a list of image files (including full path for each)
        """
        try:
            self.data_file = os.path.join(
                self.result_directory, f"{session_id}.{test_id}.{round_id}.csv"
            )
            byte_stream = self.file_provider.dataset_request(
                session_id, test_id, round_id
            )
            with open(self.data_file, "wb") as f:
                f.write(byte_stream.getbuffer())
            return self.data_file
        except RoundError as r:
            raise ClientRoundError(
                reason=r.reason, msg=r.msg, stack_trace=r.stack_trace
            )

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
            feedback_ids   : List of media ids for which feedback is required
            feedback_type  : protocols constants with the values: label, detection, characterization
            test_id        : the id of the test currently being evaluated
            round_id       : the sequential number of the round being evaluated
            session_id     : the id provided by a server denoting a session

        Returns:
            Path to a file containing containing requested feedback
        """
        self.feedback_file = os.path.join(
            self.result_directory,
            f"{session_id}.{test_id}.{round_id}_{feedback_type}.csv",
        )
        byte_stream = self.file_provider.get_feedback(
            feedback_ids, feedback_type, session_id, test_id, round_id
        )
        with open(self.feedback_file, "wb") as f:
            f.write(byte_stream.getbuffer())
        return self.feedback_file

    def post_results(
        self, result_files: Dict[str, str], test_id: str, round_id: int, session_id: str
    ) -> None:
        """
        Post client detector predictions for the dataset.

        Args:
            result_files : A dictionary of results with protocol constant as key and file path as value
            test_id        : the id of the test currently being evaluated
            round_id       : the sequential number of the round being evaluated
            session_id     : the id provided by a server denoting a session

        Returns:
            None
        """
        info = get_session_info(str(self.result_directory), session_id)
        protocol = info["activity"]["created"]["protocol"]
        domain = info["activity"]["created"]["domain"]
        base_result_path = os.path.join(str(self.result_directory), protocol, domain)
        os.makedirs(base_result_path, exist_ok=True)
        for result_key in result_files.keys():
            file_name = f"{session_id}.{test_id}_{result_key}.csv"
            dst_path = os.path.join(str(base_result_path), file_name)
            shutil.copy(result_files[result_key], dst_path)
        self.file_provider.post_results(session_id, test_id, round_id, result_files)

    def evaluate(self, test_id: str, round_id: int, session_id: str) -> str:
        """
        Get results for test(s).

        Args:
            test_id        : the id of the test currently being evaluated
            round_id       : the sequential number of the round being evaluated
            session_id     : the id provided by a server denoting a session

        Returns:
            Path to a file with the results
        """
        return self.file_provider.evaluate(session_id, test_id, round_id)

    def get_test_metadata(self, session_id: str, test_id: str) -> Dict[str, Any]:
        """
        Retrieve the metadata json for the specified test.

        Args:
            session_id        : the id of the session currently being evaluated
            test_id           : the id of the test currently being evaluated

        Returns:
            A json file containing metadata
        """
        return self.file_provider.get_test_metadata(session_id, test_id)

    def terminate_session(self, session_id: str) -> None:
        """
        Terminate the session after the evaluation for the protocol is complete.

        Args:
            session_id     : the id provided by a server denoting a session

        Returns: None
        """
        self.file_provider.terminate_session(session_id)
