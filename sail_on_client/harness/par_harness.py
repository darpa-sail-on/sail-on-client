"""Implementation of T&E Harness for PAR Server."""

import requests
import json
import io
import os
import traceback
import logging

from sail_on_client.harness.test_and_evaluation_harness import TestAndEvaluationHarness
from typing import Any, Dict, Union, List
from requests import Response
from sail_on_client.errors import ApiError, RoundError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    before_sleep_log,
)
from json import JSONDecodeError

log = logging.getLogger(__name__)


class ParHarness(TestAndEvaluationHarness):
    """Harness for PAR server."""

    def __init__(self, url: str, save_directory: str) -> None:
        """
        Initialize a client connection object.

        Args:
            url: URL for the server
            save_directory: A directory to save files

        Returns:
            None
        """
        TestAndEvaluationHarness.__init__(self)
        self.url = url
        self.save_directory = save_directory

    def get_config(self) -> Dict:
        """JSON Compliant representation of the object."""
        return {"url": self.url, "save_directory": self.save_directory}

    def _check_response(self, response: Response) -> None:
        """
        Parse errors that present in the server response.

        Args:
            response: The response object obtained from the server

        Returns:
            None
        """
        if response.status_code != 200:
            try:
                response_json = response.json()
                # Find the appropriate error class based on error code.
                for subclass in ApiError.error_classes():
                    if subclass.error_code == response.status_code:
                        raise subclass(
                            response_json["reason"],
                            response_json["message"],
                            response_json["stack_trace"],
                        )
            except JSONDecodeError:
                log.exception(f"Server Error: {traceback.format_exc()}")
                exit(1)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.INFO),
    )
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
        payload = {
            "protocol": protocol,
            "domain": domain,
            "detector_seed": detector_seed,
        }

        with open(test_assumptions, "r") as f:
            contents = f.read()

        response = requests.get(
            f"{self.url}/test/ids",
            files={
                "test_requirements": io.StringIO(json.dumps(payload)),
                "test_assumptions": io.StringIO(contents),
            },
        )

        self._check_response(response)

        filename = os.path.abspath(
            os.path.join(
                self.save_directory, f"{protocol}.{domain}.{detector_seed}.csv"
            )
        )
        with open(filename, "w") as f:
            f.write(response.content.decode("utf-8"))

        return filename

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.INFO),
    )
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
        payload = {
            "protocol": protocol,
            "novelty_detector_version": novelty_detector_version,
            "domain": domain,
            "hints": hints,
            "detection_threshold": detection_threshold,
        }

        ids = "\n".join(test_ids) + "\n"

        response = requests.post(
            f"{self.url}/session",
            files={"test_ids": ids, "configuration": io.StringIO(json.dumps(payload))},
        )

        self._check_response(response)
        return response.json()["session_id"]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.INFO),
    )
    def resume_session(self, session_id: str) -> List[str]:
        """
        Get finished test from an existing session.

        Args:
            session id : Session id that was started but not terminated

        Returns:
            List of tests finished in the session
        """
        params: Dict[str, str] = {"session_id": session_id}
        response = requests.get(f"{self.url}/session/latest", params=params)
        self._check_response(response)
        return response.json()["finished_tests"]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.INFO),
    )
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
        params: Dict[str, Union[str, int]] = {
            "session_id": session_id,
            "test_id": test_id,
            "round_id": round_id,
        }
        response = requests.get(f"{self.url}/session/dataset", params=params,)
        if response.status_code == 204:
            raise RoundError("End of Dataset", "The entire dataset has been requested")
        self._check_response(response)

        filename = os.path.abspath(
            os.path.join(self.save_directory, f"{session_id}.{test_id}.{round_id}.csv")
        )
        with open(filename, "wb") as f:
            f.write(response.content)
        return filename

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.INFO),
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
            feedback_ids: List of media ids for which feedback is required
            feedback_type: Protocols constants with the values: label, detection, characterization
            test_id: The id of the test currently being evaluated
            round_id: The sequential number of the round being evaluated
            session_id: The id provided by a server denoting a session

        Returns:
            Path to a file containing containing requested feedback
        """
        params: Dict[str, Union[str, int]] = {
            "feedback_ids": "|".join(feedback_ids),
            "session_id": session_id,
            "test_id": test_id,
            "feedback_type": feedback_type,
        }
        response = requests.get(f"{self.url}/session/feedback", params=params,)
        self._check_response(response)
        filename = os.path.abspath(
            os.path.join(
                self.save_directory,
                f"{session_id}.{test_id}.{round_id}_{feedback_type}.csv",
            )
        )
        with open(filename, "wb") as f:
            f.write(response.content)

        return filename

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.INFO),
    )
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
        payload = {
            "session_id": session_id,
            "test_id": test_id,
            "round_id": round_id,
            "result_types": "|".join(result_files.keys()),
        }

        files = {"test_identification": io.StringIO(json.dumps(payload))}

        if len(result_files.keys()) == 0:
            raise Exception("Must provide at least one result file")

        for r_type in result_files:
            with open(result_files[r_type], "r") as f:
                contents = f.read()
                files[f"{r_type}_file"] = io.StringIO(contents)

        response = requests.post(f"{self.url}/session/results", files=files)

        self._check_response(response)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.INFO),
    )
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
        raise NotImplementedError("Round wise accuracy computation is not supported")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.INFO),
    )
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
        params: Dict[str, Union[str, int]] = {
            "session_id": session_id,
            "test_id": test_id,
            "round_id": round_id,
        }
        response = requests.get(f"{self.url}/session/evaluations", params=params,)

        self._check_response(response)
        return json.loads(response.text)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.INFO),
    )
    def get_test_metadata(self, session_id: str, test_id: str) -> Dict[str, Any]:
        """
        Retrieve the metadata json for the specified test.

        Args:
            session_id: The id of the session currently being evaluated
            test_id: The id of the test currently being evaluated

        Returns:
            A dictionary containing metadata
        """
        response = requests.get(
            f"{self.url}/test/metadata",
            params={"test_id": test_id, "session_id": session_id},
        )

        self._check_response(response)
        return response.json()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.INFO),
    )
    def complete_test(self, session_id: str, test_id: str) -> None:
        """
        Mark test as completed.

        Args:
            session_id: The id of the session currently being evaluated
            test_id: The id of the test currently being evaluated

        Returns:
            None
        """
        requests.delete(
            f"{self.url}/test", params={"test_id": test_id, "session_id": session_id},
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.INFO),
    )
    def terminate_session(self, session_id: str) -> None:
        """
        Terminate the session after the evaluation for the protocol is complete.

        Args:
            session_id: The id provided by a server denoting a session

        Returns: None
        """
        response = requests.delete(
            f"{self.url}/session", params={"session_id": session_id}
        )

        self._check_response(response)
