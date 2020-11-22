"""Client implementation for Par interface."""

import requests
import json
import io
import os
import traceback
import logging

from tinker.harness import Harness
from typing import Any, Dict, Union
from requests import Response
from sail_on_client.errors import ApiError
from json import JSONDecodeError


class ParInterface(Harness):
    """Interface to PAR server."""

    def __init__(self, configfile: str, configfolder: str) -> None:
        """
        Initialize a client connection object.

        :param api_url: url for where server is hosted
        """
        Harness.__init__(self, configfile, configfolder)
        self.api_url = self.configuration_data["url"]
        self.folder = configfolder

    def _check_response(self, response: Response) -> None:
        """
        Produce appropriate output on error.

        :param response:
        :return: True
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
                logging.exception(f"Server Error: {traceback.format_exc()}")
                exit(1)

    def test_ids_request(
        self,
        protocol: str,
        domain: str,
        detector_seed: str,
        test_assumptions: str = "{}",
    ) -> str:
        """
        Request Test Identifiers as part of a series of individual tests.

        Arguments:
            -protocol   : string indicating which protocol is being evaluated
            -domain     :
            -detector_seed
            -test_assumptions
        Returns:
            -filename of file containing test ids
        """
        payload = {
            "protocol": protocol,
            "domain": domain,
            "detector_seed": detector_seed,
        }

        with open(test_assumptions, "r") as f:
            contents = f.read()

        response = requests.get(
            f"{self.api_url}/test/ids",
            files={
                "test_requirements": io.StringIO(json.dumps(payload)),
                "test_assumptions": io.StringIO(contents),
            },
        )

        self._check_response(response)

        filename = os.path.abspath(
            os.path.join(self.folder, f"{protocol}.{domain}.{detector_seed}.csv")
        )
        with open(filename, "w") as f:
            f.write(response.content.decode("utf-8"))

        return filename

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

        Arguments:
            -test_ids   : list of tests being evaluated in this session
            -protocol   : string indicating which protocol is being evaluated
            -domain     : string indicating which domain is being evaluated
            -novelty_detector_version : string indicating the version of the novelty detector being evaluated
            -hint       : a list hints provided for the session
        Returns:
            -session id
        """
        payload = {
            "protocol": protocol,
            "novelty_detector_version": novelty_detector_version,
            "domain": domain,
            "hints": hints,
        }

        ids = "\n".join(test_ids) + "\n"

        response = requests.post(
            f"{self.api_url}/session",
            files={"test_ids": ids, "configuration": io.StringIO(json.dumps(payload))},
        )

        self._check_response(response)
        return response.json()["session_id"]

    def dataset_request(self, test_id: str, round_id: int, session_id: str) -> str:
        """
        Request data for evaluation.

        Arguments:
            -test_id    : the test being evaluated at this moment.
            -round_id   : the sequential number of the round being evaluated
        Returns:
            -filename of a file containing a list of image files (including full path for each)
        """
        params: Dict[str, Union[str, int]] = {
            "session_id": session_id,
            "test_id": test_id,
            "round_id": round_id,
        }
        response = requests.get(f"{self.api_url}/session/dataset", params=params,)

        self._check_response(response)

        filename = os.path.abspath(
            os.path.join(self.folder, f"{session_id}.{test_id}.{round_id}.csv")
        )
        with open(filename, "wb") as f:
            f.write(response.content)
        return filename

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

        Arguments:
            -feedback_ids
            -test_id        : the id of the test currently being evaluated
            -round_id       : the sequential number of the round being evaluated
            -feedback_type -- label, detection, characterization
        Returns:
            -labels dictionary
        """
        params: Dict[str, Union[str, int]] = {
            "feedback_ids": "|".join(feedback_ids),
            "session_id": session_id,
            "test_id": test_id,
            "round_id": round_id,
            "feedback_type": feedback_type,
        }
        response = requests.get(f"{self.api_url}/session/feedback", params=params,)

        self._check_response(response)
        filename = os.path.abspath(
            os.path.join(
                self.folder, f"{session_id}.{test_id}.{round_id}_{feedback_type}.csv"
            )
        )
        with open(filename, "wb") as f:
            f.write(response.content)

        return filename

    def post_results(
        self, result_files: Dict[str, str], test_id: str, round_id: int, session_id: str
    ) -> None:
        """
        Post client detector predictions for the dataset.

        Arguments:
            -result_files (dict of "type : file")
            -session_id
            -test_id
            -round_id
        Returns: No return
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

        response = requests.post(f"{self.api_url}/session/results", files=files)

        self._check_response(response)

    def evaluate(self, test_id: str, round_id: int, session_id: str) -> str:
        """
        Get results for test(s).

        Arguments:
            -test_id
            -round_id
        Returns:
            -filename
        """
        params: Dict[str, Union[str, int]] = {
            "session_id": session_id,
            "test_id": test_id,
            "round_id": round_id,
        }
        response = requests.get(f"{self.api_url}/session/evaluations", params=params,)

        self._check_response(response)

        filename = os.path.abspath(
            os.path.join(
                self.folder, f"{session_id}.{test_id}.{round_id}_evaluation.csv"
            )
        )

        with open(filename, "w") as f:
            f.write(response.content.decode("utf-8"))

        return filename

    def get_test_metadata(self, session_id: str, test_id: str) -> Dict[str, Any]:
        """
        Retrieve the metadata json for the specified test.

        Arguments:
            -test_id
        Returns:
            metadata json
        """
        response = requests.get(
            f"{self.api_url}/test/metadata",
            params={"test_id": test_id, "session_id": session_id},
        )

        self._check_response(response)
        return response.json()

    def terminate_session(self, session_id: str) -> None:
        """
        Terminate the session after the evaluation for the protocol is complete.

        Arguments:
        Returns: No return
        """
        response = requests.delete(
            f"{self.api_url}/session", params={"session_id": session_id}
        )

        self._check_response(response)
