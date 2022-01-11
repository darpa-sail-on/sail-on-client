"""A provider class for evaluating with files."""

from sail_on_client.errors.errors import ProtocolError, ServerError, RoundError
from sail_on_client.harness.constants import ProtocolConstants
from sail_on_client.harness.file_provider_fn import (
    get_session_info,
    get_session_test_info,
    log_session,
    read_meta_data,
    read_gt_csv_file,
    get_classification_feedback,
    get_classificaton_score_feedback,
    get_detection_feedback,
    get_classification_var_feedback,
    get_levenshtein_feedback,
    psuedo_label_feedback,
    write_session_log_file,
)

import csv
import logging
import os
import glob
import uuid
import traceback
from typing import List, Dict, Any, Optional
from io import BytesIO


class FileProvider:
    """File-based service provider."""

    def __init__(self, folder: str, results_folder: str) -> None:
        """
        Initialize file provider.

        Args:
            folder: Folder where the data is present
            results_folder: Folder where the results are saved

        Returns:
            None
        """
        self.folder = folder
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)

    def get_test_metadata(
        self,
        session_id: str,
        test_id: str,
        api_call: bool = True,
        in_process_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Get test metadata.

        Args:
            session_id: Session id for which the info is required
            test_id: Test id for which the info is required
            api_call: Flag to change metadata to approved metadata
            in_process_only: Flag to get information while the session is active

        Returns:
            Metadata associated with the test as a dictionary
        """
        try:
            structure = get_session_info(
                self.results_folder, session_id, in_process_only=in_process_only
            )
            info = structure["created"]
            metadata_location = os.path.join(
                self.folder,
                info["protocol"],
                info["domain"],
                f"{test_id}_metadata.json",
            )
        except KeyError:
            raise ProtocolError(
                "session_id_invalid",
                f"Provided session id {session_id} could not be found or was improperly set up",
            )

        if not os.path.exists(metadata_location):
            raise ServerError(
                "metadata_not_found",
                f"Metadata file for Test Id {test_id} could not be found",
                "".join(traceback.format_stack()),
            )

        hints = []

        # List of metadata vars approved to be sent to the client
        approved_metadata = [
            "protocol",
            "known_classes",
            "max_novel_classes",
            "round_size",
            "feedback_max_ids",
            "pre_novelty_batches",
            "max_detection_feedback_ids",
        ]

        hints = info.get("hints", [])

        approved_metadata.extend([data for data in ["red_light"] if data in hints])

        overrides = {}
        for hint_data in hints:
            if "=" in hint_data:
                parts = hint_data.split("=")
                if parts[0] in approved_metadata:
                    overrides[parts[0]] = int(parts[1])

        md = read_meta_data(metadata_location)
        md.update(overrides)
        if api_call:
            return {k: v for k, v in md.items() if k in approved_metadata}
        return md

    def test_ids_request(
        self, protocol: str, domain: str, detector_seed: str, test_assumptions: str
    ) -> Dict[str, str]:
        """
        Request test IDs.

        Args:
            protocol: Name of the protocol
            domain: Domain of the application
            detector_seed: Seed used by the detector
            test_assumptions: Assumptions made when generating the tests

        Returns:
            Dictionary with the list of tests and seed used for generating tests
        """

        def _strip_id(filename: str) -> str:
            return os.path.splitext(os.path.basename(filename))[0]

        file_location = os.path.join(self.folder, protocol, domain, "test_ids.csv")
        if not os.path.exists(file_location):
            if not os.path.exists(os.path.join(self.folder, protocol)):
                msg = f"{protocol} not configured"
            elif not os.path.exists(os.path.join(self.folder, protocol, domain)):
                msg = f"domain {domain} for {protocol} not configured"
            else:
                test_ids = [
                    _strip_id(f)
                    for f in glob.glob(
                        os.path.join(self.folder, protocol, domain, "*.csv")
                    )
                ]
                with open(file_location, "w") as f:
                    f.writelines(test_ids)
                return {"test_ids": file_location, "generator_seed": "1234"}
            raise ProtocolError(
                "BadDomain", msg, "".join(traceback.format_stack()),
            )

        return {"test_ids": file_location, "generator_seed": "1234"}

    def new_session(
        self,
        test_ids: List[str],
        protocol: str,
        domain: str,
        novelty_detector_version: str,
        hints: List[str],
        detection_threshold: float,
    ) -> str:
        """
        Create a session.

        Args:
            test_ids: Test ids evaluated in the session
            protocol: Name of the protocol
            domain: Domain of the protocol
            novelty_detector_version: Version of the novelty detector
            hints: List of hints used in the session
            detection_threshold: Detection threshold used by the novelty detector

        Returns:
            Session id for the agent
        """
        # Verify's that all given test id's are valid and have associated csv files
        for test_id in test_ids:
            file_location = os.path.join(
                self.folder, protocol, domain, f"{test_id}_single_df.csv"
            )
            if not os.path.exists(file_location):
                raise ServerError(
                    "test_id_invalid",
                    f"Test Id {test_id} could not be matched to a specific file",
                    "".join(traceback.format_stack()),
                )

        session_id = str(uuid.uuid4())

        log_session(
            self.results_folder,
            session_id,
            activity="created",
            content={
                "protocol": protocol,
                "domain": domain,
                "detector": novelty_detector_version,
                "detection_threshold": detection_threshold,
                "hints": hints,
            },
        )

        return session_id

    def dataset_request(
        self, session_id: str, test_id: str, round_id: int
    ) -> Optional[BytesIO]:
        """
        Request a dataset.

        Args:
            test_id: The test being evaluated at this moment.
            round_id: The sequential number of the round being evaluated
            session_id: The identifier provided by the server for a single experiment

        Returns:
            Returns an instance of BytesIO
        """
        try:
            info = get_session_info(self.results_folder, session_id)["created"]
            test_info = get_session_test_info(self.results_folder, session_id, test_id)
            file_location = os.path.join(
                self.folder,
                info["protocol"],
                info["domain"],
                f"{test_id}_single_df.csv",
            )
        except KeyError:
            raise ProtocolError(
                "session_id_invalid",
                f"Provided session id {session_id} could not be found or was improperly set up",
            )

        if not os.path.exists(file_location):
            raise ServerError(
                "test_id_invalid",
                f"Test Id {test_id} could not be matched to a specific file",
                "".join(traceback.format_stack()),
            )

        metadata = self.get_test_metadata(session_id, test_id, False)

        if round_id is not None:
            # Check for removing leftover files from restarting tests within a session
            if int(round_id) == 0 and test_info:
                test_session_path = os.path.join(
                    self.results_folder, f"{str(session_id)}.{str(test_id)}.json"
                )
                if os.path.exists(test_session_path):
                    os.remove(test_session_path)
                test_result_paths = glob.glob(
                    os.path.join(
                        self.results_folder,
                        info["protocol"],
                        info["domain"],
                        f"{str(session_id)}.{str(test_id)}_*.csv",
                    )
                )
                for path in test_result_paths:
                    os.remove(path)

            temp_file_path = BytesIO()
            lines = read_gt_csv_file(file_location)
            lines = [x[0] for x in lines if x[0].strip("\n\t\"',.") != ""]
            try:
                round_pos = int(round_id) * int(metadata["round_size"])
            except KeyError:
                raise RoundError(
                    "no_defined_rounds",
                    f"round_size not defined in metadata for test id {test_id}",
                    "".join(traceback.format_stack()),
                )
            if round_pos >= len(lines):
                return None

            text = (
                "\n".join(lines[round_pos : round_pos + int(metadata["round_size"])])
                + "\n"
            ).encode("utf-8")
            temp_file_path.write(text)
            temp_file_path.seek(0)
        else:
            temp_file_path = open(file_location, "rb")

        log_session(
            self.results_folder,
            session_id,
            test_id=test_id,
            round_id=round_id,
            activity="data_request",
        )

        return temp_file_path

    feedback_request_mapping: Dict = {
        "image_classification": {
            ProtocolConstants.CLASSIFICATION: {
                "function": get_classification_feedback,
                "files": [ProtocolConstants.CLASSIFICATION],
                "columns": [1],
                "detection_req": ProtocolConstants.NOTIFY_AND_CONTINUE,
                "budgeted_feedback": True,
            },
            ProtocolConstants.SCORE: {
                "function": get_classificaton_score_feedback,
                "files": [ProtocolConstants.CLASSIFICATION],
                "columns": [1],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": False,
            },
            ProtocolConstants.DETECTION: {
                "function": get_detection_feedback,
                "files": [ProtocolConstants.DETECTION],
                "columns": [0],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": True,
                "required_hints": [],
                "alternate_budget": "max_detection_feedback_ids",
            },
        },
        "transcripts": {
            ProtocolConstants.CLASSIFICATION: {
                "function": get_classification_feedback,
                "files": [ProtocolConstants.CLASSIFICATION],
                "columns": [4],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": True,
            },
            ProtocolConstants.TRANSCRIPTION: {
                "function": get_levenshtein_feedback,
                "files": [ProtocolConstants.TRANSCRIPTION],
                "columns": [0],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": True,
            },
            ProtocolConstants.SCORE: {
                "function": get_classificaton_score_feedback,
                "files": [ProtocolConstants.CLASSIFICATION],
                "columns": [4],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": False,
            },
        },
        "activity_recognition": {
            ProtocolConstants.CLASSIFICATION: {
                "function": get_classification_var_feedback,
                "files": [ProtocolConstants.CLASSIFICATION],
                "columns": [5, 10],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": True,
            },
            ProtocolConstants.SCORE: {
                "function": get_classificaton_score_feedback,
                "files": [ProtocolConstants.CLASSIFICATION],
                "columns": [2],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": False,
            },
            ProtocolConstants.DETECTION: {
                "function": get_detection_feedback,
                "files": [ProtocolConstants.DETECTION],
                "columns": [0],
                "detection_req": ProtocolConstants.SKIP,
                "budgeted_feedback": True,
                "required_hints": [],
                "alternate_budget": "max_detection_feedback_ids",
            },
        },
    }

    def get_feedback(
        self, feedback_ids: List[str], feedback_type: str, session_id: str, test_id: str
    ) -> BytesIO:
        """
        Get feedback of the specified type.

        Args:
            feedback_ids: List of media ids for which feedback is required
            feedback_type: Protocols constants with the values: label, detection, characterization
            session_id: The id provided by a server denoting a session
            test_id: The id of the test currently being evaluated

        Returns:
            An instance of BytesIO with feedback
        """
        metadata = self.get_test_metadata(session_id, test_id, False)
        structure = get_session_info(self.results_folder, session_id)
        test_structure = get_session_test_info(self.results_folder, session_id, test_id)
        domain = structure["created"]["domain"]
        if domain not in self.feedback_request_mapping:
            raise ProtocolError(
                "BadDomain",
                f"The set domain does not match a domain type. Please check the metadata file for {test_id}",
                "".join(traceback.format_stack()),
            )

        # Ensure feedback type works with session domain
        # and if so, grab the proper files

        try:
            feedback_definition = self.feedback_request_mapping[domain][feedback_type]
            file_types = feedback_definition["files"]
        except Exception:
            raise ProtocolError(
                "InvalidFeedbackType",
                f"Invalid feedback type requested for the test id {test_id} with domain {domain}",
                "".join(traceback.format_stack()),
            )

        is_given_detection_mode = "red_light" in structure["created"].get("hints", [])
        budgeted_feedback = feedback_definition["budgeted_feedback"] and not (
            feedback_type == ProtocolConstants.DETECTION and is_given_detection_mode
        )

        if (
            "alternate_budget" in feedback_definition
            and feedback_definition["alternate_budget"] in metadata
        ):
            feedback_budget = int(metadata[feedback_definition["alternate_budget"]])
        else:
            feedback_budget = int(metadata.get("feedback_max_ids", 0))

        try:
            # Gets the amount of ids already requested for this type of feedback this round and
            # determines whether the limit has already been reached
            feedback_round_id = str(
                max([int(r) for r in test_structure["post_results"]["rounds"].keys()])
            )

            feedback_count = test_structure["get_feedback"]["rounds"][
                feedback_round_id
            ].get(feedback_type, 0)
            if feedback_count >= feedback_budget:
                raise ProtocolError(
                    "FeedbackBudgetExceeded",
                    f"Feedback of type {feedback_type} has already been requested on the maximum number of ids",
                )
        except KeyError:
            feedback_round_id = str(0)
            feedback_count = 0

        ground_truth_file = os.path.join(
            self.folder, metadata["protocol"], domain, f"{test_id}_single_df.csv"
        )

        if not os.path.exists(ground_truth_file):
            raise ServerError(
                "test_id_invalid",
                f"Could not find ground truth file for test Id {test_id}",
                "".join(traceback.format_stack()),
            )

        results_files = []
        for t in file_types:
            results_files.append(
                os.path.join(
                    self.results_folder,
                    metadata["protocol"],
                    domain,
                    f"{str(session_id)}.{str(test_id)}_{t}.csv",
                )
            )

        if len(results_files) < len(file_types):
            raise ServerError(
                "test_id_invalid",
                f"Could not find posted result file(s) for test Id {test_id} with feedback type {feedback_type}",
                "".join(traceback.format_stack()),
            )

        # Ensure any required hint(s) are present in the session info structure
        req_hints = feedback_definition.get("required_hints", [])
        if len(req_hints) > 0:
            for hint in req_hints:
                if hint not in structure["created"].get("hints", []):
                    logging.warning(
                        "Inform TA2 team that they are requesting feedback prior to the threshold indication"
                    )
                    return BytesIO()

        detection_requirement = feedback_definition.get(
            "detection_req", ProtocolConstants.IGNORE
        )

        # If novelty detection is required, ensure detection has been posted
        # for the requested round and novelty claimed for the test
        if detection_requirement != ProtocolConstants.IGNORE:
            test_results_structure = test_structure["post_results"]
            if "detection file path" not in test_results_structure:
                raise ProtocolError(
                    "DetectionPostRequired",
                    """A detection file is required to be posted before feedback can be requested on a round.
                       Please submit Detection results before requesting feedback""",
                )
            else:
                try:
                    with open(
                        test_results_structure["detection file path"], "r"
                    ) as d_file:
                        d_reader = csv.reader(d_file, delimiter=",")
                        detection_lines = list(d_reader)
                    predictions = [float(x[1]) for x in detection_lines]
                    # if given detection and past the detection point
                    is_given = is_given_detection_mode and metadata.get(
                        "red_light"
                    ) in [x[0] for x in detection_lines]
                    if (
                        max(predictions) <= structure["created"]["detection_threshold"]
                        and not is_given
                    ):
                        if (
                            detection_requirement
                            == ProtocolConstants.NOTIFY_AND_CONTINUE
                        ):
                            logging.error(
                                "Inform TA2 team that they are requesting feedback prior to the threshold indication"
                            )
                        elif detection_requirement == ProtocolConstants.SKIP:
                            logging.warning(
                                "Inform TA2 team that they are requesting feedback prior to the threshold indication"
                            )
                            return BytesIO()
                        else:
                            raise ProtocolError(
                                "NoveltyDetectionRequired",
                                "In order to request feedback, novelty must be declared for the test",
                            )
                except ProtocolError as e:
                    raise e
                except Exception:
                    raise ServerError(
                        "CantReadFile",
                        f"""Couldnt open the logged detection file at
                            {test_results_structure['detection file path']}.
                            Please check if the file exists and that
                            {session_id}.json has the proper file location
                            for test id {test_id}""",
                        traceback.format_exc(),
                    )

        # Add columns to metadata for use in feedback
        metadata["columns"] = feedback_definition.get("columns", [])

        # Get feedback from specified test
        try:
            if "psuedo" in feedback_type:
                feedback = psuedo_label_feedback(
                    ground_truth_file,
                    feedback_ids,
                    feedback_definition["files"][0],
                    metadata,
                    self.results_folder,
                    session_id,
                )
            else:
                feedback = feedback_definition["function"](
                    ground_truth_file, results_files, feedback_ids, metadata
                )
        except KeyError:
            raise ProtocolError(
                "feedback_type_invalid",
                f"""Feedback type {feedback_type} is not valid. Make sure the
                    provider's feedback_algorithms variable is properly set up""",
                traceback.format_exc(),
            )

        number_of_ids_to_return = len(feedback)

        # if budgeted, decrement use and check if too many has been requested
        if budgeted_feedback:
            left_over_ids = feedback_budget - feedback_count
            number_of_ids_to_return = min(number_of_ids_to_return, left_over_ids)
        feedback_count += number_of_ids_to_return

        log_session(
            self.results_folder,
            session_id=session_id,
            activity="get_feedback",
            test_id=test_id,
            round_id=int(feedback_round_id),
            content={feedback_type: feedback_count, "feedback_budget": feedback_budget},
        )

        feedback_csv = BytesIO()
        for key in feedback.keys():
            if type(feedback[key]) is not list:
                feedback_csv.write(f"{key},{feedback[key]}\n".encode("utf-8"))
            else:
                feedback_csv.write(
                    f"{key},{','.join(str(x) for x in feedback[key])}\n".encode("utf-8")
                )
            number_of_ids_to_return -= 1
            # once maximium requested number is hit, quit
            if number_of_ids_to_return == 0:
                break

        feedback_csv.seek(0)

        return feedback_csv

    def post_results(
        self,
        session_id: str,
        test_id: str,
        round_id: int,
        result_files: Dict[str, str],
    ) -> None:
        """
        Post results.

        Args:
            result_files: A dictionary of results with protocol constant as key and file path as value
            test_id: The id of the test currently being evaluated
            round_id: The sequential number of the round being evaluated
            session_id: The id provided by a server denoting a session

        Returns:
            None
        """
        # Modify session file with posted results
        structure = get_session_info(self.results_folder, session_id)
        test_structure = get_session_test_info(self.results_folder, session_id, test_id)
        if "detection" in result_files.keys():
            try:
                if (
                    "detection"
                    in test_structure["post_results"]["rounds"][str(round_id)]["types"]
                ):
                    raise ProtocolError(
                        "DetectionRepost",
                        """Cannot re post detection for a given round. If you
                           attempted to submit any other results, please
                           resubmit without detection.""",
                    )
            except KeyError:
                pass

        protocol = structure["created"]["protocol"]
        domain = structure["created"]["domain"]
        os.makedirs(os.path.join(self.results_folder, protocol, domain), exist_ok=True)
        log_content = {}
        for r_type in result_files.keys():
            filename = f"{str(session_id)}.{str(test_id)}_{r_type}.csv"
            path = os.path.join(self.results_folder, protocol, domain, filename)
            log_content[f"{r_type} file path"] = path
            with open(path, "a+") as result_file:
                result_file.write(result_files[r_type])

        # Log call
        log_content["last round"] = str(round_id)
        updated_test_structure = log_session(
            self.results_folder,
            activity="post_results",
            session_id=session_id,
            test_id=test_id,
            round_id=round_id,
            content=log_content,
            content_loc="activity",
            return_structure=True,
        )
        if updated_test_structure:
            prev_types = updated_test_structure["post_results"]["rounds"][
                str(round_id)
            ].get("types", [])
            new_types = prev_types + list(result_files.keys())
            updated_test_structure["post_results"]["rounds"][str(round_id)][
                "types"
            ] = new_types
            write_session_log_file(
                updated_test_structure,
                os.path.join(
                    self.results_folder, f"{str(session_id)}.{str(test_id)}.json"
                ),
            )

    def complete_test(self, session_id: str, test_id: str) -> None:
        """
        Mark test as completed in session logs.

        Args:
            session_id: The id of the session currently being evaluated
            test_id: The id of the test currently being evaluated

        Returns:
            None
        """
        log_session(
            self.results_folder,
            session_id=session_id,
            test_id=test_id,
            activity="completion",
        )

    def terminate_session(self, session_id: str) -> None:
        """
        Terminate the session.

        Args:
            session_id: The id provided by a server denoting a session

        Returns: None
        """
        # Modify session file to indicate session has been terminated
        log_session(self.results_folder, session_id=session_id, activity="termination")

    def latest_session_info(self, session_id: str) -> Dict:
        """
        Get tests finished from the most recent session.

        Args:
            session_id: The id of the session currently being evaluated

        Returns:
            List of tests that have been marked as completed in the session
        """
        structure = get_session_info(self.results_folder, session_id)
        latest = {}
        latest["finished_tests"] = structure["tests"]["completed_tests"]
        return latest
