"""Client implementation for local interface."""

from sail_on.api.file_provider import FileProvider
from sail_on.api.file_provider import get_session_info
from sail_on.api.errors import RoundError
from sail_on_client.errors import RoundError as ClientRoundError
from sail_on_client.evaluate.image_classification import ImageClassificationMetrics
from sail_on_client.evaluate.activity_recognition import ActivityRecognitionMetrics
from sail_on_client.evaluate.document_transcription import DocumentTranscriptionMetrics
from sailon_tinker_launcher.deprecated_tinker.harness import Harness

from tempfile import TemporaryDirectory
from typing import Any, Dict, Union
import os
import io
import logging
import ubelt as ub
import pandas as pd
import json

log = logging.getLogger(__name__)


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
        self.gt_dir = self.configuration_data["gt_dir"]
        self.gt_config = json.load(open(self.configuration_data["gt_config"], "r"))
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
        detection_threshold: float,
    ) -> str:
        """
        Create a new session to evaluate the detector using an empirical protocol.

        Args:
            test_ids   : list of tests being evaluated in this session
            protocol   : string indicating which protocol is being evaluated
            domain     : string indicating which domain is being evaluated
            novelty_detector_version : string indicating the version of the novelty detector being evaluated
            hints      : Hints used for the session
            detection_threshold      : Detection threshold for the session
        Returns:
            A session identifier provided by the server
        """
        return self.file_provider.new_session(
            test_ids, protocol, domain, novelty_detector_version, hints, detection_threshold
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
            "feedback",
            f"{session_id}.{test_id}.{round_id}_{feedback_type}.csv",
        )
        ub.ensuredir(os.path.join(self.result_directory, "feedback"))
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
        result_content = {}
        for result_key, result_file in result_files.items():
            with open(result_file, "r") as f:
                content = f.read()
            result_content[result_key] = io.StringIO(content).getvalue()
        self.file_provider.post_results(session_id, test_id, round_id, result_content)

    def evaluate(self, test_id: str, round_id: int, session_id: str) -> Dict[str, Any]:
        """
        Get results for test(s).

        Args:
            test_id        : the id of the test currently being evaluated
            round_id       : the sequential number of the round being evaluated
            session_id     : the id provided by a server denoting a session

        Returns:
            Path to a file with the results
        """
        gt_file_id = os.path.join(self.gt_dir, f"{test_id}_single_df.csv")
        gt = pd.read_csv(gt_file_id, sep=",", header=None, skiprows=1)
        info = get_session_info(str(self.result_directory), session_id)
        protocol = info["activity"]["created"]["protocol"]
        domain = info["activity"]["created"]["domain"]
        results: Dict[str, Union[Dict, float]] = {}

        # ######## Image Classification Evaluation  ###########
        if domain == "image_classification":
            detection_file_id = os.path.join(
                self.result_directory,
                protocol,
                domain,
                f"{session_id}.{test_id}_detection.csv",
            )
            detections = pd.read_csv(detection_file_id, sep=",", header=None)
            classification_file_id = os.path.join(
                self.result_directory,
                protocol,
                domain,
                f"{session_id}.{test_id}_classification.csv",
            )

            classifications = pd.read_csv(classification_file_id, sep=",", header=None)
            arm_im = ImageClassificationMetrics(protocol, **self.gt_config)
            m_num = arm_im.m_num(detections[1], gt[arm_im.detection_id])
            results["m_num"] = m_num
            m_num_stats = arm_im.m_num_stats(detections[1], gt[arm_im.detection_id])
            results["m_num_stats"] = m_num_stats
            m_ndp = arm_im.m_ndp(detections[1], gt[arm_im.detection_id])
            results["m_ndp"] = m_ndp
            m_ndp_pre = arm_im.m_ndp_pre(detections[1], gt[arm_im.detection_id])
            results["m_ndp_pre"] = m_ndp_pre
            m_ndp_post = arm_im.m_ndp_post(detections[1], gt[arm_im.detection_id])
            results["m_ndp_post"] = m_ndp_post
            m_acc = arm_im.m_acc(
                gt[arm_im.detection_id],
                classifications,
                gt[arm_im.classification_id],
                100,
                5,
            )
            results["m_acc"] = m_acc
            m_acc_failed = arm_im.m_ndp_failed_reaction(
                detections[arm_im.detection_id],
                gt[1],
                classifications,
                gt[arm_im.classification_id],
            )
            results["m_acc_failed"] = m_acc_failed
            m_is_cdt_and_is_early = arm_im.m_is_cdt_and_is_early(
                m_num_stats["GT_indx"], m_num_stats["P_indx"], gt.shape[0],
            )
            results["m_is_cdt_and_is_early"] = m_is_cdt_and_is_early

        # ######## Activity Recognition Evaluation  ###########
        elif domain == "activity_recognition":
            detection_file_id = os.path.join(
                self.result_directory,
                protocol,
                domain,
                f"{session_id}.{test_id}_detection.csv",
            )
            detections = pd.read_csv(detection_file_id, sep=",", header=None)
            classification_file_id = os.path.join(
                self.result_directory,
                protocol,
                domain,
                f"{session_id}.{test_id}_classification.csv",
            )
            classifications = pd.read_csv(classification_file_id, sep=",", header=None)
            arm_ar = ActivityRecognitionMetrics(protocol, **self.gt_config)
            m_num = arm_ar.m_num(detections[1], gt[arm_ar.novel_id])
            results["m_num"] = m_num
            m_num_stats = arm_ar.m_num_stats(detections[1], gt[arm_ar.novel_id])
            results["m_num_stats"] = m_num_stats
            m_ndp = arm_ar.m_ndp(detections[1], gt[arm_ar.novel_id])
            results["m_ndp"] = m_ndp
            m_ndp_pre = arm_ar.m_ndp_pre(detections[1], gt[arm_ar.novel_id])
            results["m_ndp_pre"] = m_ndp_pre
            m_ndp_post = arm_ar.m_ndp_post(detections[1], gt[arm_ar.novel_id])
            results["m_ndp_post"] = m_ndp_post
            m_acc = arm_ar.m_acc(
                gt[arm_ar.novel_id],
                classifications,
                gt[arm_ar.classification_id],
                100,
                5,
            )
            results["m_acc"] = m_acc
            m_acc_failed = arm_ar.m_ndp_failed_reaction(
                detections[1],
                gt[arm_ar.novel_id],
                classifications,
                gt[arm_ar.classification_id],
            )
            results["m_acc_failed"] = m_acc_failed
            m_is_cdt_and_is_early = arm_ar.m_is_cdt_and_is_early(
                m_num_stats["GT_indx"], m_num_stats["P_indx"], gt.shape[0],
            )
            results["m_is_cdt_and_is_early"] = m_is_cdt_and_is_early
        # ######## Document Transcript Evaluation  ###########
        elif domain == "transcripts":
            detection_file_id = os.path.join(
                self.result_directory,
                protocol,
                domain,
                f"{session_id}.{test_id}_detection.csv",
            )
            detections = pd.read_csv(detection_file_id, sep=",", header=None)
            classification_file_id = os.path.join(
                self.result_directory,
                protocol,
                domain,
                f"{session_id}.{test_id}_classification.csv",
            )
            classifications = pd.read_csv(classification_file_id, sep=",", header=None)
            dtm = DocumentTranscriptionMetrics(protocol, **self.gt_config)
            m_num = dtm.m_num(detections[1], gt[dtm.novel_id])
            results["m_num"] = m_num
            m_num_stats = dtm.m_num_stats(detections[1], gt[dtm.novel_id])
            results["m_num_stats"] = m_num_stats
            m_ndp = dtm.m_ndp(detections[1], gt[dtm.novel_id])
            results["m_ndp"] = m_ndp
            m_ndp_pre = dtm.m_ndp_pre(detections[1], gt[dtm.novel_id])
            results["m_ndp_pre"] = m_ndp_pre
            m_ndp_post = dtm.m_ndp_post(detections[1], gt[dtm.novel_id])
            results["m_ndp_post"] = m_ndp_post
            m_acc = dtm.m_acc(
                gt[dtm.novel_id], classifications, gt[dtm.classification_id], 100, 5
            )
            results["m_acc"] = m_acc
            m_acc_failed = dtm.m_ndp_failed_reaction(
                detections[1],
                gt[dtm.novel_id],
                classifications,
                gt[dtm.classification_id],
            )
            results["m_acc_failed"] = m_acc_failed
            m_is_cdt_and_is_early = dtm.m_is_cdt_and_is_early(
                m_num_stats["GT_indx"], m_num_stats["P_indx"], gt.shape[0],
            )
            results["m_is_cdt_and_is_early"] = m_is_cdt_and_is_early
        else:
            raise AttributeError(
                f'Domain: "{domain}" is not a real domain.  Get a clue.'
            )
        log.info(f"Results for {test_id}: {ub.repr2(results)}")
        return results

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
