"""Implementation of T&E Harness for running experiments locally."""

from sail_on_client.harness.file_provider import FileProvider
from sail_on_client.harness.file_provider_fn import get_session_info
from sail_on_client.errors import RoundError as ClientRoundError
from sail_on_client.evaluate import create_metric_instance
from sail_on_client.harness.test_and_evaluation_harness import TestAndEvaluationHarness

from tempfile import TemporaryDirectory
from typing import Any, Dict, Union, List
import os
import io
import logging
import ubelt as ub
import pandas as pd
import json

log = logging.getLogger(__name__)


class LocalHarness(TestAndEvaluationHarness):
    """Harness without any server communication."""

    def __init__(
        self, data_dir: str, result_dir: str, gt_dir: str = "", gt_config: str = ""
    ) -> None:
        """
        Initialize an object of local harness.

        Args:
            data_dir: Path to the directory with the data
            result_dir: Path to the directory where results are stored
            gt_dir: Path to directory with ground truth
            gt_config: Path to config file with column mapping for ground truth

        Returns:
            None
        """
        TestAndEvaluationHarness.__init__(self)
        self.temp_dir = TemporaryDirectory()
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.gt_config = gt_config
        self.temp_dir_name = self.temp_dir.name
        self.result_dir = result_dir
        self.file_provider = FileProvider(self.data_dir, self.result_dir)

    def get_config(self) -> Dict:
        """JSON Compliant representation of the object."""
        return {
            "data_dir": self.data_dir,
            "gt_dir": self.gt_dir,
            "result_dir": self.result_dir,
            "gt_config": self.gt_config,
        }

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
            test_ids: List of tests being evaluated in this session
            protocol: String indicating which protocol is being evaluated
            domain: String indicating which domain is being evaluated
            novelty_detector_version: The novelty detector being evaluated
            hints: Hints used for the session
            detection_threshold: Detection threshold for the session

        Returns:
            A session identifier provided by the server
        """
        return self.file_provider.new_session(
            test_ids,
            protocol,
            domain,
            novelty_detector_version,
            hints,
            detection_threshold,
        )

    def resume_session(self, session_id: str) -> List[str]:
        """
        Get finished test from an existing session.

        Args:
            session id : Session id that was started but not terminated

        Returns:
            List of tests finished in the session
        """
        return self.file_provider.latest_session_info(session_id)["finished_tests"]

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
        self.data_file = os.path.join(
            self.temp_dir_name, f"{session_id}.{test_id}.{round_id}.csv"
        )
        byte_stream = self.file_provider.dataset_request(session_id, test_id, round_id)
        if byte_stream is None:
            raise ClientRoundError(
                reason="End of Dataset", msg="All Data from dataset has been requested"
            )
        else:
            with open(self.data_file, "wb") as f:
                f.write(byte_stream.getbuffer())
            return self.data_file

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
        self.feedback_file = os.path.join(
            self.temp_dir_name,
            "feedback",
            f"{session_id}.{test_id}.{round_id}_{feedback_type}.csv",
        )
        ub.ensuredir(os.path.join(self.temp_dir_name, "feedback"))
        byte_stream = self.file_provider.get_feedback(
            feedback_ids, feedback_type, session_id, test_id
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
            result_files: A dictionary of results with protocol constant as key and file path as value
            test_id: The id of the test currently being evaluated
            round_id: The sequential number of the round being evaluated
            session_id: The id provided by a server denoting a session

        Returns:
            None
        """
        info = get_session_info(str(self.result_dir), session_id)
        protocol = info["created"]["protocol"]
        domain = info["created"]["domain"]
        base_result_path = os.path.join(str(self.result_dir), protocol, domain)
        os.makedirs(base_result_path, exist_ok=True)
        result_content = {}
        for result_key, result_file in result_files.items():
            with open(result_file, "r") as f:
                content = f.read()
            result_content[result_key] = io.StringIO(content).getvalue()
        self.file_provider.post_results(session_id, test_id, round_id, result_content)

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
        gt_file_id = os.path.join(self.gt_dir, f"{test_id}_single_df.csv")
        gt = pd.read_csv(gt_file_id, sep=",", header=None, skiprows=1, quotechar="|")
        info = get_session_info(str(self.result_dir), session_id)
        protocol = info["created"]["protocol"]
        domain = info["created"]["domain"]
        metadata = self.get_test_metadata(session_id, test_id)
        round_size = metadata["round_size"]
        results: Dict[str, Union[Dict, float]] = {}
        gt_config = json.load(open(self.gt_config, "r"))
        classification_file_id = os.path.join(
            self.result_dir,
            protocol,
            domain,
            f"{session_id}.{test_id}_classification.csv",
        )

        classifications = pd.read_csv(
            classification_file_id, sep=",", header=None, quotechar="|"
        )
        gt_round = gt.iloc[round_id * round_size : (round_id + 1) * round_size]
        classification_round = classifications[
            round_id * round_size : (round_id + 1) * round_size
        ]
        metric = create_metric_instance(protocol, domain, gt_config)
        m_acc = metric.m_acc_round_wise(
            classification_round, gt_round[metric.classification_id], round_id
        )
        results[f"m_acc_round_{round_id}"] = m_acc
        log.info(f"Accuracy for {test_id}, {round_id}: {ub.repr2(results)}")
        return results

    def evaluate(
        self,
        test_id: str,
        round_id: int,
        session_id: str,
        baseline_session_id: str = None,
    ) -> Dict[str, Any]:
        """
        Get results for test(s).

        Args:
            test_id: The id of the test currently being evaluated
            round_id: The sequential number of the round being evaluated
            session_id: The id provided by a server denoting a session

        Returns:
            Path to a file with the results
        """
        gt_file_id = os.path.join(self.gt_dir, f"{test_id}_single_df.csv")
        gt = pd.read_csv(gt_file_id, sep=",", header=None, skiprows=1, quotechar="|")
        info = get_session_info(str(self.result_dir), session_id)
        protocol = info["created"]["protocol"]
        domain = info["created"]["domain"]
        results: Dict[str, Union[Dict, float]] = {}
        gt_config = json.load(open(self.gt_config, "r"))
        detection_file_id = os.path.join(
            self.result_dir, protocol, domain, f"{session_id}.{test_id}_detection.csv",
        )
        detections = pd.read_csv(detection_file_id, sep=",", header=None, quotechar="|")
        classification_file_id = os.path.join(
            self.result_dir,
            protocol,
            domain,
            f"{session_id}.{test_id}_classification.csv",
        )

        classifications = pd.read_csv(
            classification_file_id, sep=",", header=None, quotechar="|"
        )
        if baseline_session_id is not None:
            baseline_classification_file_id = os.path.join(
                self.result_dir,
                protocol,
                domain,
                f"{baseline_session_id}.{test_id}_classification.csv",
            )
            baseline_classifications = pd.read_csv(
                baseline_classification_file_id, sep=",", header=None
            )
        metric = create_metric_instance(protocol, domain, gt_config)
        detection_idx = 1
        gt_classification_idx = metric.classification_id
        # ######## Image Classification Evaluation  ###########
        if domain == "image_classification":
            gt_detection_idx = metric.detection_id
            # p_unknown column for image classification
            novel_idx = 1
        # ######## Activity Recognition Evaluation  ###########
        elif domain == "activity_recognition":
            detection_idx = 1
            gt_detection_idx = metric.novel_id
            # p_unknown column for activity recognition
            novel_idx = 31
        # ######## Document Transcription Evaluation  ###########
        elif domain == "transcripts":
            detection_idx = 1
            gt_detection_idx = metric.novel_id
            # p_unknown column for transcripts in writer identification
            novel_idx = 50
        else:
            raise AttributeError(
                f'Domain: "{domain}" is not a real domain.  Get a clue.'
            )

        m_num = metric.m_num(detections[detection_idx], gt[gt_detection_idx])
        results["m_num"] = m_num
        m_num_stats = metric.m_num_stats(
            detections[detection_idx], gt[gt_detection_idx]
        )
        results["m_num_stats"] = m_num_stats
        m_ndp = metric.m_ndp(classifications[novel_idx], gt[gt_detection_idx])
        results["m_ndp"] = m_ndp
        m_ndp_pre = metric.m_ndp_pre(classifications[novel_idx], gt[gt_detection_idx])
        results["m_ndp_pre"] = m_ndp_pre
        m_ndp_post = metric.m_ndp_post(classifications[novel_idx], gt[gt_detection_idx])
        results["m_ndp_post"] = m_ndp_post
        m_acc = metric.m_acc(
            gt[gt_detection_idx], classifications, gt[gt_classification_idx], 100, 5,
        )
        results["m_acc"] = m_acc
        m_acc_failed = metric.m_ndp_failed_reaction(
            detections[detection_idx],
            gt[gt_detection_idx],
            classifications,
            gt[gt_classification_idx],
        )
        results["m_acc_failed"] = m_acc_failed
        m_is_cdt_and_is_early = metric.m_is_cdt_and_is_early(
            m_num_stats["GT_indx"], m_num_stats["P_indx_0.5"], gt.shape[0],
        )
        results["m_is_cdt_and_is_early"] = m_is_cdt_and_is_early
        if baseline_session_id is not None:
            m_acc_baseline = metric.m_acc(
                gt[gt_detection_idx],
                baseline_classifications,
                gt[gt_classification_idx],
                100,
                5,
            )
            log.info(f"Baseline performance for {test_id}: {ub.repr2(m_acc_baseline)}")
            m_nrp = metric.m_nrp(m_acc, m_acc_baseline)
            results["m_nrp"] = m_nrp

        log.info(f"Results for {test_id}: {ub.repr2(results)}")
        return results

    def get_test_metadata(self, session_id: str, test_id: str) -> Dict[str, Any]:
        """
        Retrieve the metadata json for the specified test.

        Args:
            session_id: The id of the session currently being evaluated
            test_id: The id of the test currently being evaluated

        Returns:
            A dictionary containing metadata
        """
        return self.file_provider.get_test_metadata(session_id, test_id)

    def complete_test(self, session_id: str, test_id: str) -> None:
        """
        Mark test as completed.

        Args:
            session_id: The id of the session currently being evaluated
            test_id: The id of the test currently being evaluated

        Returns:
            None
        """
        self.file_provider.complete_test(session_id, test_id)

    def terminate_session(self, session_id: str) -> None:
        """
        Terminate the session after the evaluation for the protocol is complete.

        Args:
            session_id: The id provided by a server denoting a session

        Returns: None
        """
        self.file_provider.terminate_session(session_id)
