"""Round for visual protocol."""

import logging
from typing import List, Any, Tuple, Dict

from sail_on_client.protocol.visual_dataclasses import (
    FeatureExtractionParams,
    WorldChangeDetectionParams,
)
from sail_on_client.harness.test_and_evaluation_harness import (
    TestAndEvaluationHarnessType,
)
from sail_on_client.utils.utils import safe_remove
from sail_on_client.utils.decorators import skip_stage


log = logging.getLogger(__name__)


class VisualRound:
    """Class with common elements for visual protocols."""

    def __init__(
        self,
        algorithm: Any,
        data_root: str,
        features_dict: Dict,
        harness: TestAndEvaluationHarnessType,
        logit_dict: Dict,
        redlight_instance: str,
        session_id: str,
        skip_stages: List[str],
        test_id: str,
    ) -> None:
        """
        Construct VisualRound.

        Args:
            algorithm: An instance of algorithm
            data_root: Root directory of the data
            features_dict: Dictionary with features for the entire dataset
            harness: An instance of the harness used for T&E
            logit_dict: Dictionary with logits for the entire dataset
            redlight_instance: The instance when the world changes
            session_id: Session id associated with the algorithm
            skip_stages: List of stages that are skipped
            test_id: Test id associated with the round

        Returns:
            None
        """
        self.algorithm = algorithm
        self.data_root = data_root
        self.features_dict = features_dict
        self.harness = harness
        self.logit_dict = logit_dict
        self.redlight_instance = redlight_instance
        self.session_id = session_id
        self.skip_stages = skip_stages
        self.test_id = test_id

    @staticmethod
    def get_instance_ids(dataset_path: str) -> List[str]:
        """
        Get instance ids from the dataset.

        Args:
            dataset_path: Path to text file with instances used in a round

        Returns:
            List of instance ids from the dataset
        """
        with open(dataset_path, "r") as dataset:
            instance_ids = dataset.readlines()
            instance_ids = [instance_id.strip() for instance_id in instance_ids]
        return instance_ids

    @skip_stage("FeatureExtraction", ({}, {}))
    def _run_feature_extraction(
        self, fe_params: FeatureExtractionParams, instance_ids: List[str]
    ) -> Tuple[Dict, Dict]:
        """
        Private helper function for running feature extraction.

        Args:
            fe_params: An instance of dataclass with parameters for feature extraction
            instance_ids: Identifiers associated with data for a round

        Returns:
            Tuple for feature and logit dictionary for a round
        """
        rfeature_dict, rlogit_dict = {}, {}
        if len(self.features_dict) > 0 and len(self.logit_dict) > 0:
            for instance_id in instance_ids:
                rfeature_dict[instance_id] = self.features_dict[instance_id]
                rlogit_dict[instance_id] = self.logit_dict[instance_id]
        else:
            fe_toolset = fe_params.get_toolset()
            rfeature_dict, rlogit_dict = self.algorithm.execute(
                fe_toolset, "FeatureExtraction"
            )
        self.rfeature_dict, self.rlogit_dict = rfeature_dict, rlogit_dict
        return rfeature_dict, rlogit_dict

    @skip_stage("WorldDetection")
    def _run_world_change_detection(
        self, wcd_params: WorldChangeDetectionParams, round_id: int,
    ) -> None:
        """
        Private helper function for detecting that the world has changed.

        Args:
            wcd_params: An instance of dataclass with parameters for world change detection
            round_id: Identifier for a round

        Returns:
            None
        """
        wd_result = self.algorithm.execute(wcd_params.get_toolset(), "WorldDetection")
        self.harness.post_results(
            {"detection": wd_result}, self.test_id, round_id, self.session_id
        )
        safe_remove(wd_result)
