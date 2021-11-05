"""Round for OND."""

import logging
from typing import List, Any, Dict, Union

from sail_on_client.protocol.ond_dataclasses import (
    NoveltyClassificationParams,
    NoveltyAdaptationParams,
)
from sail_on_client.protocol.visual_dataclasses import (
    FeatureExtractionParams,
    WorldChangeDetectionParams,
)
from sail_on_client.protocol.visual_round import VisualRound
from sail_on_client.harness.test_and_evaluation_harness import (
    TestAndEvaluationHarnessType,
)
from sail_on_client.utils.utils import safe_remove
from sail_on_client.utils.decorators import skip_stage


log = logging.getLogger(__name__)


class ONDRound(VisualRound):
    """Class Representing a round in OND."""

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
        Construct round for OND.

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
        super().__init__(
            algorithm,
            data_root,
            features_dict,
            harness,
            logit_dict,
            redlight_instance,
            session_id,
            skip_stages,
            test_id,
        )

    @skip_stage("NoveltyClassification")
    def _run_novelty_classification(
        self, nc_params: NoveltyClassificationParams, round_id: int
    ) -> None:
        """
        Private helper function for novelty classification.

        Args:
            nc_params: An instance of dataclass with parameters for novelty classification
            round_id: Identifier for a round

        Returns:
            None
        """
        ncl_result = self.algorithm.execute(
            nc_params.get_toolset(), "NoveltyClassification"
        )
        self.harness.post_results(
            {"classification": ncl_result}, self.test_id, round_id, self.session_id
        )
        safe_remove(ncl_result)

    @skip_stage("EvaluateRoundwise")
    def _evaluate_roundwise(self, round_id: int) -> Dict:
        """
        Compute roundwise accuracy.

        Args:
            round_id: Identifier for a round

        Returns:
            Dictionary with accuracy metrics for round
        """
        return self.harness.evaluate_round_wise(self.test_id, round_id, self.session_id)

    @skip_stage("NoveltyAdaptation")
    def _run_novelty_adaptation(self, na_params: NoveltyAdaptationParams) -> None:
        """
        Private helper function for adaptation.

        Args:
            na_params: An instance of dataclass with parameters for adaptation

        Returns:
            None
        """
        return self.algorithm.execute(na_params.get_toolset(), "NoveltyAdaptation")

    def __call__(self, dataset: str, round_id: int) -> Union[Dict, None]:
        """
        Core logic for running round in OND.

        Args:
            algorithm: An instance of the algorithm
            dataset: Path to a file with the dataset for the round
            round_id: An Identifier for a round

        Returns:
            Score for the round
        """
        # Run feature extraction
        fe_params = FeatureExtractionParams(dataset, self.data_root, round_id)
        instance_ids = ONDRound.get_instance_ids(dataset)
        rfeature_dict, rlogit_dict = self._run_feature_extraction(
            fe_params, instance_ids
        )
        # Run World Change Detection
        wc_params = WorldChangeDetectionParams(
            rfeature_dict, rlogit_dict, round_id, self.redlight_instance
        )
        self._run_world_change_detection(wc_params, round_id)
        # Run Novelty Classification
        nc_params = NoveltyClassificationParams(rfeature_dict, rlogit_dict, round_id)
        self._run_novelty_classification(nc_params, round_id)
        # Compute metrics for the round
        round_score = self._evaluate_roundwise(round_id)

        na_params = NoveltyAdaptationParams(round_id)
        self._run_novelty_adaptation(na_params)

        return round_score
