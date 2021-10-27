"""Round for CONDDA."""

import logging
from typing import List, Dict, Any

from sail_on_client.harness.test_and_evaluation_harness import (
    TestAndEvaluationHarnessType,
)
from sail_on_client.utils.decorators import skip_stage
from sail_on_client.protocol.visual_round import VisualRound
from sail_on_client.protocol.condda_dataclasses import NoveltyCharacterizationParams
from sail_on_client.protocol.visual_dataclasses import (
    FeatureExtractionParams,
    WorldChangeDetectionParams,
)


log = logging.getLogger(__name__)


class CONDDARound(VisualRound):
    """Class Representing a round for CONDDA."""

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
        Construct CONDDARound.

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

    @skip_stage("NoveltyCharacterization")
    def _run_novelty_characterization(
        self, nc_params: NoveltyCharacterizationParams, round_id: int
    ) -> None:
        characterization_results = self.algorithm.execute(
            nc_params.get_toolset(), "NoveltyCharacterization"
        )
        if characterization_results:
            if isinstance(characterization_results, dict):
                self.harness.post_results(
                    characterization_results, self.test_id, round_id, self.session_id
                )
            else:
                results = {"characterization": characterization_results}
                self.harness.post_results(
                    results, self.test_id, round_id, self.session_id
                )
        else:
            log.warn("No characterization result provided by the algorithm")

    def __call__(self, dataset: str, round_id: int) -> None:
        """
        Core logic for running round in CONDDA.

        Args:
            dataset: Path to a file with the dataset for the round
            round_id: An Identifier for a round

        Returns:
            None
        """
        # Run feature extraction
        fe_params = FeatureExtractionParams(dataset, self.data_root, round_id)
        instance_ids = CONDDARound.get_instance_ids(dataset)
        rfeature_dict, rlogit_dict = self._run_feature_extraction(
            fe_params, instance_ids
        )
        # Run World Change Detection
        wc_params = WorldChangeDetectionParams(
            rfeature_dict, rlogit_dict, round_id, self.redlight_instance
        )
        self._run_world_change_detection(wc_params, round_id)
        # Run Novelty Classification
        nc_params = NoveltyCharacterizationParams(rfeature_dict, rlogit_dict, round_id)
        self._run_novelty_characterization(nc_params, round_id)
