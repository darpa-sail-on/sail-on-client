"""Test for CONDDA."""

import logging
from itertools import count
from typing import Dict, List

from sail_on_client.protocol.condda_dataclasses import (
    AlgorithmAttributes,
    InitializeParams,
)
from sail_on_client.protocol.visual_test import VisualTest
from sail_on_client.protocol.condda_round import CONDDARound
from sail_on_client.harness.test_and_evaluation_harness import (
    TestAndEvaluationHarnessType,
)
from sail_on_client.utils.utils import safe_remove
from sail_on_client.errors import RoundError


log = logging.getLogger(__name__)


class CONDDATest(VisualTest):
    """Class representing CONDDA Test."""

    def __init__(
        self,
        algorithm_attributes: AlgorithmAttributes,
        data_root: str,
        domain: str,
        feature_dir: str,
        harness: TestAndEvaluationHarnessType,
        save_dir: str,
        session_id: str,
        skip_stages: List[str],
        use_consolidated_features: bool,
        use_saved_features: bool,
    ) -> None:
        """
        Construct test for CONDDA.

        Args:
            algorithm_attributes: An instance of algorithm_attributes
            data_root: Root directory for the dataset
            domain: Name of the domain for the test
            feature_dir: Directory where features are stored
            harness: An Instance of harness used for T&E
            save_dir: The directory where features are saved
            session_id: Session identifier for the test
            skip_stages: List of stages that would be skipped
            use_consolidated_features: Flag for using consolidated features
            use_saved_features: Flag for using saved features

        Returns:
            None
        """
        super().__init__(
            algorithm_attributes,
            data_root,
            domain,
            feature_dir,
            harness,
            save_dir,
            session_id,
            skip_stages,
            use_consolidated_features,
            use_saved_features,
        )

    def __call__(self, test_id: str) -> None:
        """
        Core logic for running test in CONDDA.

        Args:
            test_id: An identifier for the test

        Returns:
            Score for the test
        """
        metadata = self.harness.get_test_metadata(self.session_id, test_id)
        redlight_instance = metadata.get("red_light", "")
        # Initialize algorithm
        algorithm_instance = self.algorithm_attributes.instance
        algorithm_parameters = self.algorithm_attributes.parameters
        algorithm_init_params = InitializeParams(
            algorithm_parameters, self.session_id, test_id
        )
        algorithm_instance.execute(algorithm_init_params.get_toolset(), "Initialize")

        # Restore features
        features_dict, logit_dict = self._restore_features(test_id)

        # Initialize Round
        round_instance = CONDDARound(
            algorithm_instance,
            self.data_root,
            features_dict,
            self.harness,
            logit_dict,
            redlight_instance,
            self.session_id,
            self.skip_stages,
            test_id,
        )
        aggregated_features_dict: Dict = {}
        aggregated_logit_dict: Dict = {}
        # Run algorithm for multiple rounds
        for round_id in count(0):
            log.info(f"Start round: {round_id}")
            # see if there is another round available
            try:
                dataset = self.harness.dataset_request(
                    test_id, round_id, self.session_id
                )
            except RoundError:
                # no more rounds available, this test is done.
                break
            round_instance(dataset, round_id)
            (
                aggregated_features_dict,
                aggregated_logit_dict,
            ) = self._aggregate_features_across_round(
                round_instance, aggregated_features_dict, aggregated_logit_dict
            )
            # cleanup the dataset file for the round
            safe_remove(dataset)
            log.info(f"Round complete: {round_id}")
        self.harness.complete_test(self.session_id, test_id)
        self._save_features(test_id, aggregated_features_dict, aggregated_logit_dict)
