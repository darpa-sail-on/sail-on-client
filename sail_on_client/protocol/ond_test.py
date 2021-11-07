"""Test for OND."""

import logging
from itertools import count
from typing import Union, Dict, Any, List

from sail_on_client.protocol.ond_dataclasses import (
    AlgorithmAttributes,
    InitializeParams,
    NoveltyCharacterizationParams,
)
from sail_on_client.protocol.ond_round import ONDRound
from sail_on_client.protocol.visual_test import VisualTest
from sail_on_client.feedback import create_feedback_instance, feedback_type
from sail_on_client.harness.test_and_evaluation_harness import (
    TestAndEvaluationHarnessType,
)
from sail_on_client.utils.utils import safe_remove
from sail_on_client.utils.decorators import skip_stage
from sail_on_client.errors import RoundError


log = logging.getLogger(__name__)


class ONDTest(VisualTest):
    """Class Representing OND test."""

    def __init__(
        self,
        algorithm_attributes: AlgorithmAttributes,
        data_root: str,
        domain: str,
        feedback_type: str,
        feature_dir: str,
        harness: TestAndEvaluationHarnessType,
        save_dir: str,
        session_id: str,
        skip_stages: List[str],
        use_consolidated_features: bool,
        use_saved_features: bool,
    ) -> None:
        """
        Construct test for OND.

        Args:
            algorithm_attributes: An instance of algorithm_attributes
            data_root: Root directory for the dataset
            domain: Name of the domain for the test
            feedback_type: Type of feedback used in the test
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
        self.feedback_type = feedback_type

    @skip_stage("CreateFeedbackInstance")
    def _create_feedback_instance(
        self, test_id: str, feedback_max_ids: int
    ) -> feedback_type:
        """
        Private function for creating feedback object.

        Args:
           test_id: An identifier for the test
           feedback_max_ids: Budget provided in metadata

        Return:
            An instance of feedback for the domain
        """
        log.info("Creating Feedback object")
        if feedback_max_ids == 0:
            log.warn(
                """feedback_max_ids was missing from metadata, thus setting
                        the feedback budget to 0"""
            )
        feedback_params = {
            "first_budget": feedback_max_ids,
            "income_per_batch": feedback_max_ids,
            "maximum_budget": feedback_max_ids,
            "interface": self.harness,
            "session_id": self.session_id,
            "test_id": test_id,
            "feedback_type": self.feedback_type,
        }
        feedback_instance = create_feedback_instance(self.domain, feedback_params)
        return feedback_instance

    @skip_stage("NoveltyCharacterization")
    def _run_novelty_characterization(
        self, algorithm: Any, nc_params: NoveltyCharacterizationParams, test_id: str
    ) -> None:
        characterization_results = algorithm.execute(
            nc_params.get_toolset(), "NoveltyCharacterization"
        )
        if characterization_results:
            if isinstance(characterization_results, dict):
                self.harness.post_results(
                    characterization_results, test_id, 0, self.session_id
                )
            else:
                results = {"characterization": characterization_results}
                self.harness.post_results(results, test_id, 0, self.session_id)
        else:
            log.warn("No characterization result provided by the algorithm")

    def __call__(self, test_id: str) -> Union[Dict, None]:
        """
        Core logic for running test in OND.

        Args:
            test_id: An identifier for the test

        Returns:
            Score for the test
        """
        metadata = self.harness.get_test_metadata(self.session_id, test_id)
        redlight_instance = metadata.get("red_light", "")
        feedback_max_ids = metadata.get("feedback_max_ids", 0)
        # Initialize feedback object for the domains
        feedback_instance = self._create_feedback_instance(test_id, feedback_max_ids)

        # Initialize algorithm
        algorithm_instance = self.algorithm_attributes.instance
        algorithm_parameters = self.algorithm_attributes.parameters
        algorithm_init_params = InitializeParams(
            algorithm_parameters, self.session_id, test_id, feedback_instance
        )
        algorithm_instance.execute(algorithm_init_params.get_toolset(), "Initialize")

        # Restore features
        features_dict, logit_dict = self._restore_features(test_id)

        # Initialize Round
        round_instance = ONDRound(
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
        test_score = {}
        test_instances = []
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
            round_score = round_instance(dataset, round_id)
            test_instances.extend(ONDRound.get_instance_ids(dataset))
            if round_score:
                test_score[f"Round {round_id}"] = round_score
            (
                aggregated_features_dict,
                aggregated_logit_dict,
            ) = self._aggregate_features_across_round(
                round_instance, aggregated_features_dict, aggregated_logit_dict
            )
            # cleanup the dataset file for the round
            safe_remove(dataset)
            log.info(f"Round complete: {round_id}")
        nc_params = NoveltyCharacterizationParams(test_instances)
        self._run_novelty_characterization(algorithm_instance, nc_params, test_id)
        self.harness.complete_test(self.session_id, test_id)
        self._save_features(test_id, aggregated_features_dict, aggregated_logit_dict)
        return test_score
