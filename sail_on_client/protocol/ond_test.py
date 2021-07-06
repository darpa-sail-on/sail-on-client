"""Test for OND."""

import logging
import os
import pickle as pkl
from itertools import count
from typing import Union, Tuple, Dict, Any, List
import ubelt as ub

from sail_on_client.protocol.ond_dataclasses import AlgorithmAttributes, InitializeParams, NoveltyCharacterizationParams
from sail_on_client.protocol.ond_round import ONDRound
from sail_on_client.feedback import create_feedback_instance, feedback_type
from sail_on_client.protocol.parinterface import ParInterface
from sail_on_client.protocol.localinterface import LocalInterface
from sail_on_client.utils.utils import safe_remove
from sail_on_client.utils.decorators import skip_stage
from sail_on_client.errors import RoundError


log = logging.getLogger(__name__)


class ONDTest:
    """Class Representing OND test."""

    def __init__(
            self,
            algorithm_attributes: AlgorithmAttributes,
            data_root: str,
            domain: str,
            feedback_type: str,
            harness: Union[LocalInterface, ParInterface],
            save_dir: str,
            session_id: str,
            skip_stages: List[str],
            use_consolidated_features: bool,
            use_saved_features: bool) -> None:

        """
        Constructor for test for OND.

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
        self.algorithm_attributes = algorithm_attributes
        self.data_root = data_root
        self.domain = domain
        self.feedback_type = feedback_type
        self.harness = harness
        self.save_dir = save_dir
        self.session_id = session_id
        self.skip_stages = skip_stages
        self.use_consolidated_features = use_consolidated_features
        self.use_saved_features = use_saved_features

    @skip_stage("CreateFeedbackInstance")
    def _create_feedback_instance(self, test_id) -> feedback_type:
        """
        Private function for creating feedback object.

        Args:
           test_id: An identifier for the test

        Return:
            An instance of feedback for the domain
        """
        algorithm_parameters = self.algorithm_attributes.parameters
        feedback_params = algorithm_parameters["feedback_params"]
        log.info("Creating Feedback object")
        feedback_params = {
                "interface": self.harness,
                "session_id": self.session_id,
                "test_id": test_id,
                "feedback_type": self.feedback_type
                }
        feedback_instance = create_feedback_instance(
            self.domain, feedback_params
        )
        return feedback_instance

    def _restore_features(
            self,
            test_id: str) -> Tuple[Dict, Dict]:
        """
        Private function to _restore_features.

        Args:
           test_id: An identifier for the test

        Returns:
            Tuple of dictionary with features and logits obtained from the feature extractor
        """
        features_dict: Dict = {}
        logit_dict: Dict = {}
        algorithm_name = self.algorithm_attributes.name
        if self.use_saved_features:
            if os.path.isdir(self.save_dir):
                if self.use_consolidated_features:
                    feature_fname = f"{algorithm_name}_features.pkl"
                else:
                    feature_fname = f"{test_id}_{algorithm_name}_features.pkl"
                feature_path = os.path.join(self.save_dir, feature_fname)
                test_features = pkl.load(open(feature_path, "rb"))
            else:
                test_features = pkl.load(open(self.save_dir, "rb"))
            features_dict = test_features["features_dict"]
            logit_dict = test_features["logit_dict"]
        return features_dict, logit_dict

    def _aggregate_features_across_round(
            self,
            round_instance: ONDRound,
            feature_dict: Dict,
            logit_dict: Dict) -> Tuple[Dict, Dict]:
        """
        Aggregate features across multiple rounds.

        Args:
            round_instance: Instance of ond round
            feature_dict: Aggregated features until this function was called
            logit_dict: Aggregated logit until this function was called

        Return:
            Tuple of features and logits with features and logits from the round
        """
        feature_dict.update(getattr(round_instance, "rfeature_dict", {}))
        logit_dict.update(getattr(round_instance, "rlogit_dict", {}))
        return feature_dict, logit_dict

    @skip_stage("SaveFeatures")
    def _save_features(self,
                       test_id: str,
                       feature_dict: Dict,
                       logit_dict: Dict) -> None:
        """
        Save features for a test.

        Args:
            test_id: An identifier for the test
            feature_dict: Features for the test
            logit_dict: Logit for the test

        Return:
            None
        """
        ub.ensuredir(self.save_dir)
        algorithm_name = self.algorithm_attributes.name
        feature_path = os.path.join(
            self.save_dir, f"{test_id}_{algorithm_name}_features.pkl"
        )
        log.info(f"Saving features in {feature_path}")
        with open(feature_path, "wb") as f:
            pkl.dump({"features_dict": feature_dict, "logit_dict": logit_dict}, f)

    @skip_stage("NoveltyCharacterization")
    def _run_novelty_characterization(self,
                                      algorithm: Any,
                                      nc_params, test_id) -> None:
        characterization_results = algorithm.execute(nc_params.get_toolset(),
                                                     "NoveltyCharacterization")
        if characterization_results:
            if isinstance(characterization_results, dict):
                self.harness.post_results(characterization_results,
                                          test_id, 0, self.session_id)
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
        metadata = self.harness.get_test_metadata(
            self.session_id, test_id
        )
        redlight_instance = metadata.get("red_light", "")
        # Initialize feedback object for the domains
        feedback_instance = self._create_feedback_instance(test_id)

        # Initialize algorithm
        algorithm_instance = self.algorithm_attributes.instance
        algorithm_parameters = self.algorithm_attributes.parameters
        algorithm_init_params = InitializeParams(algorithm_parameters,
                                                 self.session_id,
                                                 test_id,
                                                 feedback_instance)
        algorithm_instance.execute(algorithm_init_params.get_toolset(), "Initialize")

        # Restore features
        features_dict, logit_dict = self._restore_features(test_id)

        # Initialize Round
        round_instance = ONDRound(algorithm_instance, self.data_root, features_dict,
                                  self.harness, logit_dict, redlight_instance,
                                  self.session_id, self.skip_stages, test_id)
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
            aggregated_features_dict, aggregated_logit_dict = \
                    self._aggregate_features_across_round(round_instance,
                                                          aggregated_features_dict,
                                                          aggregated_logit_dict)
            # cleanup the dataset file for the round
            safe_remove(dataset)
            log.info(f"Round complete: {round_id}")
        nc_params = NoveltyCharacterizationParams(test_instances)
        self._run_novelty_characterization(algorithm_instance, nc_params, test_id)
        self.harness.complete_test(self.session_id, test_id)
        return test_score
