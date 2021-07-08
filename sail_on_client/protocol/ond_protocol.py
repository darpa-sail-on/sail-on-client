"""OND protocol."""

from sail_on_client.protocol.visual_protocol import VisualProtocol
from sail_on_client.protocol.ond_config import OndConfig
from sail_on_client.utils.utils import (
    update_harness_parameters,
)
from sail_on_client.utils.numpy_encoder import NumpyEncoder
from sail_on_client.protocol.parinterface import ParInterface
from sail_on_client.protocol.localinterface import LocalInterface
from sail_on_client.protocol.ond_dataclasses import AlgorithmAttributes
from sail_on_client.protocol.ond_test import ONDTest
from sail_on_client.utils.decorators import skip_stage

import os
import json
import logging

from typing import Dict, List, Any, Union

log = logging.getLogger(__name__)


class SailOn(VisualProtocol):
    """OND protocol."""

    def __init__(
        self,
        discovered_plugins: Dict[str, Any],
        algorithmsdirectory: str,
        harness: Union[ParInterface, LocalInterface],
        config_file: str,
    ) -> None:
        """
        Construct OND protocol.

        Args:
            discovered_plugins: Dict of algorithms that can be used by the protocols
            algorithmsdirectory: Directory with the algorithms
            harness: An object for the harness used for T&E
            config_file: Path to a config file used by the protocol

        Returns:
            None
        """
        super().__init__(discovered_plugins,
                         algorithmsdirectory,
                         harness, config_file)
        with open(config_file, "r") as f:
            overriden_config = json.load(f)
        self.config = OndConfig(overriden_config)
        self.harness = update_harness_parameters(harness, self.config["harness_config"])

    def create_algorithm_attributes(
            self,
            algorithm_name: str,
            algorithm_param: Dict,
            baseline_algorithm_name: str,
            has_baseline: bool,
            has_reaction_baseline: bool,
            test_ids) -> AlgorithmAttributes:
        """
        Create an instance of algorithm attributes.

        Args:
            algorithm_name: Name of the algorithm
            algorithm_param: Parameters for the algorithm
            baseline_algorithm_name: Name of the baseline algorithm
            has_baseline: Flag to check if a baseline is present in the config
            has_reaction_baseline: Flag to check if a reaction baseline is present in the config
            test_ids: List of test

        Returns:
            An instance of AlgorithmAttributes
        """
        algorithm_instance = self.get_algorithm(
            algorithm_name,
            algorithm_param,
        )
        is_baseline = algorithm_name == baseline_algorithm_name
        session_id = self.config.get("resumed_session_ids", {}).get(algorithm_name, "")
        return AlgorithmAttributes(
                algorithm_name,
                algorithm_param.get("detection_threshold", 0.5),
                algorithm_instance,
                has_baseline and is_baseline,
                has_reaction_baseline and is_baseline,
                algorithm_param.get("package_name", None),
                algorithm_param,
                session_id,
                test_ids)

    def _find_baseline_session_id(
            self,
            algorithms_attributes: List[AlgorithmAttributes]) -> str:
        """
        Find baseline session id based on the attributes of algorithms.

        Args:
            algorithms_attributes: List of algorithm attributes

        Returns:
            Baseline session id
        """
        for algorithm_attributes in algorithms_attributes:
            if algorithm_attributes.is_baseline or algorithm_attributes.is_reaction_baseline:
                return algorithm_attributes.session_id
        raise Exception("Failed to find baseline, this is required to compute reaction perfomance")

    @skip_stage("EvaluateAlgorithms")
    def _evaluate_algorithms(
            self,
            algorithms_attributes: List[AlgorithmAttributes],
            algorithm_scores: Dict,
            save_dir: str) -> None:
        """
        Evaluate algorithms after all tests have been submitted.

        Args:
            algorithms_attributes: All algorithms present in the config
            algorithm_scores: Scores for round wise evaluation
            save_dir: Directory where the scores are stored

        Returns:
            None
        """
        baseline_session_id = self._find_baseline_session_id(algorithms_attributes)
        for algorithm_attributes in algorithms_attributes:
            if algorithm_attributes.is_baseline or algorithm_attributes.is_reaction_baseline:
                continue
            session_id = algorithm_attributes.session_id
            test_ids = algorithm_attributes.test_ids
            algorithm_name = algorithm_attributes.name
            test_scores = algorithm_scores[algorithm_name]
            log.info(f"Started evaluating {algorithm_name}")
            for test_id in test_ids:
                score = self.harness.evaluate(
                    test_id, 0, session_id, baseline_session_id
                )
                score.update(test_scores[test_id])
                with open(os.path.join(save_dir,
                          f"{test_id}_{algorithm_name}.json"), "w") as f:  # type: ignore
                    log.info(f"Saving results in {save_dir}")
                    json.dump(score, f, indent=4, cls=NumpyEncoder)  # type: ignore
            log.info(f"Finished evaluating {algorithm_name}")

    def update_skip_stages(
            self,
            skip_stages: List[str],
            is_eval_enabled: bool,
            is_eval_roundwise_enabled: bool,
            use_feedback: bool,
            save_features: bool,
            feature_extraction_only: bool) -> List[str]:
        """
        Update skip stages based on the boolean values in config.

        Args:
            skip_stages: List of skip stages specified in the config
            is_eval_enabled: Flag to enable evaluation
            is_eval_roundwise_enabled: Flag to enable evaluation in every round
            use_feedback: Flag to enable using feedback
            save_features: Flag to enable saving features
            feature_extraction_only: Flag to only run feature extraction

        Returns:
            Update list of skip stages
        """
        if not is_eval_enabled:
            skip_stages.append("EvaluateAlgorithms")
            skip_stages.append("EvaluateRoundwise")
        if not is_eval_roundwise_enabled:
            skip_stages.append("EvaluateRoundwise")
        if not use_feedback:
            skip_stages.append("CreateFeedbackInstance")
            skip_stages.append("NoveltyAdaptation")
        if not save_features:
            skip_stages.append("SaveFeatures")
        if feature_extraction_only:
            skip_stages.append("CreateFeedbackInstance")
            skip_stages.append("WorldDetection")
            skip_stages.append("NoveltyClassification")
            skip_stages.append("NoveltyAdaptation")
            skip_stages.append("NoveltyCharacterization")
        return skip_stages

    def run_protocol(self) -> None:
        """Run the protocol."""
        log.info("Starting OND")
        # provide all of the configuration information in the toolset
        self.toolset.update(self.config)
        detector_params = self.config["detectors"]
        has_reaction_baseline = detector_params["has_reaction_baseline"]
        has_baseline = detector_params["has_baseline"]
        save_dir = detector_params["csv_folder"]
        algorithm_params = detector_params["detector_configs"]
        algorithm_names = algorithm_params.keys()
        baseline_algorithm_name = detector_params.get("baseline_class", None)
        test_ids = self.config["test_ids"]
        domain = self.config["domain"]
        hints = self.config["hints"]
        resume_session = self.config["resume_session"]
        is_eval_enabled = self.config["is_eval_enabled"]
        is_eval_roundwise_enabled = self.config["is_eval_roundwise_enabled"]
        data_root = self.config["dataset_root"]
        save_dir = self.config["save_dir"]
        save_features = self.config["save_features"]
        use_feedback = self.config["use_feedback"]
        feedback_type = self.config["feedback_type"]
        use_saved_features = self.config["use_saved_features"]
        use_consolidated_features = self.config["use_consolidated_features"]
        feature_extraction_only = self.config["feature_extraction_only"]
        self.skip_stages = self.config["skip_stages"]
        self.skip_stages = self.update_skip_stages(self.skip_stages,
                                                   is_eval_enabled,
                                                   is_eval_roundwise_enabled,
                                                   use_feedback,
                                                   save_features,
                                                   feature_extraction_only)

        algorithms_attributes = []

        # Populate most of the attributes for the algorithm
        for algorithm_name in algorithm_names:
            algorithm_param = algorithm_params[algorithm_name]
            algorithm_attributes = self.create_algorithm_attributes(
                        algorithm_name,
                        algorithm_param,
                        baseline_algorithm_name,
                        has_baseline,
                        has_reaction_baseline,
                        test_ids
                    )
            # Add common parameters to algorithm specific config with some exclusions
            algorithm_attributes.merge_detector_params(detector_params,
                                                       ["has_baseline",
                                                        "has_reaction_baseline",
                                                        "baseline_class",
                                                        "detector_configs"])
            log.info(f"Consolidating attributes for {algorithm_name}")
            algorithms_attributes.append(algorithm_attributes)

        # Create sessions an instances of all the algorithms and populate
        # session_id for algorithm attributes
        for idx, algorithm_attributes in enumerate(algorithms_attributes):
            algorithms_attributes[idx] = self.create_algorithm_session(
                                                            algorithm_attributes,
                                                            domain,
                                                            hints,
                                                            resume_session,
                                                            "OND"
                                                    )

        # Run tests for all the algorithms
        algorithm_scores = {}
        for algorithm_attributes in algorithms_attributes:
            algorithm_name = algorithm_attributes.name
            session_id = algorithm_attributes.session_id
            test_ids = algorithm_attributes.test_ids
            log.info(f"Starting session: {session_id} for algorithm: {algorithm_name}")
            skip_stages = self.skip_stages.copy()
            if algorithm_attributes.is_reaction_baseline:
                skip_stages.append("WorldDetection")
                skip_stages.append("NoveltyCharacterization")
            ond_test = ONDTest(algorithm_attributes, data_root, domain, feedback_type,
                               self.harness, save_dir, session_id, skip_stages,
                               use_consolidated_features, use_saved_features)
            test_scores = {}
            for test_id in test_ids:
                log.info(f"Start test: {test_id}")
                test_score = ond_test(test_id)
                test_scores[test_id] = test_score
                log.info(f"Test complete: {test_id}")
            algorithm_scores[algorithm_name] = test_scores

        # Evaluate algorithms
        self._evaluate_algorithms(algorithms_attributes, algorithm_scores, save_dir)

        # Terminate algorithms
        for algorithm_attributes in algorithms_attributes:
            algorithm_name = algorithm_attributes.name
            session_id = algorithm_attributes.session_id
            self.harness.terminate_session(session_id)
            log.info(f"Session ended for {algorithm_name}: {session_id}")
