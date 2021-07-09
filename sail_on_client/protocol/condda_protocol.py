"""CONDDA protocol."""

from sail_on_client.protocol.visual_protocol import VisualProtocol
from sail_on_client.protocol.condda_config import ConddaConfig
from sail_on_client.utils.utils import update_harness_parameters
from sail_on_client.protocol.parinterface import ParInterface
from sail_on_client.protocol.localinterface import LocalInterface
from sail_on_client.protocol.condda_dataclasses import AlgorithmAttributes
from sail_on_client.protocol.condda_test import CONDDATest

import os
import json
import sys
import logging

from typing import Dict, Any, Union, List

log = logging.getLogger(__name__)


class Condda(VisualProtocol):
    """CONDDA protocol."""

    def __init__(
        self,
        discovered_plugins: Dict[str, Any],
        algorithmsdirectory: str,
        harness: Union[ParInterface, LocalInterface],
        config_file: str,
    ) -> None:
        """
        Initialize condda protocol object.

        Args:
            discovered_plugins: Dict of algorithms that can be used by the protocols
            algorithmsdirectory: Directory with the algorithms
            harness: An object for the harness used for T&E
            config_file: Path to a config file used by the protocol

        Returns:
            None
        """

        super().__init__(discovered_plugins, algorithmsdirectory, harness, config_file)
        if not os.path.exists(config_file):
            log.error(f"{config_file} does not exist")
            sys.exit(1)

        with open(config_file, "r") as f:
            overriden_config = json.load(f)
        self.config = ConddaConfig(overriden_config)
        self.harness = update_harness_parameters(harness, self.config["harness_config"])

    def create_algorithm_attributes(
        self, algorithm_name: str, algorithm_param: Dict, test_ids: List[str]
    ) -> AlgorithmAttributes:
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
        algorithm_instance = self.get_algorithm(algorithm_name, algorithm_param,)
        session_id = self.config.get("resumed_session_ids", {}).get(algorithm_name, "")
        return AlgorithmAttributes(
            algorithm_name,
            algorithm_param.get("detection_threshold", 0.5),
            algorithm_instance,
            algorithm_param.get("package_name", None),
            algorithm_param,
            session_id,
            test_ids,
        )

    def create_algorithm_session(
        self,
        algorithm_attributes: AlgorithmAttributes,
        domain: str,
        hints: List[str],
        has_a_session: bool,
        protocol_name: str,
    ) -> AlgorithmAttributes:
        """
        Create/resume session for an algorithm.

        Args:
            algorithm_attributes: An instance of AlgorithmAttributes
            domain: Domain for the algorithm
            hints: List of hints used in the session
            has_a_session: Already has a session and we want to resume it
            protocol_name: Name of the algorithm

        Returns:
            An AlgorithmAttributes object with updated session id or test id
        """
        test_ids = algorithm_attributes.test_ids
        named_version = algorithm_attributes.named_version()
        detection_threshold = algorithm_attributes.detection_threshold

        if has_a_session:
            session_id = algorithm_attributes.session_id
            finished_test = self.harness.resume_session(session_id)
            algorithm_attributes.remove_completed_tests(finished_test)
            log.info(f"Resumed session {session_id} for {algorithm_attributes.name}")
        else:
            session_id = self.harness.session_request(
                test_ids,
                protocol_name,
                domain,
                named_version,
                hints,
                detection_threshold,
            )
            algorithm_attributes.session_id = session_id
            log.info(f"Created session {session_id} for {algorithm_attributes.name}")
        return algorithm_attributes

    def update_skip_stages(
        self, skip_stages: List[str], save_features: bool, feature_extraction_only: bool
    ) -> List[str]:
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
        if not save_features:
            skip_stages.append("SaveFeatures")
        if feature_extraction_only:
            skip_stages.append("WorldDetection")
            skip_stages.append("NoveltyCharacterization")
        return skip_stages

    def run_protocol(self) -> None:
        """Run protocol."""
        log.info("Starting CONDDA")
        # provide all of the configuration information in the toolset
        self.toolset.update(self.config)
        detector_params = self.config["detectors"]
        save_dir = detector_params["csv_folder"]
        algorithm_params = detector_params["detector_configs"]
        algorithm_names = algorithm_params.keys()
        test_ids = self.config["test_ids"]
        domain = self.config["domain"]
        hints = self.config["hints"]
        resume_session = self.config["resume_session"]
        data_root = self.config["dataset_root"]
        save_dir = self.config["save_dir"]
        save_features = self.config["save_features"]
        use_saved_features = self.config["use_saved_features"]
        use_consolidated_features = self.config["use_consolidated_features"]
        feature_extraction_only = self.config["feature_extraction_only"]
        self.skip_stages = self.config["skip_stages"]
        self.skip_stages = self.update_skip_stages(
            self.skip_stages, save_features, feature_extraction_only
        )

        algorithms_attributes = []
        # Populate most of the attributes for the algorithm
        for algorithm_name in algorithm_names:
            algorithm_param = algorithm_params[algorithm_name]
            algorithm_attributes = self.create_algorithm_attributes(
                algorithm_name, algorithm_param, test_ids
            )
            # Add common parameters to algorithm specific config with some exclusions
            algorithm_attributes.merge_detector_params(
                detector_params, ["detector_configs"]
            )
            log.info(f"Consolidating attributes for {algorithm_name}")
            algorithms_attributes.append(algorithm_attributes)

        # Create sessions an instances of all the algorithms and populate
        # session_id for algorithm attributes
        for idx, algorithm_attributes in enumerate(algorithms_attributes):
            algorithms_attributes[idx] = self.create_algorithm_session(
                algorithm_attributes, domain, hints, resume_session, "CONDDA"
            )

        # Run tests for all the algorithms
        for algorithm_attributes in algorithms_attributes:
            algorithm_name = algorithm_attributes.name
            session_id = algorithm_attributes.session_id
            test_ids = algorithm_attributes.test_ids
            log.info(f"Starting session: {session_id} for algorithm: {algorithm_name}")
            skip_stages = self.skip_stages.copy()
            condda_test = CONDDATest(
                algorithm_attributes,
                data_root,
                domain,
                self.harness,
                save_dir,
                session_id,
                skip_stages,
                use_consolidated_features,
                use_saved_features,
            )
            for test_id in test_ids:
                log.info(f"Start test: {test_id}")
                condda_test(test_id)
                log.info(f"Test complete: {test_id}")

        # Terminate algorithms
        for algorithm_attributes in algorithms_attributes:
            algorithm_name = algorithm_attributes.name
            session_id = algorithm_attributes.session_id
            self.harness.terminate_session(session_id)
            log.info(f"Session ended for {algorithm_name}: {session_id}")
