"""OND protocol."""

from sail_on_client.agent.ond_agent import ONDAgent
from sail_on_client.harness.test_and_evaluation_harness import (
    TestAndEvaluationHarnessType,
)
from sail_on_client.protocol.visual_protocol import VisualProtocol
from sail_on_client.utils.numpy_encoder import NumpyEncoder
from sail_on_client.protocol.ond_dataclasses import AlgorithmAttributes
from sail_on_client.protocol.ond_test import ONDTest
from sail_on_client.utils.decorators import skip_stage

import os
import json
import logging

from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class ONDProtocol(VisualProtocol):
    """OND protocol."""

    def __init__(
        self,
        algorithms: Dict[str, ONDAgent],
        dataset_root: str,
        domain: str,
        harness: TestAndEvaluationHarnessType,
        save_dir: str,
        seed: str,
        test_ids: List[str],
        baseline_class: str = "",
        feature_extraction_only: bool = False,
        has_baseline: bool = False,
        has_reaction_baseline: bool = False,
        hints: List = None,
        is_eval_enabled: bool = False,
        is_eval_roundwise_enabled: bool = False,
        resume_session: bool = False,
        resume_session_ids: Dict = None,
        save_attributes: bool = False,
        saved_attributes: Dict = None,
        save_elementwise: bool = False,
        save_features: bool = False,
        feature_dir: str = "",
        skip_stages: List = None,
        use_feedback: bool = False,
        feedback_type: str = "classification",
        use_consolidated_features: bool = False,
        use_saved_attributes: bool = False,
        use_saved_features: bool = False,
    ) -> None:
        """
        Construct OND protocol.

        Args:
            algorithms: Dictionary of algorithms that are used run based on the protocol
            baseline_class: Name of the baseline class
            dataset_root: Root directory of the dataset
            domain: Domain of the problem
            save_dir: Directory where results are saved
            seed: Seed for the experiments
            feedback_type: Type of feedback
            test_ids: List of tests
            feature_extraction_only: Flag to only run feature extraction
            has_baseline: Flag to check if the session has baseline
            has_reaction_baseline: Flag to check if the session has reaction baseline
            hints: List of hint provided in the session
            harness: A harness for test and evaluation
            is_eval_enabled: Flag to check if evaluation is enabled in session
            is_eval_roundwise_enabled: Flag to check if evaluation is enabled for rounds
            resume_session: Flag to resume session
            resume_session_ids: Dictionary for resuming sessions
            save_attributes: Flag to save attributes
            saved_attributes: Dictionary for attributes
            save_elementwise: Flag to save features elementwise
            save_features: Flag to save  features
            feature_dir: Directory to save features
            skip_stages: List of stages that are skipped
            use_feedback: Flag to use feedback
            use_saved_attributes: Flag to use saved attributes
            use_saved_features: Flag to use saved features

        Returns:
            None
        """
        super().__init__(algorithms, harness)
        self.baseline_class = baseline_class
        self.dataset_root = dataset_root
        self.domain = domain
        self.feature_extraction_only = feature_extraction_only
        self.feature_dir = feature_dir
        self.feedback_type = feedback_type
        self.has_baseline = has_baseline
        self.has_reaction_baseline = has_reaction_baseline
        if hints is None:
            self.hints = []
        else:
            self.hints = hints
        self.is_eval_enabled = is_eval_enabled
        self.is_eval_roundwise_enabled = is_eval_roundwise_enabled
        self.resume_session = resume_session
        if resume_session_ids is None:
            self.resume_session_ids = {}
        else:
            self.resume_session_ids = resume_session_ids
        self.save_attributes = save_attributes
        if saved_attributes is None:
            self.saved_attributes = {}
        else:
            self.saved_attributes = saved_attributes
        self.save_dir = save_dir
        self.save_elementwise = save_elementwise
        self.save_features = save_features
        if skip_stages is None:
            self.skip_stages = []
        else:
            self.skip_stages = skip_stages
        self.seed = seed
        self.test_ids = test_ids
        self.use_consolidated_features = use_consolidated_features
        self.use_feedback = use_feedback
        self.use_saved_attributes = use_saved_attributes
        self.use_saved_features = use_saved_features

    def get_config(self) -> Dict:
        """Get dictionary representation of the object."""
        config = super().get_config()
        config.update(
            {
                "baseline_class": self.baseline_class,
                "dataset_root": self.dataset_root,
                "domain": self.domain,
                "feature_extraction_only": self.feature_extraction_only,
                "feature_dir": self.feature_dir,
                "feedback_type": self.feedback_type,
                "has_baseline": self.has_baseline,
                "has_reaction_baseline": self.has_reaction_baseline,
                "hints": self.hints,
                "is_eval_enabled": self.is_eval_enabled,
                "is_eval_roundwise_enabled": self.is_eval_roundwise_enabled,
                "resume_session": self.resume_session,
                "resume_session_ids": self.resume_session_ids,
                "save_attributes": self.save_attributes,
                "saved_attributes": self.saved_attributes,
                "save_dir": self.save_dir,
                "save_elementwise": self.save_elementwise,
                "save_features": self.save_features,
                "skip_stages": self.skip_stages,
                "seed": self.seed,
                "test_ids": self.test_ids,
                "use_feedback": self.use_feedback,
                "use_saved_attributes": self.use_saved_attributes,
                "use_saved_features": self.use_saved_features,
            }
        )
        return config

    def create_algorithm_attributes(
        self,
        algorithm_name: str,
        algorithm_param: Dict,
        baseline_algorithm_name: str,
        has_baseline: bool,
        has_reaction_baseline: bool,
        test_ids: List[str],
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
        algorithm_instance = self.algorithms[algorithm_name]
        is_baseline = algorithm_name == baseline_algorithm_name
        session_id = self.resume_session_ids.get(algorithm_name, "")
        return AlgorithmAttributes(
            algorithm_name,
            algorithm_param.get("detection_threshold", 0.5),
            algorithm_instance,
            has_baseline and is_baseline,
            has_reaction_baseline and is_baseline,
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

    def _find_baseline_session_id(
        self, algorithms_attributes: List[AlgorithmAttributes]
    ) -> str:
        """
        Find baseline session id based on the attributes of algorithms.

        Args:
            algorithms_attributes: List of algorithm attributes

        Returns:
            Baseline session id
        """
        for algorithm_attributes in algorithms_attributes:
            if (
                algorithm_attributes.is_baseline
                or algorithm_attributes.is_reaction_baseline
            ):
                return algorithm_attributes.session_id
        raise Exception(
            "Failed to find baseline, this is required to compute reaction perfomance"
        )

    @skip_stage("EvaluateAlgorithms")
    def _evaluate_algorithms(
        self,
        algorithms_attributes: List[AlgorithmAttributes],
        algorithm_scores: Dict,
        save_dir: str,
    ) -> None:
        """
        Evaluate algorithms after all tests have been submitted.

        Args:
            algorithms_attributes: All algorithms present in the config
            algorithm_scores: Scores for round wise evaluation
            save_dir: Directory where the scores are stored

        Returns:
            None
        """
        if self.has_baseline or self.has_reaction_baseline:
            baseline_session_id: Optional[str] = self._find_baseline_session_id(
                algorithms_attributes
            )
        else:
            baseline_session_id = None
        for algorithm_attributes in algorithms_attributes:
            if (
                algorithm_attributes.is_baseline
                or algorithm_attributes.is_reaction_baseline
            ):
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
                with open(
                    os.path.join(save_dir, f"{test_id}_{algorithm_name}.json"), "w"
                ) as f:  # type: ignore
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
        feature_extraction_only: bool,
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

    def run_protocol(self, config: Dict) -> None:
        """
        Run the protocol.

        Args:
            config: Parameters provided in the config

        Returns:
            None
        """
        log.info("Starting OND")
        self.skip_stages = self.update_skip_stages(
            self.skip_stages,
            self.is_eval_enabled,
            self.is_eval_roundwise_enabled,
            self.use_feedback,
            self.save_features,
            self.feature_extraction_only,
        )

        algorithms_attributes = []
        # Populate most of the attributes for the algorithm
        for algorithm_name in self.algorithms.keys():
            algorithm_param = self.algorithms[algorithm_name].get_config()
            algorithm_attributes = self.create_algorithm_attributes(
                algorithm_name,
                algorithm_param,
                self.baseline_class,
                self.has_baseline,
                self.has_reaction_baseline,
                self.test_ids,
            )
            log.info(f"Consolidating attributes for {algorithm_name}")
            algorithms_attributes.append(algorithm_attributes)

        # Create sessions an instances of all the algorithms and populate
        # session_id for algorithm attributes
        for idx, algorithm_attributes in enumerate(algorithms_attributes):
            algorithms_attributes[idx] = self.create_algorithm_session(
                algorithm_attributes,
                self.domain,
                self.hints,
                self.resume_session,
                "OND",
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
            ond_test = ONDTest(
                algorithm_attributes,
                self.dataset_root,
                self.domain,
                self.feedback_type,
                self.feature_dir,
                self.harness,
                self.save_dir,
                session_id,
                skip_stages,
                self.use_consolidated_features,
                self.use_saved_features,
            )
            test_scores = {}
            for test_id in test_ids:
                log.info(f"Start test: {test_id}")
                test_score = ond_test(test_id)
                test_scores[test_id] = test_score
                log.info(f"Test complete: {test_id}")
            algorithm_scores[algorithm_name] = test_scores

        # Evaluate algorithms
        self._evaluate_algorithms(
            algorithms_attributes, algorithm_scores, self.save_dir
        )

        # Terminate algorithms
        for algorithm_attributes in algorithms_attributes:
            algorithm_name = algorithm_attributes.name
            session_id = algorithm_attributes.session_id
            self.harness.terminate_session(session_id)
            log.info(f"Session ended for {algorithm_name}: {session_id}")
