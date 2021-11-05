"""Test for Visual Protocol."""

import logging
import os
import pickle as pkl
from typing import Union, Tuple, Dict, List
import ubelt as ub

from sail_on_client.harness.test_and_evaluation_harness import (
    TestAndEvaluationHarnessType,
)
from sail_on_client.protocol.visual_round import VisualRound
from sail_on_client.utils.decorators import skip_stage
from sail_on_client.protocol.ond_dataclasses import (
    AlgorithmAttributes as ONDAlgorithmAttributes,
)
from sail_on_client.protocol.condda_dataclasses import (
    AlgorithmAttributes as CONDDAAlgorithmAttributes,
)


log = logging.getLogger(__name__)


class VisualTest:
    """Class representing test in visual protocol."""

    def __init__(
        self,
        algorithm_attributes: Union[ONDAlgorithmAttributes, CONDDAAlgorithmAttributes],
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
        Construct visual test.

        Args:
            algorithm_attributes: An instance of algorithm_attributes
            data_root: Root directory for the dataset
            domain: Name of the domain for the test
            feature_dir: Directory to save features
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
        self.harness = harness
        self.feature_dir = feature_dir
        self.save_dir = save_dir
        self.session_id = session_id
        self.skip_stages = skip_stages
        self.use_consolidated_features = use_consolidated_features
        self.use_saved_features = use_saved_features

    def _restore_features(self, test_id: str) -> Tuple[Dict, Dict]:
        """
        Private function to restore features.

        Args:
           test_id: An identifier for the test

        Returns:
            Tuple of dictionary with features and logits obtained from the feature extractor
        """
        features_dict: Dict = {}
        logit_dict: Dict = {}
        algorithm_name = self.algorithm_attributes.name
        if self.use_saved_features:
            if os.path.isdir(self.feature_dir):
                if self.use_consolidated_features:
                    feature_fname = f"{algorithm_name}_features.pkl"
                else:
                    feature_fname = f"{test_id}_{algorithm_name}_features.pkl"
                feature_path = os.path.join(self.feature_dir, feature_fname)
                test_features = pkl.load(open(feature_path, "rb"))
            else:
                test_features = pkl.load(open(self.feature_dir, "rb"))
            features_dict = test_features["features_dict"]
            logit_dict = test_features["logit_dict"]
        return features_dict, logit_dict

    def _aggregate_features_across_round(
        self, round_instance: VisualRound, feature_dict: Dict, logit_dict: Dict
    ) -> Tuple[Dict, Dict]:
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
    def _save_features(
        self, test_id: str, feature_dict: Dict, logit_dict: Dict
    ) -> None:
        """
        Save features for a test.

        Args:
            test_id: An identifier for the test
            feature_dict: Features for the test
            logit_dict: Logit for the test

        Return:
            None
        """
        ub.ensuredir(self.feature_dir)
        algorithm_name = self.algorithm_attributes.name
        feature_path = os.path.join(
            self.feature_dir, f"{test_id}_{algorithm_name}_features.pkl"
        )
        log.info(f"Saving features in {feature_path}")
        with open(feature_path, "wb") as f:
            pkl.dump({"features_dict": feature_dict, "logit_dict": logit_dict}, f)
