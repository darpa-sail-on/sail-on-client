"""Round for CONDDA."""

import logging
from typing import Union, List, Dict, Any, Tuple

from sail_on_client.protocol.parinterface import ParInterface
from sail_on_client.protocol.localinterface import LocalInterface
from sail_on_client.utils.utils import safe_remove
from sail_on_client.utils.decorators import skip_stage
from sail_on_client.protocol.condda_dataclasses import (FeatureExtractionParams,
                                                        WorldChangeDetectionParams,
                                                        NoveltyCharacterizationParams)


log = logging.getLogger(__name__)


class CONDDARound:
    """Class Representing a round for CONDDA."""

    def __init__(
            self,
            algorithm: Any,
            data_root: str,
            features_dict: Dict,
            harness: Union[LocalInterface, ParInterface],
            logit_dict: Dict,
            redlight_instance: str,
            session_id: str,
            skip_stages: List[str],
            test_id: str) -> None:
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
            self,
            fe_params: FeatureExtractionParams,
            instance_ids: List[str]) -> Tuple[Dict, Dict]:
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
            rfeature_dict, rlogit_dict = self.algorithm.execute(fe_toolset,
                                                                "FeatureExtraction")
        self.rfeature_dict, self.rlogit_dict = rfeature_dict, rlogit_dict
        return rfeature_dict, rlogit_dict

    @skip_stage("WorldDetection")
    def _run_world_change_detection(
            self,
            wcd_params: WorldChangeDetectionParams,
            round_id: int,
            ) -> None:
        """
        Private helper function for detecting that the world has changed.

        Args:
            wcd_params: An instance of dataclass with parameters for world change detection
            round_id: Identifier for a round

        Returns:
            None
        """
        wd_result = self.algorithm.execute(wcd_params.get_toolset(),
                                           "WorldDetection")
        self.harness.post_results({"detection": wd_result}, self.test_id, round_id,
                                  self.session_id)
        safe_remove(wd_result)

    @skip_stage("NoveltyCharacterization")
    def _run_novelty_characterization(self,
                                      nc_params: NoveltyCharacterizationParams,
                                      round_id: int) -> None:
        characterization_results = self.algorithm.execute(nc_params.get_toolset(),
                                                          "NoveltyCharacterization")
        if characterization_results:
            if isinstance(characterization_results, dict):
                self.harness.post_results(characterization_results,
                                          self.test_id, round_id, self.session_id)
            else:
                results = {"characterization": characterization_results}
                self.harness.post_results(results, self.test_id, round_id,
                                          self.session_id)
        else:
            log.warn("No characterization result provided by the algorithm")

    def __call__(
            self,
            dataset: str,
            round_id: int) -> None:
        """
        Core logic for running round in CONDDA.

        Args:
            dataset: Path to a file with the dataset for the round
            round_id: An Identifier for a round

        Returns:
            None
        """
        # Run feature extraction
        fe_params = FeatureExtractionParams(dataset,
                                            self.data_root,
                                            self.redlight_instance,
                                            round_id)
        instance_ids = CONDDARound.get_instance_ids(dataset)
        rfeature_dict, rlogit_dict = self._run_feature_extraction(fe_params,
                                                                  instance_ids)
        # Run World Change Detection
        wc_params = WorldChangeDetectionParams(rfeature_dict, rlogit_dict, round_id)
        self._run_world_change_detection(wc_params, round_id)
        # Run Novelty Classification
        nc_params = NoveltyCharacterizationParams(rfeature_dict, rlogit_dict, round_id)
        self._run_novelty_characterization(nc_params, round_id)
