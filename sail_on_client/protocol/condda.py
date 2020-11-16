"""CONDDA protocol."""

from tinker.baseprotocol import BaseProtocol
from sail_on_client.protocol.condda_config import ConddaConfig
from sail_on_client.errors import RoundError
from sail_on_client.utils import safe_remove, safe_remove_results
from sail_on_client.protocol.parinterface import ParInterface
from itertools import count
import os
import json
import sys
import logging
import pickle as pkl
import ubelt as ub  # type: ignore

from typing import Dict, Any


class Condda(BaseProtocol):
    """CONDDA protocol."""

    def __init__(
        self,
        discovered_plugins: Dict[str, Any],
        algorithmsdirectory: str,
        harness: ParInterface,
        config_file: str,
    ) -> None:
        """Initialize."""

        BaseProtocol.__init__(
            self, discovered_plugins, algorithmsdirectory, harness, config_file
        )
        # The duplication is mainly to prevent mypy attribute error associated with the harness
        self.harness = harness
        if not os.path.exists(config_file):
            logging.error(f"{config_file} does not exist")
            sys.exit(1)

        with open(config_file, "r") as f:
            overriden_config = json.load(f)
        self.config = ConddaConfig(overriden_config)

    def run_protocol(self) -> None:
        """Run protocol."""

        # provide all of the configuration information in the toolset
        self.toolset.update(self.config)
        novelty_algorithm = self.get_algorithm(
            self.config["novelty_detector_class"], self.toolset
        )
        # TODO: fix the version below
        novelty_detector_version = "1.0.0"
        novelty_detector_cv = (
            f"{self.config['novelty_detector_class']}" f"{novelty_detector_version}"
        )
        self.toolset["session_id"] = self.harness.session_request(
            self.config["test_ids"],
            "CONDDA",
            self.config["domain"],
            novelty_detector_cv,
        )
        session_id = self.toolset["session_id"]
        logging.info(f"New session: {self.toolset['session_id']}")
        for test_id in self.config["test_ids"]:
            self.metadata = self.harness.get_test_metadata(session_id, test_id)
            self.toolset["test_id"] = test_id
            self.toolset["test_type"] = ""
            self.toolset["metadata"] = self.metadata
            if "red_light" in self.metadata:
                self.toolset["red_light_image"] = self.toolset["metadata"]["red_light"]
            else:
                self.toolset["red_light_image"] = ""
            novelty_algorithm.execute(self.toolset, "Initialize")
            self.toolset["image_features"] = {}
            self.toolset["dataset_root"] = self.config["dataset_root"]
            self.toolset["dataset_ids"] = []
            logging.info(f"Start test: {self.toolset['test_id']}")

            if self.config["save_features"] and not self.config["use_saved_features"]:
                test_features: Dict[str, Dict] = {"features_dict": {}, "logit_dict": {}}

            if self.config["use_saved_features"]:
                feature_dir = self.config["feature_save_dir"]
                test_features = pkl.load(
                    open(os.path.join(feature_dir, f"{test_id}_features.pkl"), "rb")
                )
                self.toolset.update(test_features)

            for round_id in count(0):
                self.toolset["round_id"] = round_id
                logging.info(f"Start round: {self.toolset['round_id']}")
                # see if there is another round available
                try:
                    self.toolset["dataset"] = self.harness.dataset_request(
                        test_id, round_id, session_id
                    )
                except RoundError:
                    # no more rounds available, this test is done.
                    break
                if not self.config["use_saved_features"]:
                    (
                        self.toolset["features_dict"],
                        self.toolset["logit_dict"],
                    ) = novelty_algorithm.execute(self.toolset, "FeatureExtraction")

                    if self.config["save_features"]:
                        test_features["features_dict"].update(
                            self.toolset["features_dict"]
                        )
                        test_features["logit_dict"].update(self.toolset["logit_dict"])
                        if self.config["feature_extraction_only"]:
                            continue

                results: Dict[str, Any] = {}
                results["detection"] = novelty_algorithm.execute(
                    self.toolset, "WorldDetection"
                )
                results["characterization"] = novelty_algorithm.execute(
                    self.toolset, "NoveltyCharacterization"
                )
                self.harness.post_results(results, test_id, round_id, session_id)
                logging.info(f"Round complete: {self.toolset['round_id']}")
                novelty_algorithm.execute(self.toolset, "NoveltyAdaption")
                # cleanup the round files
                safe_remove(self.toolset["dataset"])
                safe_remove_results(results)
            logging.info(f"Test complete: {self.toolset['test_id']}")

            if self.config["save_features"] and not self.config["use_saved_features"]:
                feature_dir = self.config["feature_save_dir"]
                ub.ensuredir(feature_dir)
                feature_path = os.path.join(feature_dir, f"{test_id}_features.pkl")
                logging.info(f"Saving features in {feature_path}")
                with open(feature_path, "wb") as f:
                    pkl.dump(test_features, f)
                if self.config["feature_extraction_only"]:
                    continue
        logging.info(f"Session ended: {self.toolset['session_id']}")
        self.harness.terminate_session(session_id)
