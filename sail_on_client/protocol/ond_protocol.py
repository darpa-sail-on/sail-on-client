"""OND protocol."""

from tinker.baseprotocol import BaseProtocol

from sail_on_client.protocol.ond_config import OndConfig
from sail_on_client.errors import RoundError
from sail_on_client.utils import safe_remove
from sail_on_client.protocol.parinterface import ParInterface
from itertools import count
import os
import json
import sys
import logging
import pickle as pkl
import ubelt as ub

from typing import Dict, Any


class SailOn(BaseProtocol):
    """The base protocol for Sail On."""

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
        self.config = OndConfig(overriden_config)

    def run_protocol(self) -> None:
        """Run the protocol."""

        # provide all of the configuration information in the toolset
        self.toolset.update(self.config)
        novelty_algorithm = self.get_algorithm(
            self.config["novelty_detector_class"], self.toolset
        )

        # TODO: fix the version below
        novelty_detector_version = "1.0.0"

        self.toolset["session_id"] = self.harness.session_request(
            self.config["test_ids"],
            "OND",
            "%s.%s" % (self.config["novelty_detector_class"], novelty_detector_version),
        )
        session_id = self.toolset["session_id"]

        logging.info(f"New session: {self.toolset['session_id']}")

        for test in self.config["test_ids"]:
            self.metadata = self.harness.get_test_metadata(test)
            self.toolset["test_id"] = test
            self.toolset["test_type"] = ""
            self.toolset["metadata"] = self.harness.get_test_metadata(test)
            if "red_light" in self.toolset["metadata"]:
                self.toolset["redlight_image"] = os.path.join(
                    self.toolset["dataset_root"], self.toolset["metadata"]["red_light"]
                )
            else:
                self.toolset["redlight_image"] = ""
            novelty_algorithm.execute(self.toolset, "Initialize")
            self.toolset["image_features"] = {}
            self.toolset["dataset_root"] = self.config["dataset_root"]
            self.toolset["dataset_ids"] = []

            logging.info(f"Start test: {self.toolset['test_id']}")
            if self.config["save_features"]:
                test_features = {"features_dict": {}, "logit_dict": {}}

            for round_id in count(0):
                self.toolset["round_id"] = round_id

                logging.info(f"Start round: {self.toolset['round_id']}")
                # see if there is another round available
                try:
                    self.toolset["dataset"] = self.harness.dataset_request(
                        test, round_id, session_id
                    )
                except RoundError:
                    # no more rounds available, this test is done.
                    break

                (
                    self.toolset["features_dict"],
                    self.toolset["logit_dict"],
                ) = novelty_algorithm.execute(self.toolset, "FeatureExtraction")

                if self.config["save_features"]:
                    test_features["features_dict"].update(self.toolset["features_dict"])
                    test_features["logit_dict"].update(self.toolset["logit_dict"])
                    if self.config["feature_extraction_only"]:
                        continue

                results: Dict[str, Any] = {}

                results["detection"] = novelty_algorithm.execute(
                    self.toolset, "WorldDetection"
                )

                results["classification"] = novelty_algorithm.execute(
                    self.toolset, "NoveltyClassification"
                )

                self.harness.post_results(results, test, round_id, session_id)
                with open(self.toolset["dataset"], "r") as dataset:
                    self.toolset["dataset_ids"].extend(dataset.readlines())
                logging.info(f"Round complete: {self.toolset['round_id']}")

                # cleanup the round files
                safe_remove(self.toolset["dataset"])
                safe_remove(results["detection"])
                safe_remove(results["classification"])

            if self.config["save_features"]:
                feature_dir = self.config["feature_save_dir"]
                ub.ensuredir(feature_dir)
                feature_path = os.path.join(feature_dir, f"{test}_features.pkl")
                logging.info(f"Saving features in {feature_path}")
                with open(feature_path, "wb") as f:
                    pkl.dump(test_features, f)
                if self.config["feature_extraction_only"]:
                    continue

            results = {}

            self.toolset["dataset_ids"] = [
                image_id.strip() for image_id in self.toolset["dataset_ids"]
            ]

            results["characterization"] = novelty_algorithm.execute(
                self.toolset, "NoveltyCharacterization"
            )
            if results["characterization"] is not None and os.path.exists(
                results["characterization"]
            ):
                self.harness.post_results(results, test, 0, session_id)
            logging.info(f"Test complete: {self.toolset['test_id']}")

            # cleanup the characterization file
            safe_remove(results["characterization"])

        logging.info(f"Session ended: {self.toolset['session_id']}")
        self.harness.terminate_session(session_id)
