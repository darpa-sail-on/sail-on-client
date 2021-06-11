"""CONDDA protocol."""

from sailon_tinker_launcher.deprecated_tinker.baseprotocol import BaseProtocol
from sail_on_client.protocol.condda_config import ConddaConfig
from sail_on_client.errors import RoundError
from sail_on_client.utils import (
    safe_remove,
    safe_remove_results,
    update_harness_parameters,
)
from sail_on_client.protocol.parinterface import ParInterface
from sail_on_client.protocol.localinterface import LocalInterface
from itertools import count
import os
import json
import sys
import logging
import pickle as pkl
import ubelt as ub  # type: ignore

from typing import Dict, Any, Union

log = logging.getLogger(__name__)


class Condda(BaseProtocol):
    """CONDDA protocol."""

    def __init__(
        self,
        discovered_plugins: Dict[str, Any],
        algorithmsdirectory: str,
        harness: Union[ParInterface, LocalInterface],
        config_file: str,
    ) -> None:
        """Initialize."""

        BaseProtocol.__init__(
            self, discovered_plugins, algorithmsdirectory, harness, config_file
        )
        # The duplication is mainly to prevent mypy attribute error associated with the harness
        self.harness = harness
        if not os.path.exists(config_file):
            log.error(f"{config_file} does not exist")
            sys.exit(1)

        with open(config_file, "r") as f:
            overriden_config = json.load(f)
        self.config = ConddaConfig(overriden_config)
        self.harness = update_harness_parameters(harness, self.config["harness_config"])

    def run_protocol(self) -> None:
        """Run protocol."""

        # provide all of the configuration information in the toolset
        self.toolset.update(self.config)
        algorithm_names = self.config["detectors"]["detector_configs"].keys()
        algorithms = {}
        sessions = {}
        # Create sessions an instances of all the algorithms
        for algorithm_name in algorithm_names:
            algorithm = self.get_algorithm(
                algorithm_name,
                self.config["detectors"]["detector_configs"][algorithm_name],
            )
            algorithms[algorithm_name] = algorithm
            algorithm_toolset = self.config["detectors"]["detector_configs"][
                algorithm_name
            ]
            # TODO: fix the version below
            novelty_detector_version = "1.0.0"
            novelty_detector_class = algorithm_name
            if "detection_threshold" in algorithm_toolset:
                detector_threshold = float(algorithm_toolset["detection_threshold"])
            else:
                detector_threshold = 0.5
            test_ids = self.config["test_ids"]
            if self.config["resume_session"]:
                if algorithm_name in self.config["resumed_session_ids"]:
                    session_id = self.config["resumed_session_ids"][algorithm_name]
                else:
                    raise ValueError(f"Failed to resume session for {algorithm_name}")
                finished_test = self.harness.resume_session(session_id)
                remaining_tests = []
                for test_id in test_ids:
                    if test_id not in finished_test:
                        remaining_tests.append(test_id)
                test_ids = remaining_tests
            else:
                session_id = self.harness.session_request(
                    test_ids,
                    "CONDDA",
                    self.config["domain"],
                    f"{novelty_detector_version}.{novelty_detector_class}",
                    self.config["hints"],
                    detector_threshold,
                )
            sessions[algorithm_name] = session_id
        for algorithm_name, session_id in sessions.items():
            log.info(f"New session: {session_id} for algorithm: {algorithm_name}")
            for test_id in test_ids:
                self.metadata = self.harness.get_test_metadata(session_id, test_id)
                self.toolset["test_id"] = test_id
                self.toolset["test_type"] = ""
                self.toolset["metadata"] = self.metadata
                if "red_light" in self.metadata:
                    self.toolset["redlight_image"] = self.toolset["metadata"][
                        "red_light"
                    ]
                else:
                    self.toolset["redlight_image"] = ""
                algorithm_toolset = {}
                for config_name, config_value in self.config["detectors"].items():
                    if (
                        config_name == "has_baseline"
                        or config_name == "has_reaction_baseline"
                        or config_name == "baseline_class"
                    ):
                        continue
                    elif config_name == "detector_configs":
                        algorithm_toolset.update(config_value[algorithm_name])
                    else:
                        algorithm_toolset[config_name] = config_value

                self.config["detectors"]["detector_configs"][algorithm_name]
                algorithm_toolset["session_id"] = session_id
                algorithm_toolset["test_id"] = test_id
                algorithm_toolset["test_type"] = ""
                algorithms[algorithm_name].execute(algorithm_toolset, "Initialize")

                self.toolset["image_features"] = {}
                self.toolset["dataset_root"] = self.config["dataset_root"]
                self.toolset["dataset_ids"] = []

                log.info(f"Start test: {self.toolset['test_id']}")

                if (
                    self.config["save_features"]
                    and not self.config["use_saved_features"]
                ):
                    test_features: Dict[str, Dict] = {
                        "features_dict": {},
                        "logit_dict": {},
                    }

                if self.config["use_saved_features"]:
                    feature_dir = self.config["feature_save_dir"]
                    if os.path.isdir(feature_dir):
                        if self.config["use_consolidated_features"]:
                            test_features = pkl.load(
                                open(
                                    os.path.join(
                                        feature_dir, f"{algorithm_name}_features.pkl",
                                    ),
                                    "rb",
                                )
                            )
                        else:
                            test_features = pkl.load(
                                open(
                                    os.path.join(
                                        feature_dir,
                                        f"{test_id}_{algorithm_name}_features.pkl",
                                    ),
                                    "rb",
                                )
                            )
                    else:
                        test_features = pkl.load(open(feature_dir, "rb"))

                    features_dict = test_features["features_dict"]
                    logit_dict = test_features["logit_dict"]

                for round_id in count(0):
                    self.toolset["round_id"] = round_id
                    log.info(f"Start round: {self.toolset['round_id']}")
                    # see if there is another round available
                    try:
                        self.toolset["dataset"] = self.harness.dataset_request(
                            test_id, round_id, session_id
                        )
                    except RoundError:
                        # no more rounds available, this test is done.
                        break

                    with open(self.toolset["dataset"], "r") as dataset:
                        dataset_ids = dataset.readlines()
                        image_ids = [image_id.strip() for image_id in dataset_ids]

                    if self.config["use_saved_features"]:
                        self.toolset["features_dict"] = {}
                        self.toolset["logit_dict"] = {}
                        for image_id in image_ids:
                            self.toolset["features_dict"][image_id] = features_dict[
                                image_id
                            ]
                            self.toolset["logit_dict"][image_id] = logit_dict[image_id]
                    else:
                        (
                            self.toolset["features_dict"],
                            self.toolset["logit_dict"],
                        ) = algorithms[algorithm_name].execute(
                            self.toolset, "FeatureExtraction"
                        )

                        if self.config["save_features"]:
                            test_features["features_dict"].update(
                                self.toolset["features_dict"]
                            )
                            test_features["logit_dict"].update(
                                self.toolset["logit_dict"]
                            )
                            if self.config["feature_extraction_only"]:
                                continue

                    results: Dict[str, Any] = {}
                    results["detection"] = algorithms[algorithm_name].execute(
                        self.toolset, "WorldDetection"
                    )
                    results["characterization"] = algorithms[algorithm_name].execute(
                        self.toolset, "NoveltyCharacterization"
                    )
                    algorithms[algorithm_name].execute(self.toolset, "NoveltyAdaption")
                    self.harness.post_results(results, test_id, round_id, session_id)
                    log.info(f"Round complete: {self.toolset['round_id']}")
                    # cleanup the round files
                    safe_remove(self.toolset["dataset"])
                    safe_remove_results(results)
                log.info(f"Test complete: {self.toolset['test_id']}")
                self.harness.complete_test(session_id, test_id)
                if (
                    self.config["save_features"]
                    and not self.config["use_saved_features"]
                ):
                    feature_dir = self.config["save_dir"]
                    ub.ensuredir(feature_dir)
                    feature_path = os.path.join(
                        feature_dir, f"{test_id}_{algorithm_name}_features.pkl"
                    )
                    log.info(f"Saving features in {feature_path}")
                    with open(feature_path, "wb") as f:
                        pkl.dump(test_features, f)
        for algorithm_name, session_id in sessions.items():
            log.info(f"Session ended for {algorithm_name}: {session_id}")
            self.harness.terminate_session(session_id)
