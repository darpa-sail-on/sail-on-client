from framework.baseprotocol import BaseProtocol
from ond_config import OndConfig
from importlib_metadata import version
from itertools import count
import os
import json
import sys
from typing import List


def get_image_ids_from_dataset(dataset_path: str) -> List[str]:
    """
    Helper function read a file containing images ids associated with a test

    :param dataset_path: Path to a csv containing the image ids
    :return A list of image ids
    """
    with open(dataset_path, "r") as dataset:
        image_ids = dataset.readlines()
        image_ids = [image_id.strip() for image_id in image_ids]
    return image_ids


def l1_scaling(values: List[float]) -> List[float]:
    """
    Helper function to rescale class probabilities to sum to 1

    :param values: A list of floating point values
    :return A list of rescaled values that sum to 1
    """
    scaling_factor = sum(values)
    rescaled_values = [value/scaling_factor for value in values]
    return rescaled_values


class OndPostProcess(BaseProtocol):
    def __init__(self, discovered_plugins, algorithmsdirectory, harness,
                 config_file):
        BaseProtocol.__init__(self, discovered_plugins, algorithmsdirectory,
                              harness, config_file)
        if not os.path.exists(config_file):
            print(f"{config_file} does not exist", file=sys.stderr)
            sys.exit(1)

        with open(config_file, 'r') as f:
            overriden_config = json.load(f)
        self.config = OndConfig(overriden_config)

    def run_protocol(self):
        # provide all of the configuration information in the toolset
        self.toolset.update(self.config)
        novelty_algorithm = self.get_algorithm(self.config['novelty_detector_class'],
                                               self.toolset)
        #TODO: fix the version below
        #novelty_detector_version = version(novelty_algorithm.__module__)
        novelty_detector_version ='1.0.0'
        nd_class_wt_version = (f"{self.config['novelty_detector_class']}."
                               f"{novelty_detector_version}")
        session_id = \
                self.test_harness.session_request(self.config['test_ids'],
                                                  "OND",
                                                  nd_class_wt_version,
                                                  [])
        session_id_wt_hint = \
                self.test_harness.session_request(self.config['test_ids'],
                                                  "OND",
                                                  nd_class_wt_version,
                                                  ["red_light"])

        self.toolset['session_id']  = session_id
        self.toolset['session_id_wt_hint'] = session_id_wt_hint

        for test_id in self.config['test_ids']:
            self.metadata = self.test_harness.get_test_metadata(test_id,
                                                                session_id)
            metadata_with_rl = \
                    self.test_harness.get_test_metadata(test_id,
                                                        session_id_wt_hint)
            if "red_light" in metadata_with_rl:
                red_light_image = metadata_with_rl["red_light"]
            else:
                print(f"Unable to process {test_id} since red light hint is "
                      f"missing from metadata", file=sys.stderr)
                red_light_image = None

            self.toolset['test_id'] = test_id
            self.toolset['test_type'] = ""
            novelty_algorithm.execute(self.toolset, "Initialize")
            self.toolset['image_features'] = {}
            self.toolset['dataset_root'] = self.config['dataset_root']
            self.toolset['dataset_ids'] = list()
            aggregated_image_ids = []
            for round_id in count(0):
                self.toolset['round_id'] = round_id
                # see if there is another round available
                try:
                    self.toolset['dataset'] = \
                            self.test_harness.dataset_request(test_id,
                                                              round_id,
                                                              session_id)
                    rl_dataset = \
                            self.test_harness.dataset_request(test_id,
                                                              round_id,
                                                              session_id_wt_hint)
                except:
                    # no more rounds available, this test is done.
                    break

                self.toolset['features_dict'], self.toolset['logit_dict'] = \
                        novelty_algorithm.execute(self.toolset,
                                                  "FeatureExtraction")
                nd_results = novelty_algorithm.execute(self.toolset,
                                                       "WorldDetection")
                ncl_results = novelty_algorithm.execute(self.toolset,
                                                        "NoveltyClassification")
                results = {"detection": nd_results,
                           "classification": ncl_results}

                self.test_harness.post_results(results, test_id,
                                               round_id,
                                               session_id)
                image_ids = get_image_ids_from_dataset(self.toolset["dataset"])
                aggregated_image_ids.extend(image_ids)
                if red_light_image is not None:
                    if red_light_image in aggregated_image_ids:
                        red_light = True
                    else:
                        red_light = False
                    self.post_process_redlight(nd_results,
                                               ncl_results,
                                               red_light)
                    rl_results = {"detection": nd_results,
                                  "classification": ncl_results}
                    self.test_harness.post_results(rl_results, test_id,
                                                   round_id,
                                                   session_id_wt_hint)
                # Remove detection and classification files after posting
                # results
                if nd_results is not None and \
                        os.path.exists(nd_results):
                    os.remove(nd_results)
                if ncl_results is not None and \
                        os.path.exists(ncl_results):
                    os.remove(ncl_results)

            results = dict()
            self.toolset['dataset_ids'] = aggregated_image_ids
            results['characterization'] = \
                    novelty_algorithm.execute(self.toolset,
                                              "NoveltyCharacterization")
            if results['characterization'] is not None and \
               os.path.exists(results['characterization']):
                self.test_harness.post_results(results, test_id, 0, session_id)
                self.test_harness.post_results(results, test_id, 0,
                                               session_id_wt_hint)
                os.remove(results['characterization'])
        self.test_harness.terminate_session(session_id)
        self.test_harness.terminate_session(session_id_wt_hint)

    def post_process_redlight(self, novelty_detection_path: str,
                              novelty_classification_path: str,
                              is_redlight_on: bool) ->None:
        """
        Open the detection and classification files and update them according
        to the redlight indicator
        :param novelty_detection_path: Path to file containing detection results
        :param novelty_classification_path: Path to file containing classification results
        :param is_redlight_on: Boolean indicator for red light

        :return None
        """
        # process the detections file first
        if is_redlight_on:
            world_changed = 1
        else:
            world_changed = 0

        with open(novelty_detection_path, "r" ) as f:
            lines = f.readlines()
        with open(novelty_detection_path, "w" ) as f:
            for line in lines:
                linedata = line.strip().split(",")
                f.write(f"{linedata[0]},{world_changed}\n")

        # next process the classification file
        if not is_redlight_on:
            with open(novelty_classification_path, "r" ) as f:
                lines = f.readlines()

            with open(novelty_classification_path, "w" ) as f:
                for line in lines:
                    linedata = line.strip().split(",")
                    rescaled_values = l1_scaling(list(map(float, linedata[2:])))
                    rescaled_str = ",".join(map(str, rescaled_values))
                    f.write(f"{linedata[0]},{0},{rescaled_str}\n")
