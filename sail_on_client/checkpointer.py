"""Checkpoint to save and restore attributes."""

import logging
import torch
import os
import pickle as pkl
from torch import Tensor

from typing import Dict, Any

log = logging.getLogger(__name__)


class Checkpointer(object):
    """Checkpoint object to save and restore attributes."""

    def __init__(self, toolset: Dict) -> None:
        """
        Initialize.

        Args:
            toolset: Dictionary with parameter for the mixin

        Returns:
            None
        """
        self.toolset = toolset

    def _save_elementwise_attribute(
        self, detector: Any, attribute: str, attribute_dict: Dict
    ) -> Dict:
        """
        Private method to save attributes element wise.

        Args:
            detector: Instance of novelty detector
            attribute: Name of the detector attribute that needs to be saved
            attribute dict: A dictonary containing attribute value for other tests

        Returns:
            Update attribute dictionary
        """
        attribute_val = getattr(detector, attribute)
        test_id = self.toolset["test_id"]
        if isinstance(attribute_val, dict):
            if test_id in attribute_dict:
                attribute_dict[test_id].update(attribute_val)
            else:
                attribute_dict[test_id] = attribute_val
        elif isinstance(attribute_val, list):
            if test_id in attribute_dict:
                attribute_dict[test_id].extend(attribute_val)
            else:
                attribute_dict[test_id] = attribute_val
        elif isinstance(attribute_val, tuple):
            if test_id in attribute_dict:
                old_attr_val = list(attribute_dict[test_id])
                old_attr_val.extend(list(attribute_val))
                attribute_dict[test_id] = tuple(old_attr_val)
            else:
                attribute_dict[test_id] = attribute_val
        elif isinstance(attribute_val, Tensor):
            if test_id in attribute_dict:
                attribute_dict[test_id] = torch.cat(
                    [attribute_dict[test_id], attribute_val]
                )
            else:
                attribute_dict[test_id] = attribute_val
        else:
            log.info(
                "Treating attribute value as a single element rather than an iterable"
            )
            attribute_dict[test_id] = attribute_val
        return attribute_dict

    def save_attributes(self, step_descriptor: str) -> None:
        """
        Save attribute for a detector.

        Args:
            step_descriptor: String describing steps for protocol

        Returns
            None
        """
        if step_descriptor in self.toolset["saved_attributes"]:
            attributes = self.toolset["saved_attributes"][step_descriptor]
        else:
            attributes = []
        save_elementwise = self.toolset["save_elementwise"]
        attribute_dict = self.toolset["attributes"]
        self.detector: Any
        if len(attributes) > 0:
            for attribute in attributes:
                if hasattr(self.detector, attribute) and save_elementwise:
                    if attribute not in attribute_dict:
                        attribute_dict[attribute] = {}
                    attribute_dict[attribute] = self._save_elementwise_attribute(
                        self.detector, attribute, attribute_dict[attribute]
                    )
                elif hasattr(self.detector, attribute) and not save_elementwise:
                    raise NotImplementedError(
                        "Saving attributes for an entire round is not supported"
                    )
                else:
                    log.warn(f"Detector does not have {attribute} attribute")
        else:
            log.info(f"No attributes found for {step_descriptor}")
        self.toolset["attributes"] = attribute_dict

    def _restore_elementwise_attribute(
        self, detector: Any, attribute_name: str, attribute_val: Dict
    ) -> Any:
        """
        Private method to restore attributes element wise.

        Args:
            detector: Instance of novelty detector
            attribute_name: Name of the detector attribute that needs to be saved
            attribute_val: A dictonary containing value for attributes

        Returns
            Detector with updated value for attributes
        """
        dataset_ids = list(
            map(lambda x: x.strip(), open(self.toolset["dataset"], "r").readlines())
        )
        round_id = self.toolset["round_id"]
        round_len = len(dataset_ids)
        if isinstance(attribute_val, dict):
            round_attribute_val = {}
            for dataset_id in dataset_ids:
                round_attribute_val[dataset_id] = attribute_val[dataset_id]
        elif (
            isinstance(attribute_val, list)
            or isinstance(attribute_val, tuple)
            or isinstance(attribute_val, Tensor)
        ):
            round_attribute_val = attribute_val[
                round_id * round_len : (round_id + 1) * round_len
            ]
        else:
            log.info(
                "Treating attribute value as a single element rather than an iterable."
            )
            round_attribute_val = attribute_val
        setattr(detector, attribute_name, round_attribute_val)
        return detector

    def restore_attributes(self, step_descriptor: str) -> None:
        """
        Restore attribute for a detector.

        Args:
            step_descriptor: String describing steps for protocol

        Returns:
            None
        """
        if (
            self.toolset["use_saved_attributes"]
            and step_descriptor in self.toolset["saved_attributes"]
        ):
            attributes = self.toolset["saved_attributes"][step_descriptor]
            save_elementwise = self.toolset["save_elementwise"]
            save_dir = self.toolset["save_dir"]
            test_id = self.toolset["test_id"]
            if os.path.isdir(save_dir):
                attribute_file = os.path.join(save_dir, f"{test_id}_attribute.pkl")
                attribute_val = pkl.load(open(attribute_file, "rb"))
            else:
                attribute_val = pkl.load(open(save_dir, "rb"))
            for attribute in attributes:
                if save_elementwise:
                    self.detector = self._restore_elementwise_attribute(
                        self.detector, attribute, attribute_val[attribute][test_id]
                    )
                else:
                    raise NotImplementedError(
                        "Restoring attributes for an entire round is not supported."
                    )

        else:
            log.info(f"No attributes found for {step_descriptor}.")
