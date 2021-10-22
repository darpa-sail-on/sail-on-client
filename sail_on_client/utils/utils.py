"""Utility function for sail-on-client."""

import os
import logging

from typing import Dict, List
from sail_on_client.harness.test_and_evaluation_harness import TestAndEvaluationHarnessType

log = logging.getLogger(__name__)


def safe_remove(file_path: str) -> None:
    """
    Remove a file after checking that it exists.

    Args:
        file_path (str): File path that should be removed

    Return:
        None
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def safe_remove_results(results: dict) -> None:
    """
    Remove results present in a dict.

    Args:
        file_path (str): File path that should be removed

    Return:
        None
    """
    for result_files in results.values():
        safe_remove(result_files)


def update_harness_parameters(ip_harness: TestAndEvaluationHarnessType,
                              new_parameters: Dict) -> TestAndEvaluationHarnessType:
    """
    Update parameters in a harness.

    Args:
        ip_harness (harness): Input harness with old parameters
        new_parameters (dict): Dictionary containing new parameters

    Return:
        Updated harness object
    """
    for param_name, param_value in new_parameters.items():
        if not hasattr(ip_harness, param_name):
            log.warn(
                f"{param_name} is not an attribute in the harness, adding the attribute"
            )
        setattr(ip_harness, param_name, param_value)
    if hasattr(ip_harness, "update_provider"):
        ip_harness.update_provider()
    return ip_harness


def merge_dictionaries(base_dict: Dict, other_dict: Dict, exclude_keys: List) -> Dict:
    """
    Shallow merge for two dictionaries with certain keys would be skipped.

    Args:
        base_dict: Dictionary with defaults
        other_dict: Dictionary with arguments that are added or overriden
        exclude_keys: Keys in other dict that shouldn't be used in the merge

    Returns:
        Dictionary with parameters obtained by merging two dictionaries
    """
    merged_dict = base_dict.copy()
    included_keys = set(other_dict.keys()).difference(set(exclude_keys))
    for key in included_keys:
        merged_dict[key] = other_dict[key]
    return merged_dict
