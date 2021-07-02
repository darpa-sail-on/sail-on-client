"""Utility function for sail-on-client."""

import os
import numpy as np
import logging
import json
import functools
from sailon_tinker_launcher.deprecated_tinker import harness

from typing import Dict, Any, Callable, List

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

def update_harness_parameters(ip_harness: harness, new_parameters: Dict) -> harness:
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


def merge_dictionaries(base_dict, other_dict, exclude_keys) -> Dict:
    """
    Shallow merge for two dictionaries taking into account that certain keys should be skipped.
    """
    merged_dict = base_dict.copy()
    included_keys = set(other_dict.keys()).difference(set(exclude_keys))
    for key in included_keys:
        merged_dict[key] = other_dict[key]
    return merged_dict


def skip_stage(stage_name: str,
               skip_return: Any = None) -> Callable:
    """
    A decorator for skipping stages in the protocol.

    Args:
        stage_name: Name of the stage that is covered by the decorated function
        skip_stages: List of stages that should be skipped
        skip_return: Optional return types when the stage is skipped

    Returns:
        Decorated function call
    """
    def skip_stage_decorator(stage_fn: Callable) -> Callable:
        """
        Wrapper to capturing the stage function

        Args:
            stage_fn: The callable function that would be wrapped

        Returns:
            Wrapped function
        """
        @functools.wraps(stage_fn)
        def skip_stage_fn(self, *args, **kwargs):
            if hasattr(self, "skip_stages"):
                skip_stages = self.skip_stages
            else:
                raise ValueError("The class does not skip_stages")

            if stage_name in skip_stages:
                return skip_return
            else:
                return stage_fn(self, *args, **kwargs)
        return skip_stage_fn
    return skip_stage_decorator


class NumpyEncoder(json.JSONEncoder):
    """An encoder to convert numpy data types to python primitives."""

    def default(self, obj: Any) -> Any:
        """Defaults for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NumpyEncoder, self).default(obj)
