"""Common dataclasses for visual protocols."""

from dataclasses import dataclass
import logging
from typing import Dict


log = logging.getLogger(__name__)


@dataclass
class FeatureExtractionParams:
    """Class for storing feature extraction parameters associated with an algorithm."""

    dataset: str
    data_root: str
    round_id: int

    def get_toolset(self) -> Dict:
        """
        Convert the data present in the class into a dictionary.

        Returns
            A dictionary with data associated with the class
        """
        return {
            "dataset": self.dataset,
            "dataset_root": self.data_root,
            "round_id": self.round_id,
        }


@dataclass
class WorldChangeDetectionParams:
    """Class for storing parameters associated world change in an algorithm."""

    features_dict: Dict
    logit_dict: Dict
    round_id: int
    redlight_image: str

    def get_toolset(self) -> Dict:
        """
        Convert the data present in the class into a dictionary.

        Returns
            A dictionary with data associated with the class
        """
        return {
            "features_dict": self.features_dict,
            "logit_dict": self.logit_dict,
            "round_id": self.round_id,
            "redlight_image": self.redlight_image,
        }
