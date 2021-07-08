"""Test for skipping stages."""

import pytest
from sail_on_client.utils.decorators import skip_stage
from typing import List


class SkipableIncrement:
    """Class for optionally incrementing based on skip stages."""

    def __init__(self, x: int) -> None:
        """
        Constructor for incrementing initial value.

        Args:
            x: Initial value that would be incremented

        Returns:
            None
        """
        self.skip_stages: List[str] = []
        self.x = x

    @skip_stage("increment")
    def increment(self, increment_value: int) -> None:
        """
        Increment value.

        Args:
            increment_value: Value that would be added to x

        Return:
            None
        """
        self.x += increment_value


def test_skip_stage():
    """Test for skipping stage in a class."""
    skipable_increment = SkipableIncrement(0)
    skipable_increment.increment(5)
    assert skipable_increment.x == 5
    skipable_increment.skip_stages = ["increment"]
    skipable_increment.increment(5)
    assert skipable_increment.x == 5
