"""Tests for Utilities."""

from tempfile import TemporaryDirectory
from sail_on_client.harness.local_harness import LocalHarness
from sail_on_client.utils.utils import update_harness_parameters


def test_update_harness_paramerters(get_local_harness_params):
    """
    Test to update harness parameters.

    Args:
        get_local_harness_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    with TemporaryDirectory() as updated_data_dir:
        data_dir, gt_dir, gt_config = get_local_harness_params
        local_interface = LocalHarness(data_dir, gt_dir, gt_config)
        assert local_interface.data_dir != updated_data_dir
        local_interface = update_harness_parameters(
            local_interface, {"data_dir": updated_data_dir}
        )
        assert local_interface.data_dir == updated_data_dir
