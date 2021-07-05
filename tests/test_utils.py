"""Tests for Utilities."""

from tempfile import TemporaryDirectory
from sail_on_client.protocol.localinterface import LocalInterface
from sail_on_client.utils.utils import update_harness_parameters


def test_update_harness_paramerters(get_interface_params):
    """
    Test to update harness parameters.

    Args:
        get_interface_params (tuple): Tuple to configure local interface

    Return:
        None
    """
    with TemporaryDirectory() as updated_data_dir:
        config_directory, config_name = get_interface_params
        local_interface = LocalInterface(config_name, config_directory)
        assert local_interface.data_dir != updated_data_dir
        local_interface = update_harness_parameters(
            local_interface, {"data_dir": updated_data_dir}
        )
        assert local_interface.data_dir == updated_data_dir
