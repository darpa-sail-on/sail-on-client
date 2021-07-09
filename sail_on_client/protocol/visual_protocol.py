"""Visual Protocol."""

import os
import logging
import sys
from typing import Dict, Any, Union

from sail_on_client.protocol.parinterface import ParInterface
from sail_on_client.protocol.localinterface import LocalInterface
from sailon_tinker_launcher.deprecated_tinker.baseprotocol import BaseProtocol


log = logging.getLogger(__name__)


class VisualProtocol(BaseProtocol):
    """Protocol for visual tasks."""

    def __init__(
        self,
        discovered_plugins: Dict[str, Any],
        algorithmsdirectory: str,
        harness: Union[ParInterface, LocalInterface],
        config_file: str,
    ) -> None:
        """
        Construct visual protocol.

        Args:
            discovered_plugins: Dict of algorithms that can be used by the protocols
            algorithmsdirectory: Directory with the algorithms
            harness: An object for the harness used for T&E
            config_file: Path to a config file used by the protocol

        Returns:
            None
        """
        BaseProtocol.__init__(
            self, discovered_plugins, algorithmsdirectory, harness, config_file
        )
        if not os.path.exists(config_file):
            log.error(f"{config_file} does not exist")
            sys.exit(1)
