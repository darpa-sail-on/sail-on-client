"""Visual Protocol."""

import os
import logging
import sys
from typing import Dict, List, Any, Union

from sail_on_client.protocol.parinterface import ParInterface
from sail_on_client.protocol.localinterface import LocalInterface
from sail_on_client.protocol.ond_dataclasses import AlgorithmAttributes as ONDAlgorithmAttributes
from sail_on_client.protocol.condda_dataclasses import AlgorithmAttributes as CONDDAAlgorithmAttributes
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

    def create_algorithm_session(
            self,
            algorithm_attributes: Union[ONDAlgorithmAttributes, CONDDAAlgorithmAttributes],
            domain: str,
            hints: List[str],
            has_a_session,
            protocol_name: str) -> Union[ONDAlgorithmAttributes, CONDDAAlgorithmAttributes]:
        """
        Create/resume session for an algorithm.

        Args:
            algorithm_attributes: An instance of AlgorithmAttributes
            domain: Domain for the algorithm
            hints: List of hints used in the session
            has_a_session: Already has a session and we want to resume it
            protocol_name: Name of the algorithm

        Returns:
            An AlgorithmAttributes object with updated session id or test id
        """
        test_ids = algorithm_attributes.test_ids
        named_version = algorithm_attributes.named_version()
        detection_threshold = algorithm_attributes.detection_threshold

        if has_a_session:
            session_id = algorithm_attributes.session_id
            finished_test = self.harness.resume_session(session_id)
            algorithm_attributes.remove_completed_tests(finished_test)
            log.info(f"Resumed session {session_id} for {algorithm_attributes.name}")
        else:
            session_id = self.harness.session_request(
                test_ids,
                protocol_name,
                domain,
                named_version,
                hints,
                detection_threshold,
            )
            algorithm_attributes.session_id = session_id
            log.info(f"Created session {session_id} for {algorithm_attributes.name}")
        return algorithm_attributes
