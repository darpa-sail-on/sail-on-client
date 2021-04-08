"""Abstract class for feedback for sail-on."""

import pandas as pd
from sail_on_client.protocol.parinterface import ParInterface
from sail_on_client.protocol.localinterface import LocalInterface

from typing import Union, Dict


class Feedback:
    """Base class for Feedback."""

    def __init__(
        self,
        first_budget: int,
        income_per_batch: int,
        maximum_budget: int,
        interface: Union[LocalInterface, ParInterface],
        session_id: str,
        test_id: str,
        feedback_type: str,
    ) -> None:
        """
        Initialize.

        Args:
            first_budget: Initial budget
            income_per_batch: Additional labels added after every batch
            maximum_budget: Max labels that can be requested
            interface: An instance of evaluation interface
            session_id: Session identifier
            test_id: Test identifier
            feedback_type: Type of feedback that can be requested

        Returns:
            None
        """
        self.budget = first_budget
        self.income_per_batch = income_per_batch
        self.maximum_budget = maximum_budget
        self.current_round = -1
        self.interface = interface
        self.session_id = session_id
        self.test_id = test_id
        self.feedback_type = feedback_type


    def get_score_feedback(
        self, round_id: int, images_id_list: list, image_names: list
    ) -> Union[Dict, None]:
        """
        Get accuracy value for the round. Note: this is not budgeted.

        Args:
            round_id: Round identifier
            image_id_list: List if indices for images
            image_names: List of image names for the round

        Return:
            A dictionary with the accuracy value or None if
            feedback is requested for an older round
        """
        if round_id > self.current_round:
            self.deposit_income()
            self.current_round = round_id
            image_ids = [image_names[int(idx)] for idx in images_id_list]
            feedback_score = self.interface.get_feedback_request(
                image_ids,
                self.feedback_type,
                self.test_id,
                round_id,
                self.session_id,
            )
            return feedback_score
        else:
            return None

    def get_feedback(
        self, round_id: int, images_id_list: list, image_names: list
    ) -> Union[pd.DataFrame, Dict, None]:
        """
        Get feedback for the round.

        Args:
            round_id: Round identifier
            image_id_list: List if indices for images
            image_names: List of image names for the round

        Return:
            Either a dataframe or dictionary with score if the request is valid
            for the current round.
        """
        if self.feedback_type == "classification":
            feedback_fn = self.get_labeled_feedback
        elif self.feedback_type == "score":
            feedback_fn = self.get_score_feedback
        else:
            raise ValueError("Unsupported feedback type {self.feedback_type} specified")
        return feedback_fn(round_id, images_id_list, image_names)

    def deposit_income(self) -> None:
        """Get income for a round."""
        self.budget = min(self.maximum_budget, (self.budget + self.income_per_batch))

    def get_budget(self) -> int:
        """Get current budget."""
        return self.budget
