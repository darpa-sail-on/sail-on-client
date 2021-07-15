"""Sail On client feedback package."""

from sail_on_client.feedback.image_classification_feedback import (
    ImageClassificationFeedback,
)
from sail_on_client.feedback.document_transcription_feedback import (
    DocumentTranscriptionFeedback,
)
from sail_on_client.feedback.activity_recognition_feedback import (
    ActivityRecognitionFeedback,
)
from typing import Dict, Union


feedback_type = Union[
    ActivityRecognitionFeedback,
    DocumentTranscriptionFeedback,
    ImageClassificationFeedback,
]


def create_feedback_instance(domain: str, feedback_config: Dict) -> feedback_type:
    """
    Create an instance of feedback object.

    Args:
        domain: Name of the domain
        feedback_config: arguments used to instantiate the metric object
    """
    if domain == "image_classification":
        return ImageClassificationFeedback(**feedback_config)
    elif domain == "activity_recognition":
        return ActivityRecognitionFeedback(**feedback_config)
    elif domain == "transcripts":
        return DocumentTranscriptionFeedback(**feedback_config)
    else:
        raise ValueError(f'Domain: "{domain}" is not a real domain.')
