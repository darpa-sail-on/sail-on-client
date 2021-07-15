"""Sail On client evaluate package."""

from sail_on_client.evaluate.image_classification import ImageClassificationMetrics
from sail_on_client.evaluate.document_transcription import DocumentTranscriptionMetrics
from sail_on_client.evaluate.activity_recognition import ActivityRecognitionMetrics
from typing import Dict, Union


metric_type = Union[
    ActivityRecognitionMetrics, DocumentTranscriptionMetrics, ImageClassificationMetrics
]


def create_metric_instance(protocol: str, domain: str, gt_config: Dict) -> metric_type:
    """
    Create an instance of metric object.

    Args:
        protocol: Name of the protocol
        domain: Name of the domain
        gt_config: Additional arguments used by the metric object
    """
    if domain == "image_classification":
        return ImageClassificationMetrics(protocol, **gt_config)
    elif domain == "activity_recognition":
        return ActivityRecognitionMetrics(protocol, **gt_config)
    elif domain == "transcripts":
        return DocumentTranscriptionMetrics(protocol, **gt_config)
    else:
        raise ValueError(f'Domain: "{domain}" is not a real domain.')
