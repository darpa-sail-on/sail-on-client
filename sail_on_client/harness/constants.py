"""Constants used by the file provider."""


class ProtocolConstants:
    """Class with constants for the protocol."""

    # Feedback types
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    CHARACTERIZATION = "characterization"
    TRANSCRIPTION = "transcription"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    PSUEDO_CLASSIFICATION = "psuedo_labels_classification"
    SCORE = "score"
    LABELS = "labels"

    # Detection requirement
    REQUIRED = "REQUIRED"
    NOTIFY_AND_CONTINUE = "NOTIFY_AND_CONTINUE"
    IGNORE = "IGNORE"
    SKIP = "SKIP"

    # Domain specific encoding
    NLT_ENCODING = "utf-16"
    VAR_ENCODING = "utf-8"
    WTR_ENCODING = "utf-8"
