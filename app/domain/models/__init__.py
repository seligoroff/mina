"""Доменные модели."""

from app.domain.models.transcript import Segment, Transcript
from app.domain.models.protocol import ProtocolConfig, ProtocolRequest, ProtocolResponse
from app.domain.models.word_analysis import WordAnalysisConfig, WordFrequencyResult

__all__ = [
    "Segment",
    "Transcript",
    "ProtocolConfig",
    "ProtocolRequest",
    "ProtocolResponse",
    "WordAnalysisConfig",
    "WordFrequencyResult",
]





