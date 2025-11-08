"""Порты (интерфейсы) для взаимодействия с внешними зависимостями."""

from app.application.ports.transcription_port import ITranscriptionEngine
from app.application.ports.output_port import ITranscriptSegmentWriter
from app.application.ports.api_port import ILLMProtocolClient
from app.application.ports.word_analysis_port import ITextSource, IStopwordsProvider

__all__ = [
    "ITranscriptionEngine",
    "ITranscriptSegmentWriter",
    "ILLMProtocolClient",
    "ITextSource",
    "IStopwordsProvider",
]


