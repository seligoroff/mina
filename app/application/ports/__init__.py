"""Порты (интерфейсы) для взаимодействия с внешними зависимостями."""

from app.application.ports.transcription_port import ITranscriptionEngine
from app.application.ports.output_port import ITranscriptSegmentWriter
from app.application.ports.api_port import ILLMProtocolClient
from app.application.ports.word_analysis_port import ITextSource, IStopwordsProvider
from app.application.ports.chat_export_port import IChatExporter, IMessageStore

__all__ = [
    "ITranscriptionEngine",
    "ITranscriptSegmentWriter",
    "ILLMProtocolClient",
    "ITextSource",
    "IStopwordsProvider",
    "IChatExporter",
    "IMessageStore",
]


