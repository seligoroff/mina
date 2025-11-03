"""Порты (интерфейсы) для взаимодействия с внешними зависимостями."""

from app.application.ports.transcription_port import ITranscriptionEngine
from app.application.ports.output_port import ITranscriptSegmentWriter

__all__ = ["ITranscriptionEngine", "ITranscriptSegmentWriter"]


