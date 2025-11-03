"""Адаптеры для движков транскрипции Whisper."""

from app.adapters.output.whisper.whisper_adapter import WhisperAdapter
from app.adapters.output.whisper.faster_whisper_adapter import FasterWhisperAdapter

__all__ = ["WhisperAdapter", "FasterWhisperAdapter"]



