"""Фабрики для создания компонентов приложения."""

from app.factories.transcription_factory import (
    create_transcription_adapter,
    create_transcription_service,
)
from app.factories.protocol_factory import (
    create_protocol_client,
    create_protocol_service,
)
from app.factories.tag_factory import create_word_analysis_service
from app.factories.telegram_factory import create_telethon_chat_exporter

__all__ = [
    "create_transcription_adapter",
    "create_transcription_service",
    "create_protocol_client",
    "create_protocol_service",
    "create_word_analysis_service",
    "create_telethon_chat_exporter",
]




