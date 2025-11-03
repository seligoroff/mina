"""Фабрики для создания компонентов приложения."""

from app.factories.transcription_factory import (
    create_transcription_adapter,
    create_transcription_service
)

__all__ = [
    "create_transcription_adapter",
    "create_transcription_service"
]



