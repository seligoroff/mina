from __future__ import annotations

from typing import Iterable, Protocol

from app.domain.exceptions import ChatExportError
from app.domain.models import ChatExportConfig, ChatMessage


class IChatExporter(Protocol):
    """Порт экспорта сообщений чата/канала.

    Контракты:
    - Сообщения возвращаются в монотонно возрастающем порядке по дате/ID.
    - Сетевые/временные сбои преобразуются в ChatExportError.
    """

    def export(self, config: ChatExportConfig) -> Iterable[ChatMessage]:
        """Возвращает итератор нормализованных сообщений согласно конфигурации."""
        ...


class IMessageStore(Protocol):
    """Опциональный порт для записи сообщений в хранилище (файлы, S3, БД)."""

    def write(self, messages: Iterable[ChatMessage], destination: str, format: str = "jsonl") -> None:
        """Сериализует и записывает поток сообщений в указанный destination."""
        ...


