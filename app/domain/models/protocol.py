"""Доменные модели для команды protocol."""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass(frozen=True)
class ProtocolConfig:
    """Настройки формирования протокола.

    Attributes:
        provider: Идентификатор провайдера LLM (например, "deepseek").
        model: Название модели у выбранного провайдера.
        api_key: Секрет для доступа к API.
        instructions_path: Путь к файлу с инструкциями (CLI-слой читает содержимое).
        temperature: Температура выборки (если поддерживает провайдер).
        extra_params: Дополнительные настройки, специфичные для провайдера.
    """

    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: str = ""
    instructions_path: Optional[str] = None
    temperature: float = 0.7
    extra_params: Dict[str, object] = None

    def __post_init__(self):
        if self.extra_params is None:
            object.__setattr__(self, "extra_params", {})


@dataclass(frozen=True)
class ProtocolRequest:
    """Запрос на генерацию протокола."""

    instructions: str
    transcript: str
    config: ProtocolConfig

    def render_prompt(self) -> str:
        """Формирует промпт в формате, принятом текущей реализацией."""
        return (
            f"{self.instructions}\n\n"
            "**Расшифровка:**\n\n"
            f"{self.transcript}"
        )


@dataclass(frozen=True)
class ProtocolResponse:
    """Результат генерации протокола."""

    content: str
    provider_raw: Optional[dict] = None

