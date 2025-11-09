"""Порт для провайдеров LLM, генерирующих протоколы."""

from abc import ABC, abstractmethod
from app.domain.models.protocol import ProtocolRequest, ProtocolResponse


class ILLMProtocolClient(ABC):
    """Абстракция клиента LLM, генерирующего протоколы на основе стенограммы."""

    @abstractmethod
    def generate_protocol(self, request: ProtocolRequest) -> ProtocolResponse:
        """Генерирует протокол по переданному запросу.

        Args:
            request: Подготовленный запрос (инструкции, стенограмма, конфиг).

        Returns:
            ProtocolResponse: сгенерированный текст и, опционально, сырой ответ провайдера.

        Raises:
            ProtocolClientError: если провайдер вернул ошибку или ответ некорректен.
        """
        raise NotImplementedError


