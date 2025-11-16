"""Сервис для генерации протоколов на основе стенограммы."""

from app.application.ports import ILLMProtocolClient
from app.domain.models.protocol import ProtocolConfig, ProtocolRequest, ProtocolResponse


class ProtocolService:
    """Оркеструет генерацию протокола через LLM-провайдера."""

    def __init__(self, client: ILLMProtocolClient):
        """
        Args:
            client: Реализация порта для выбранного провайдера LLM.
        """
        self._client = client

    def build_request(
        self,
        instructions: str,
        transcript: str,
        config: ProtocolConfig,
    ) -> ProtocolRequest:
        """Формирует ProtocolRequest для дальнейшей передачи адаптеру.

        Выделено в отдельный метод, чтобы обеспечить единообразный шаблон промпта
        и упростить тестирование.
        """
        return ProtocolRequest(
            instructions=instructions,
            transcript=transcript,
            config=config,
        )

    def generate_protocol(
        self,
        instructions: str,
        transcript: str,
        config: ProtocolConfig,
    ) -> ProtocolResponse:
        """Генерирует протокол, обращаясь к провайдеру LLM.

        Args:
            instructions: Текст инструкций для модели.
            transcript: Текст расшифровки встречи.
            config: Конфигурация выбранного провайдера/модели.

        Returns:
            ProtocolResponse: результат генерации протокола.
        """
        request = self.build_request(instructions, transcript, config)
        return self._client.generate_protocol(request)




