"""Тесты для ProtocolService."""

from unittest.mock import Mock

import pytest

from app.application.services import ProtocolService
from app.domain.models.protocol import ProtocolConfig, ProtocolResponse


@pytest.mark.unit
class TestProtocolService:
    """Набор тестов для ProtocolService."""

    def test_generate_protocol_delegates_to_client(self):
        config = ProtocolConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key="key",
        )
        client = Mock()
        expected_response = ProtocolResponse(content="result")
        client.generate_protocol.return_value = expected_response

        service = ProtocolService(client=client)

        response = service.generate_protocol(
            instructions="Инструкции",
            transcript="Стенограмма",
            config=config,
        )

        client.generate_protocol.assert_called_once()
        request = client.generate_protocol.call_args.args[0]
        assert request.instructions == "Инструкции"
        assert request.transcript == "Стенограмма"
        assert request.render_prompt().startswith("Инструкции")
        assert response is expected_response

    def test_generate_protocol_propagates_client_errors(self):
        config = ProtocolConfig(provider="deepseek", api_key="key")
        client = Mock()
        client.generate_protocol.side_effect = RuntimeError("provider failure")
        service = ProtocolService(client=client)

        with pytest.raises(RuntimeError, match="provider failure"):
            service.generate_protocol(
                instructions="Инструкции",
                transcript="Стенограмма",
                config=config,
            )



