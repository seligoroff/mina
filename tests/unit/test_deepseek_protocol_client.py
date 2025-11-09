"""Тесты для DeepSeekProtocolClient."""

from unittest.mock import Mock

import pytest
import requests

from app.adapters.output.api import DeepSeekProtocolClient
from app.domain.models.protocol import ProtocolConfig, ProtocolRequest
from app.domain.exceptions import ProtocolClientError


@pytest.mark.unit
class TestDeepSeekProtocolClient:
    """Набор тестов для адаптера DeepSeek."""

    def _create_request(self) -> ProtocolRequest:
        config = ProtocolConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key="key",
            temperature=0.7,
        )
        return ProtocolRequest(
            instructions="Инструкции",
            transcript="Стенограмма",
            config=config,
        )

    def test_generate_protocol_success(self):
        request = self._create_request()
        http_client = Mock()
        http_client.post.return_value.ok = True
        http_client.post.return_value.json.return_value = {
            "choices": [
                {"message": {"content": "Готовый протокол"}},
            ],
        }

        client = DeepSeekProtocolClient(api_key="key", http_client=http_client)
        response = client.generate_protocol(request)

        http_client.post.assert_called_once()
        args, kwargs = http_client.post.call_args
        assert args[0] == DeepSeekProtocolClient.DEFAULT_BASE_URL
        assert kwargs["headers"]["Authorization"] == "Bearer key"
        assert kwargs["json"]["model"] == "deepseek-chat"
        assert kwargs["json"]["temperature"] == 0.7
        assert response.content == "Готовый протокол"
        assert "choices" in response.provider_raw

    def test_generate_protocol_merges_extra_params(self):
        config = ProtocolConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key="key",
            extra_params={"max_tokens": 4096},
        )
        request = ProtocolRequest(
            instructions="Инструкции",
            transcript="Стенограмма",
            config=config,
        )
        http_client = Mock()
        http_client.post.return_value.ok = True
        http_client.post.return_value.json.return_value = {
            "choices": [
                {"message": {"content": "Готовый протокол"}},
            ],
        }

        client = DeepSeekProtocolClient(api_key="key", http_client=http_client)
        client.generate_protocol(request)

        payload = http_client.post.call_args.kwargs["json"]
        assert payload["max_tokens"] == 4096

    def test_generate_protocol_handles_http_error(self):
        request = self._create_request()
        http_client = Mock()
        mock_response = Mock(ok=False, status_code=500, text="error")
        mock_response.json.side_effect = ValueError("bad json")
        http_client.post.return_value = mock_response

        client = DeepSeekProtocolClient(api_key="key", http_client=http_client)

        with pytest.raises(ProtocolClientError, match="DeepSeek вернул ошибку 500"):
            client.generate_protocol(request)

    def test_generate_protocol_handles_network_error(self):
        request = self._create_request()
        http_client = Mock()
        http_client.post.side_effect = requests.RequestException("network")

        client = DeepSeekProtocolClient(api_key="key", http_client=http_client)

        with pytest.raises(ProtocolClientError, match="Ошибка при отправке"):
            client.generate_protocol(request)

    def test_generate_protocol_handles_invalid_json(self):
        request = self._create_request()
        http_client = Mock()
        mock_response = Mock(ok=True)
        mock_response.json.side_effect = ValueError("invalid")
        http_client.post.return_value = mock_response

        client = DeepSeekProtocolClient(api_key="key", http_client=http_client)

        with pytest.raises(ProtocolClientError, match="Некорректный JSON"):
            client.generate_protocol(request)

    def test_generate_protocol_handles_missing_content(self):
        request = self._create_request()
        http_client = Mock()
        mock_response = Mock(ok=True)
        mock_response.json.return_value = {}
        http_client.post.return_value = mock_response

        client = DeepSeekProtocolClient(api_key="key", http_client=http_client)

        with pytest.raises(ProtocolClientError, match="не содержит контент"):
            client.generate_protocol(request)



