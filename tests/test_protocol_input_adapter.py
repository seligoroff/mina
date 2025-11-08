"""Тесты для входного адаптера команды protocol."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from app.adapters.input.cli import (
    ProtocolCommandHandler,
    ProtocolCommandOptions,
)
from app.domain.models.protocol import ProtocolConfig, ProtocolResponse


@pytest.mark.unit
class TestProtocolCommandHandlerUnit:
    """Юнит-тесты для ProtocolCommandHandler."""

    def test_execute_calls_service_and_writer(self, tmp_path):
        transcript_file = tmp_path / "transcript.txt"
        transcript_file.write_text("raw transcript", encoding="utf-8")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("placeholder", encoding="utf-8")

        instructions_file = tmp_path / "instructions.md"
        instructions_file.write_text("instructions", encoding="utf-8")

        config_loader = Mock(return_value={
            "provider": "deepseek",
            "deepseek": {
                "api_key": "key",
                "model": "deepseek-chat",
                "instructions": str(instructions_file),
            },
        })
        config = ProtocolConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key="key",
            instructions_path=str(instructions_file),
            temperature=0.7,
        )
        config_parser = Mock(return_value=config)
        instructions_reader = Mock(return_value="INST")
        transcript_reader = Mock(return_value="TRANS")

        protocol_client = object()
        protocol_client_factory = Mock(return_value=protocol_client)
        response = ProtocolResponse(content="RESULT")
        service = Mock()
        service.generate_protocol.return_value = response
        protocol_service_factory = Mock(return_value=service)
        output_writer = Mock()

        handler = ProtocolCommandHandler(
            config_loader=config_loader,
            config_parser=config_parser,
            instructions_reader=instructions_reader,
            transcript_reader=transcript_reader,
            protocol_client_factory=protocol_client_factory,
            protocol_service_factory=protocol_service_factory,
            output_writer=output_writer,
        )

        options = ProtocolCommandOptions(
            transcript_path=str(transcript_file),
            output_path=str(tmp_path / "out.txt"),
            config_path=str(config_file),
        )

        handler.execute(options)

        config_loader.assert_called_once_with(str(config_file))
        config_parser.assert_called_once()
        instructions_reader.assert_called_once_with(str(instructions_file))
        transcript_reader.assert_called_once_with(str(transcript_file))
        protocol_client_factory.assert_called_once_with(config)
        protocol_service_factory.assert_called_once_with(protocol_client)
        service.generate_protocol.assert_called_once_with(
            instructions="INST",
            transcript="TRANS",
            config=config,
        )
        output_writer.assert_called_once_with(str(tmp_path / "out.txt"), "RESULT")

    def test_execute_raises_when_provider_section_missing(self, tmp_path):
        transcript_file = tmp_path / "input.txt"
        transcript_file.write_text("data", encoding="utf-8")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("placeholder", encoding="utf-8")

        handler = ProtocolCommandHandler(
            config_loader=lambda _: {},
            config_parser=Mock(),
            instructions_reader=Mock(),
            transcript_reader=Mock(return_value=""),
            protocol_client_factory=Mock(),
            protocol_service_factory=Mock(),
            output_writer=Mock(),
        )

        options = ProtocolCommandOptions(
            transcript_path=str(transcript_file),
            output_path=None,
            config_path=str(config_file),
        )

        with pytest.raises(ValueError, match="отсутствует секция"):
            handler.execute(options)


@pytest.mark.unit
def test_protocol_handler_integration_flow(tmp_path):
    """Мини-интеграционный тест: реальные чтения файлов, заглушки клиента/сервиса."""
    transcript_path = tmp_path / "transcript.txt"
    transcript_text = "Это стенограмма встречи."
    transcript_path.write_text(transcript_text, encoding="utf-8")

    instructions_path = tmp_path / "instructions.md"
    instructions_text = "Сформируй протокол."
    instructions_path.write_text(instructions_text, encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join([
            "provider: deepseek",
            "deepseek:",
            "  api_key: test-key",
            "  model: deepseek-chat",
            "  temperature: 0.8",
            f"  instructions: {instructions_path.name}",
        ]),
        encoding="utf-8",
    )

    captured = {}

    class DummyService:
        def __init__(self, expected_config: ProtocolConfig):
            self.expected_config = expected_config

        def generate_protocol(self, instructions: str, transcript: str, config: ProtocolConfig):
            assert instructions == instructions_text
            assert transcript == transcript_text
            assert config.provider == "deepseek"
            return ProtocolResponse(content="Готовый протокол")

    def protocol_client_factory(config: ProtocolConfig):
        captured["config"] = config
        return object()

    def protocol_service_factory(client):
        return DummyService(captured["config"])

    def output_writer(path: str, content: str):
        captured["output_path"] = path
        captured["content"] = content

    handler = ProtocolCommandHandler(
        protocol_client_factory=protocol_client_factory,
        protocol_service_factory=protocol_service_factory,
        output_writer=output_writer,
    )

    options = ProtocolCommandOptions(
        transcript_path=str(transcript_path),
        output_path=str(tmp_path / "protocol.txt"),
        config_path=str(config_path),
    )

    handler.execute(options)

    assert captured["output_path"] == str(tmp_path / "protocol.txt")
    assert captured["content"] == "Готовый протокол"
    assert captured["config"].api_key == "test-key"
    assert captured["config"].instructions_path == str(instructions_path)

