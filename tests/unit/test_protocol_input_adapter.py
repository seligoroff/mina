"""Тесты для входного адаптера команды protocol."""

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

    def test_execute_raises_when_transcript_missing(self, tmp_path):
        handler = ProtocolCommandHandler(
            config_loader=Mock(return_value={"deepseek": {"api_key": "key"}})
        )

        options = ProtocolCommandOptions(
            transcript_path=str(tmp_path / "missing.txt"),
            output_path=None,
            config_path=str(tmp_path / "config.yaml"),
        )

        with pytest.raises(FileNotFoundError, match="Файл с расшифровкой не найден"):
            handler.execute(options)

    def test_execute_raises_when_instructions_missing(self, tmp_path):
        transcript_file = tmp_path / "input.txt"
        transcript_file.write_text("data", encoding="utf-8")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("provider: deepseek\ndeepseek:\n  api_key: key\n  instructions: missing.md", encoding="utf-8")

        handler = ProtocolCommandHandler()

        options = ProtocolCommandOptions(
            transcript_path=str(transcript_file),
            output_path=None,
            config_path=str(config_file),
        )

        with pytest.raises(FileNotFoundError, match="Файл с инструкциями не найден"):
            handler.execute(options)


