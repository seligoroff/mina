from pathlib import Path

import pytest

from app.adapters.input.cli import ProtocolCommandHandler, ProtocolCommandOptions
from app.domain.models.protocol import ProtocolConfig, ProtocolResponse


@pytest.mark.integration
def test_protocol_handler_integration_flow(tmp_path: Path):
    """Мини-интеграционный сценарий с реальными файлами и заглушками сервисов."""
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
        transcript_reader=lambda path: transcript_path.read_text(encoding="utf-8"),
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



