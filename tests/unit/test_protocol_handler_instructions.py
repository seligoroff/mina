from pathlib import Path

from app.adapters.input.cli import ProtocolCommandHandler, ProtocolCommandOptions


def _fake_load_config(_path: str) -> dict:
    return {
        "provider": "deepseek",
        "deepseek": {
            "api_key": "x",
            "model": "deepseek-chat",
            "instructions": "resources/protocol/deepseek-protocol-instructions.md",
            "temperature": 0.7,
        },
    }


def _fake_read_text(path: str) -> str:
    return f"BASE_INSTR({Path(path).name})"


def _fake_transcript_reader(path: str) -> str:
    return "dummy transcript"


class _DummyClient:
    pass


class _DummyService:
    def __init__(self, _client):
        pass

    def generate_protocol(self, *, instructions, transcript, config):
        class R:
            content = f"INSTR={instructions}\nTRANSCRIPT={transcript[:5]}"
        return R()


def _fake_client_factory(_cfg):
    return _DummyClient()


def _fake_service_factory(_client):
    return _DummyService(_client)


def _capture_output(holder: list, content: str) -> None:
    holder.append(content)


def test_protocol_uses_override_and_extra(tmp_path):
    out = []
    h = ProtocolCommandHandler(
        config_loader=_fake_load_config,
        instructions_reader=_fake_read_text,
        transcript_reader=_fake_transcript_reader,
        protocol_client_factory=_fake_client_factory,
        protocol_service_factory=_fake_service_factory,
        output_writer=lambda p, c: _capture_output(out, c),
    )

    # Prepare real files to satisfy existence checks
    transcript_path = tmp_path / "t.txt"
    transcript_path.write_text("ignored", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("provider: deepseek", encoding="utf-8")

    # Use absolute path to existing instructions file in repo (src root is parents[2])
    repo_src = Path(__file__).resolve().parents[2]  # .../src
    instr_path = repo_src / "resources/protocol/telegram-chat-instructions.md"

    opts = ProtocolCommandOptions(
        transcript_path=str(transcript_path),
        output_path=None,
        config_path=str(config_path),
        instructions_override=str(instr_path),
        extra_text="FOCUS: video issues",
    )

    h.execute(opts)

    output = out[0]
    assert "telegram-chat-instructions.md" in output
    assert "FOCUS: video issues" in output

