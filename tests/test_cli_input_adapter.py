"""Тесты для входного адаптера CLI команды scribe."""

from unittest.mock import Mock

import pytest

from app.adapters.input.cli import (
    DEFAULT_BEAM_SIZE,
    ScribeCommandHandler,
    ScribeCommandOptions,
)
from app.application.ports import ITranscriptSegmentWriter, ITranscriptionEngine


@pytest.mark.unit
class TestScribeCommandHandler:
    """Набор тестов для ScribeCommandHandler."""

    def test_execute_invokes_dependencies_with_expected_arguments(self):
        """Обработчик должен корректно вызывать фабрики и сервис."""
        adapter = Mock(spec=ITranscriptionEngine)
        resolved_model_name = "resolved-small"

        adapter_factory = Mock(return_value=(adapter, resolved_model_name))
        service = Mock()
        service.transcribe.return_value = iter(["segment"])
        service_factory = Mock(return_value=service)
        writer = Mock(spec=ITranscriptSegmentWriter)
        writer_factory = Mock(return_value=writer)

        handler = ScribeCommandHandler(
            transcription_adapter_factory=adapter_factory,
            transcription_service_factory=service_factory,
            transcript_writer_factory=writer_factory,
        )

        options = ScribeCommandOptions(
            input_path="audio.mp3",
            output_path="out.txt",
            model="faster:base",
            language="en",
            compute_type="float16",
            verbose=False,
        )

        handler.execute(options)

        adapter_factory.assert_called_once_with("faster:base", "float16")
        service_factory.assert_called_once_with(adapter)
        writer_factory.assert_called_once_with("out.txt", False)
        service.transcribe.assert_called_once_with(
            input_path="audio.mp3",
            output_writer=writer,
            model_name=resolved_model_name,
            language="en",
            verbose=False,
            beam_size=DEFAULT_BEAM_SIZE,
        )

    def test_execute_closes_writer_and_reraises_on_error(self):
        """При ошибке транскрипции writer должен закрываться и исключение пробрасывается."""
        adapter = Mock(spec=ITranscriptionEngine)
        adapter_factory = Mock(return_value=(adapter, "resolved-model"))

        service = Mock()
        service.transcribe.side_effect = RuntimeError("engine failure")
        service_factory = Mock(return_value=service)

        writer = Mock(spec=ITranscriptSegmentWriter)
        writer_factory = Mock(return_value=writer)

        handler = ScribeCommandHandler(
            transcription_adapter_factory=adapter_factory,
            transcription_service_factory=service_factory,
            transcript_writer_factory=writer_factory,
        )

        options = ScribeCommandOptions(
            input_path="audio.mp3",
            output_path="out.txt",
            model="small",
            language="ru",
            compute_type="int8",
            verbose=True,
        )

        with pytest.raises(RuntimeError, match="engine failure"):
            handler.execute(options)

        writer.close.assert_called_once_with()


