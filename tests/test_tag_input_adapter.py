"""Тесты для TagCommandHandler."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from app.adapters.input.cli import TagCommandHandler, TagCommandOptions
from app.domain.models.word_analysis import WordFrequencyResult


@pytest.mark.unit
class TestTagCommandHandler:
    def test_execute_writes_output(self, tmp_path: Path):
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("foo bar", encoding="utf-8")

        file_reader = Mock(return_value=["foo bar"])
        stopwords_loader = Mock(return_value=["bar"])
        service = Mock()
        service.analyze.return_value = WordFrequencyResult(items=[("foo", 1)])
        analysis_factory = Mock(return_value=service)
        output_writer = Mock()

        handler = TagCommandHandler(
            file_reader=file_reader,
            stopwords_loader=stopwords_loader,
            analysis_service_factory=analysis_factory,
            output_writer=output_writer,
        )

        options = TagCommandOptions(
            transcript_path=str(transcript),
            output_path=str(tmp_path / "out.txt"),
            limit=10,
            lemmatize=True,
            stopwords_path="stopwords.txt",
            exclude_names=True,
        )

        handler.execute(options)

        file_reader.assert_called_once_with(str(transcript))
        stopwords_loader.assert_called_once_with("stopwords.txt")
        analysis_factory.assert_called_once()
        service.analyze.assert_called_once()
        output_writer.assert_called_once_with(str(tmp_path / "out.txt"), "foo: 1")

    def test_execute_raises_when_file_missing(self):
        handler = TagCommandHandler(
            file_reader=Mock(),
            stopwords_loader=Mock(),
            analysis_service_factory=Mock(),
            output_writer=Mock(),
        )

        options = TagCommandOptions(
            transcript_path="missing.txt",
            output_path=None,
        )

        with pytest.raises(FileNotFoundError):
            handler.execute(options)

