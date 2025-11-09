"""Входные адаптеры для CLI-команд."""

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple

from app.adapters.output import FileOutputWriter
from app.application.services.word_analysis import WordAnalysisService
from app.application.ports import ITranscriptionEngine, ITranscriptSegmentWriter
from app.domain.models.protocol import ProtocolConfig
from app.domain.models.word_analysis import WordAnalysisConfig
from app.factories import (
    create_transcription_adapter,
    create_transcription_service,
    create_protocol_client,
    create_protocol_service,
    create_word_analysis_service,
)
from app.utils.config import load_config

DEFAULT_BEAM_SIZE = 5


@dataclass(frozen=True)
class ScribeCommandOptions:
    """Структура входных параметров для команды scribe."""

    input_path: str
    output_path: str
    model: str = "small"
    language: str = "ru"
    compute_type: str = "int8"
    verbose: bool = True


class ScribeCommandHandler:
    """Оркестрация команды scribe."""

    def __init__(
        self,
        transcription_adapter_factory: Optional[
            Callable[[str, str], Tuple[ITranscriptionEngine, str]]
        ] = None,
        transcription_service_factory: Optional[
            Callable[[ITranscriptionEngine], Any]
        ] = None,
        transcript_writer_factory: Optional[
            Callable[[str, bool], ITranscriptSegmentWriter]
        ] = None,
    ) -> None:
        self._transcription_adapter_factory = (
            transcription_adapter_factory or self._default_adapter_factory
        )
        self._transcription_service_factory = (
            transcription_service_factory or self._default_service_factory
        )
        self._transcript_writer_factory = (
            transcript_writer_factory or self._default_writer_factory
        )

    def execute(self, options: ScribeCommandOptions) -> None:
        adapter, model_name = self._transcription_adapter_factory(
            options.model, options.compute_type
        )
        service = self._transcription_service_factory(adapter)
        writer = self._transcript_writer_factory(options.output_path, options.verbose)

        try:
            list(
                service.transcribe(
                    input_path=options.input_path,
                    output_writer=writer,
                    model_name=model_name,
                    language=options.language,
                    verbose=options.verbose,
                    beam_size=DEFAULT_BEAM_SIZE,
                )
            )
        except Exception:
            writer.close()
            raise

    @staticmethod
    def _default_adapter_factory(model: str, compute_type: str) -> Tuple[ITranscriptionEngine, str]:
        return create_transcription_adapter(model=model, compute_type=compute_type)

    @staticmethod
    def _default_service_factory(engine: ITranscriptionEngine):
        return create_transcription_service(engine=engine)

    @staticmethod
    def _default_writer_factory(output_path: str, verbose: bool) -> ITranscriptSegmentWriter:
        return FileOutputWriter(output_path=output_path, verbose=verbose)


@dataclass(frozen=True)
class ProtocolCommandOptions:
    transcript_path: str
    output_path: Optional[str]
    config_path: Optional[str]


class ProtocolCommandHandler:
    """Оркестрация команды protocol."""

    def __init__(
        self,
        config_loader: Optional[Callable[[str], dict]] = None,
        config_parser: Optional[Callable[[str, dict, Optional[str]], ProtocolConfig]] = None,
        instructions_reader: Optional[Callable[[str], str]] = None,
        transcript_reader: Optional[Callable[[str], str]] = None,
        protocol_client_factory: Optional[Callable[[ProtocolConfig], Any]] = None,
        protocol_service_factory: Optional[Callable[[Any], Any]] = None,
        output_writer: Optional[Callable[[Optional[str], str], None]] = None,
    ) -> None:
        self._config_loader = config_loader or load_config
        self._config_parser = config_parser or self._default_config_parser
        self._instructions_reader = instructions_reader or self._read_text_file
        self._transcript_reader = transcript_reader or self._read_text_file
        self._protocol_client_factory = protocol_client_factory or create_protocol_client
        self._protocol_service_factory = protocol_service_factory or create_protocol_service
        self._output_writer = output_writer or self._default_output_writer

    def execute(self, options: ProtocolCommandOptions) -> None:
        if not os.path.exists(options.transcript_path):
            raise FileNotFoundError(f"Файл с расшифровкой не найден: {options.transcript_path}")

        config_path = options.config_path
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "config.yaml"))

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Файл конфигурации не найден: {config_path}. Укажите путь через --config"
            )

        raw_config = self._config_loader(config_path)
        provider_key = raw_config.get("provider", "deepseek")
        provider_section = raw_config.get(provider_key, {})
        if not provider_section:
            raise ValueError(f"В конфиге отсутствует секция '{provider_key}'")

        instructions_path = provider_section.get("instructions")
        if instructions_path and not os.path.isabs(instructions_path):
            config_dir = os.path.dirname(config_path)
            instructions_path = os.path.join(config_dir, instructions_path)

        config = self._config_parser(provider_key, provider_section, instructions_path)

        instructions_text = ""
        if instructions_path:
            if not os.path.exists(instructions_path):
                raise FileNotFoundError(f"Файл с инструкциями не найден: {instructions_path}")
            instructions_text = self._instructions_reader(instructions_path)

        transcript_text = self._transcript_reader(options.transcript_path)
        print("Отправка запроса к провайдеру протоколов...", flush=True)

        client = self._protocol_client_factory(config)
        service = self._protocol_service_factory(client)
        response = service.generate_protocol(
            instructions=instructions_text,
            transcript=transcript_text,
            config=config,
        )
        self._output_writer(options.output_path, response.content)

    @staticmethod
    def _default_config_parser(
        provider: str,
        section: dict,
        instructions_path: Optional[str],
    ) -> ProtocolConfig:
        api_key = section.get("api_key")
        if not api_key:
            raise ValueError(f"В конфиге провайдера '{provider}' отсутствует api_key")

        model = section.get("model", "deepseek-chat")
        temperature = section.get("temperature", 0.7)
        known_keys = {"api_key", "model", "instructions", "temperature"}
        extra_params = {k: v for k, v in section.items() if k not in known_keys}

        return ProtocolConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            instructions_path=instructions_path,
            temperature=temperature,
            extra_params=extra_params,
        )

    @staticmethod
    def _read_text_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _default_output_writer(output_path: Optional[str], content: str) -> None:
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            print(content)


@dataclass(frozen=True)
class TagCommandOptions:
    transcript_path: str
    output_path: Optional[str]
    limit: int = 50
    lemmatize: bool = False
    stopwords_path: Optional[str] = None
    exclude_names: bool = False


class TagCommandHandler:
    """Оркестрация команды tag."""

    def __init__(
        self,
        file_reader: Optional[Callable[[str], Iterable[str]]] = None,
        stopwords_loader: Optional[Callable[[Optional[str]], Iterable[str]]] = None,
        analysis_service_factory: Optional[Callable[[], WordAnalysisService]] = None,
        output_writer: Optional[Callable[[Optional[str], str], None]] = None,
    ) -> None:
        self._file_reader = file_reader or self._default_file_reader
        self._stopwords_loader = stopwords_loader or self._default_stopwords_loader
        self._analysis_service_factory = analysis_service_factory or create_word_analysis_service
        self._output_writer = output_writer or self._default_output_writer

    def execute(self, options: TagCommandOptions) -> None:
        if not os.path.exists(options.transcript_path):
            raise FileNotFoundError(f"Файл с расшифровкой не найден: {options.transcript_path}")

        lines = self._file_reader(options.transcript_path)
        stopwords = self._stopwords_loader(options.stopwords_path)
        config = WordAnalysisConfig(
            lemmatize=options.lemmatize,
            exclude_names=options.exclude_names,
            limit=options.limit,
        )

        service = self._analysis_service_factory()
        result = service.analyze(lines=lines, stopwords=stopwords, config=config)
        self._output_writer(options.output_path, result.to_text())

    @staticmethod
    def _default_file_reader(path: str) -> Iterable[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.readlines()
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise IOError(f"Ошибка при чтении файла {path}: {exc}")

    @staticmethod
    def _default_stopwords_loader(path: Optional[str]) -> Iterable[str]:
        if not path:
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip().lower() for line in f if line.strip()]
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise IOError(f"Ошибка при чтении файла со стоп-словами {path}: {exc}")

    @staticmethod
    def _default_output_writer(output_path: Optional[str], content: str) -> None:
        if output_path:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as exc:
                raise IOError(f"Ошибка при записи в файл {output_path}: {exc}")
        else:
            print(content)
