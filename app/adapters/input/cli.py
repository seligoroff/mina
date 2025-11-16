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
from app.utils.config import get_telegram_config
from app.domain.models import ChatExportConfig
from app.application.ports import IChatExporter, IMessageStore
from app.factories import create_telethon_chat_exporter
from app.adapters.output.storage import FileMessageStore
from datetime import datetime
from typing import List

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
    instructions_override: Optional[str] = None
    extra_text: Optional[str] = None


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

        # Опциональная переопределяемая инструкция
        instructions_path = options.instructions_override or provider_section.get("instructions")
        if instructions_path and not os.path.isabs(instructions_path):
            # если путь относительный — разрешаем относительно директории конфига
            config_dir = os.path.dirname(config_path)
            instructions_path = os.path.join(config_dir, instructions_path)

        config = self._config_parser(provider_key, provider_section, instructions_path)

        instructions_text = ""
        if instructions_path:
            if not os.path.exists(instructions_path):
                raise FileNotFoundError(f"Файл с инструкциями не найден: {instructions_path}")
            instructions_text = self._instructions_reader(instructions_path)
        # Добавить дополнительное уточнение, если передано
        if options.extra_text:
            suffix = options.extra_text.strip()
            if suffix:
                instructions_text = (instructions_text + "\n\n" if instructions_text else "") + suffix

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
class TelegramLoginOptions:
    """Параметры для инициализации Telegram-сессии."""
    config_path: Optional[str] = None


class TelegramLoginCommandHandler:
    """Инициализация (логин) Telethon-сессии по данным из config.yaml."""

    def __init__(
        self,
        config_loader: Optional[Callable[[str], dict]] = None,
        path_expander: Optional[Callable[[str], str]] = None,
        ensure_dir: Optional[Callable[[str], None]] = None,
        stdout: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._config_loader = config_loader or load_config
        self._path_expander = path_expander or self._default_expanduser
        self._ensure_dir = ensure_dir or self._default_ensure_dir
        self._stdout = stdout or (lambda m: print(m, flush=True))

    def execute(self, options: TelegramLoginOptions) -> None:
        config_path = options.config_path
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "config.yaml"))
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}. Укажите путь через --config")

        raw = self._config_loader(config_path)
        tg = raw.get("telegram", {})
        api_id = str(tg.get("api_id", "")).strip()
        api_hash = tg.get("api_hash")
        session_path = tg.get("session_path")
        if not api_id or not api_hash or not session_path:
            raise ValueError("В секции telegram требуется заполнить api_id, api_hash и session_path")

        # Удаляем возможные префиксные двоеточия для совместимости
        while api_id.startswith(":"):
            api_id = api_id[1:]
        try:
            api_id_int = int(api_id)
        except ValueError:
            raise ValueError("telegram.api_id должен быть числом (или строкой, приводимой к числу)")

        session_path = self._path_expander(session_path)
        self._ensure_dir(os.path.dirname(session_path))

        # Lazy import Telethon, чтобы не тянуть зависимость при отсутствии команды
        try:
            from telethon import TelegramClient  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Не удалось импортировать Telethon: {exc}")

        self._stdout(f"Инициализация сессии: {session_path}")
        client = TelegramClient(session_path, api_id_int, api_hash)
        # Это вызовет интерактивный вход (номер, код)
        client.start()
        self._stdout("Готово: сессия создана/обновлена.")

    @staticmethod
    def _default_expanduser(path: str) -> str:
        return os.path.expanduser(path)

    @staticmethod
    def _default_ensure_dir(directory: str) -> None:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

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

@dataclass(frozen=True)
class ExportChatCommandOptions:
    """Параметры для экспорта переписки из Telegram."""
    peer: str
    output_path: str
    output_format: str = "jsonl"
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    include_media: bool = False
    limit_per_call: int = 200
    rate_limit_s: float = 0.2
    config_path: Optional[str] = None


class ExportChatCommandHandler:
    """Оркестрация команды export-chat."""

    def __init__(
        self,
        config_loader: Optional[Callable[[str], dict]] = None,
        exporter_factory: Optional[Callable[[dict], IChatExporter]] = None,
        message_store: Optional[IMessageStore] = None,
        stdout: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._config_loader = config_loader or load_config
        self._exporter_factory = exporter_factory or self._default_exporter_factory
        self._message_store = message_store or FileMessageStore()
        self._stdout = stdout or (lambda m: print(m, flush=True))

    def execute(self, options: ExportChatCommandOptions) -> None:
        config_path = options.config_path
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "config.yaml"))
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}. Укажите путь через --config")

        raw = self._config_loader(config_path)
        tg = get_telegram_config(raw)

        exporter = self._exporter_factory(tg)
        chat_cfg = ChatExportConfig(
            peer=options.peer or tg.get("peer", ""),
            date_from=self._parse_iso8601(options.date_from),
            date_to=self._parse_iso8601(options.date_to),
            limit_per_call=options.limit_per_call,
            include_media=options.include_media,
            output_format=options.output_format.lower(),
            rate_limit_s=options.rate_limit_s,
        )
        if chat_cfg.date_from and chat_cfg.date_to and chat_cfg.date_from > chat_cfg.date_to:
            raise ValueError("Некорректный диапазон: --from больше, чем --to")

        self._stdout(f"Экспорт чата '{chat_cfg.peer}' → {options.output_path} ({chat_cfg.output_format})")
        try:
            messages = exporter.export(chat_cfg)
            count = 0
            first_date: Optional[datetime] = None
            last_date: Optional[datetime] = None

            def with_progress(iterable):
                nonlocal count, first_date, last_date
                for msg in iterable:
                    count += 1
                    if first_date is None or msg.date < first_date:
                        first_date = msg.date
                    if last_date is None or msg.date > last_date:
                        last_date = msg.date
                    if count % 20 == 0:
                        self._stdout(f"{count} сообщений...")
                    yield msg

            self._message_store.write(with_progress(messages), options.output_path, format=chat_cfg.output_format)
            span = ""
            if first_date and last_date:
                span = f", период: {first_date.isoformat()} → {last_date.isoformat()}"
            self._stdout(f"Готово: записано {count} сообщений в {options.output_path}{span}")
        except ValueError as e:
            raise e
        except Exception as e:
            hint = " Если сессия не создана, выполните команду: mina tg-login"
            raise RuntimeError(f"Ошибка экспорта: {e}.{hint}")

    @staticmethod
    def _default_exporter_factory(tg: dict) -> IChatExporter:
        return create_telethon_chat_exporter(
            dependencies={
                "api_id": tg["api_id"],
                "api_hash": tg["api_hash"],
                "session_path": tg["session_path"],
                "rate_limit_s": tg.get("rate_limit_s", 0.2),
            }
        )

    @staticmethod
    def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        val = value.strip()
        if val.endswith("Z"):
            val = val[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(val)
        except Exception:
            raise ValueError(f"Некорректный формат даты: {value}. Ожидается ISO8601, напр. 2024-01-01T00:00:00+03:00")

        return None

@dataclass(frozen=True)
class TelegramListChatsOptions:
    """Параметры для команды списка доступных чатов/диалогов."""
    config_path: Optional[str] = None
    limit: int = 100
    include_users: bool = True
    include_groups: bool = True
    include_channels: bool = True


class TelegramListChatsCommandHandler:
    """Выводит список доступных чатов/каналов/диалогов из Telethon-сессии."""

    def __init__(
        self,
        config_loader: Optional[Callable[[str], dict]] = None,
        stdout: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._config_loader = config_loader or load_config
        self._stdout = stdout or (lambda m: print(m, flush=True))

    def execute(self, options: TelegramListChatsOptions) -> None:
        config_path = options.config_path
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "config.yaml"))
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}. Укажите путь через --config")

        raw = self._config_loader(config_path)
        tg = get_telegram_config(raw)

        # Lazy import, чтобы не тянуть зависимость без надобности
        try:
            from telethon import TelegramClient  # type: ignore[import]
            from telethon.tl.types import User, Chat, Channel  # type: ignore[import]
        except Exception as exc:
            raise RuntimeError(f"Не удалось импортировать Telethon: {exc}")

        async def _run():
            async with TelegramClient(tg["session_path"], tg["api_id"], tg["api_hash"]) as client:
                out: List[str] = []
                idx = 0
                async for d in client.iter_dialogs(limit=options.limit):
                    ent = d.entity
                    kind = None
                    if isinstance(ent, User) and options.include_users:
                        kind = "user"
                    elif isinstance(ent, Chat) and options.include_groups:
                        kind = "group"
                    elif isinstance(ent, Channel) and options.include_channels:
                        kind = "channel"
                    if kind is None:
                        continue
                    name = getattr(ent, "title", None) or getattr(ent, "first_name", None) or ""
                    username = getattr(ent, "username", None)
                    peer_repr = f"@{username}" if username else f"id={getattr(ent, 'id', 'unknown')}"
                    idx += 1
                    out.append(f"{idx:>3}. [{kind}] {name} ({peer_repr})")
                if not out:
                    self._stdout("Нет доступных диалогов по заданным фильтрам.")
                else:
                    for line in out:
                        self._stdout(line)

        import asyncio as _asyncio
        _asyncio.run(_run())
