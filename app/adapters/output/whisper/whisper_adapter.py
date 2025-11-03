"""Адаптер для OpenAI Whisper."""

from typing import Iterator, Any
from app.application.ports import ITranscriptionEngine
from app.domain.models.transcript import Segment


class WhisperAdapter(ITranscriptionEngine):
    """Адаптер для OpenAI Whisper.
    
    Оборачивает библиотеку OpenAI Whisper и адаптирует её API
    к нашему интерфейсу ITranscriptionEngine.
    """
    
    def __init__(self, whisper_module):
        """
        Args:
            whisper_module: Модуль OpenAI Whisper
        """
        self._whisper = whisper_module
    
    def load_model(self, model_name: str, **kwargs) -> Any:
        """Загружает модель OpenAI Whisper.
        
        Args:
            model_name: Название модели (например, 'base', 'small', 'medium')
            **kwargs: Дополнительные параметры (игнорируются для OpenAI Whisper)
        
        Returns:
            Загруженная модель Whisper
        """
        return self._whisper.load_model(model_name)
    
    def transcribe(self,
                   model: Any,
                   audio_path: str,
                   language: str,
                   **kwargs) -> Iterator[Segment]:
        """Выполняет транскрипцию аудиофайла через OpenAI Whisper.
        
        Args:
            model: Загруженная модель Whisper (результат load_model)
            audio_path: Путь к аудиофайлу
            language: Код языка транскрипции (ISO 639-1, например 'ru', 'en')
            **kwargs: Дополнительные параметры (verbose и т.д.)
        
        Yields:
            Segment: Сегменты транскрипции с таймингами
        """
        verbose = kwargs.get('verbose', True)
        result = model.transcribe(audio_path, language=language, verbose=verbose)
        
        # Конвертируем словари OpenAI Whisper в доменные модели Segment
        for segment_dict in result['segments']:
            yield Segment(
                start=segment_dict['start'],
                end=segment_dict['end'],
                text=segment_dict['text'].strip()
            )

