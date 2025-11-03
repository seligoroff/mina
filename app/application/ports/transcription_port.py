"""Порт (интерфейс) для движков транскрипции."""

from abc import ABC, abstractmethod
from typing import Iterator, Any, Optional
from app.domain.models.transcript import Segment


class ITranscriptionEngine(ABC):
    """Порт (интерфейс) для движков транскрипции.
    
    Этот интерфейс абстрагирует различные реализации транскрипции
    (Whisper, faster-whisper, будущие движки), что позволяет:
    - Легко заменять движки без изменения бизнес-логики
    - Мокировать в тестах
    - Следовать принципу инверсии зависимостей (Dependency Inversion Principle)
    """
    
    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> Any:
        """Загружает модель для транскрипции.
        
        Args:
            model_name: Название модели (например, 'base', 'small', 'medium')
            **kwargs: Дополнительные параметры (игнорируются большинством адаптеров)
                     Примечание: Специфичные параметры (например, compute_type для faster-whisper)
                     должны быть переданы в конструктор адаптера, а не в load_model()
        
        Returns:
            Загруженная модель (тип зависит от реализации)
        """
        ...
    
    @abstractmethod
    def transcribe(self, 
                   model: Any,
                   audio_path: str,
                   language: str,
                   **kwargs) -> Iterator[Segment]:
        """Выполняет транскрипцию аудиофайла.
        
        Args:
            model: Загруженная модель (результат load_model)
            audio_path: Путь к аудиофайлу
            language: Код языка транскрипции (ISO 639-1, например 'ru', 'en')
            **kwargs: Дополнительные параметры (beam_size для faster-whisper и т.д.)
        
        Yields:
            Segment: Сегменты транскрипции с таймингами
        """
        ...


