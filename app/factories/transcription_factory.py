"""Фабрики для создания компонентов транскрипции."""

from typing import Tuple, Optional
from app.application.ports import ITranscriptionEngine


def _create_transcription_adapter_internal(model: str,
                                          whisper_module,
                                          faster_whisper_model_class,
                                          compute_type: str = 'int8') -> Tuple[ITranscriptionEngine, str]:
    """
    Внутренний метод для создания адаптера транскрипции.
    
    Определяет тип движка по формату модели и создает соответствующий адаптер.
    
    Args:
        model: Название модели или "faster:model_name" для faster-whisper
        whisper_module: Модуль OpenAI Whisper
        faster_whisper_model_class: Класс WhisperModel из faster_whisper
        compute_type: Тип вычислений для faster-whisper ('int8', 'float16', 'float32')
    
    Returns:
        Tuple[ITranscriptionEngine, str]: Адаптер и имя модели (без префикса "faster:")
    """
    use_faster = model.startswith('faster:')
    
    if use_faster:
        # Извлекаем имя модели из "faster:model_name"
        _, model_name = model.split(':', 1)
        
        # Создаем адаптер для faster-whisper с compute_type в конструкторе
        from app.adapters.output.whisper import FasterWhisperAdapter
        adapter = FasterWhisperAdapter(faster_whisper_model_class, compute_type=compute_type)
        
        return adapter, model_name
    else:
        model_name = model
        
        # Создаем адаптер для OpenAI Whisper
        from app.adapters.output.whisper import WhisperAdapter
        adapter = WhisperAdapter(whisper_module)
        
        return adapter, model_name


def create_transcription_adapter(model: str,
                                 compute_type: str = 'int8',
                                 dependencies: Optional[dict] = None) -> Tuple[ITranscriptionEngine, str]:
    """
    Фабричный метод для создания адаптера транскрипции.
    
    Определяет тип движка по формату модели и создает соответствующий адаптер.
    Если зависимости не указаны, использует get_transcription_dependencies() из app.main.
    
    Args:
        model: Название модели или "faster:model_name" для faster-whisper
        compute_type: Тип вычислений для faster-whisper ('int8', 'float16', 'float32')
        dependencies: Словарь зависимостей (если None, используется get_transcription_dependencies())
            - 'whisper_module': Модуль OpenAI Whisper
            - 'faster_whisper_model_class': Класс WhisperModel из faster_whisper
    
    Returns:
        Tuple[ITranscriptionEngine, str]: Адаптер и имя модели (без префикса "faster:")
    
    Example:
        # Использование с автоматическим разрешением зависимостей
        adapter, model_name = create_transcription_adapter(model='small', compute_type='int8')
        
        # Использование с явными зависимостями (для тестов)
        deps = {'whisper_module': mock_whisper, 'faster_whisper_model_class': MockClass}
        adapter, model_name = create_transcription_adapter(model='small', dependencies=deps)
    """
    if dependencies is None:
        from app.main import get_transcription_dependencies
        dependencies = get_transcription_dependencies()
    
    return _create_transcription_adapter_internal(
        model=model,
        whisper_module=dependencies['whisper_module'],
        faster_whisper_model_class=dependencies['faster_whisper_model_class'],
        compute_type=compute_type
    )


def create_transcription_service(engine: ITranscriptionEngine):
    """
    Фабричный метод для создания TranscriptionService.
    
    Args:
        engine: Адаптер движка транскрипции
    
    Returns:
        TranscriptionService: Сервис транскрипции
    """
    from app.application.services import TranscriptionService
    return TranscriptionService(engine=engine)

