"""Точка входа приложения - инициализация адаптеров и сервисов.

Этот модуль отвечает за создание и инициализацию компонентов приложения
согласно гексагональной архитектуре: адаптеры, сервисы, фабрики.

Примечание: Фабричные методы находятся в app.factories.*
"""


def get_transcription_dependencies():
    """Возвращает зависимости для транскрипции.
    
    Централизует импорт и создание зависимостей для транскрипции.
    Это позволяет:
    - Изолировать CLI от конкретных библиотек
    - Легко менять реализации в одном месте
    - Легко тестировать с мокированными зависимостями
    
    Returns:
        dict: Словарь с зависимостями для транскрипции:
            - 'whisper_module': Модуль OpenAI Whisper
            - 'faster_whisper_model_class': Класс WhisperModel из faster_whisper
    """
    import whisper
    from faster_whisper import WhisperModel as FasterWhisperModel
    
    return {
        'whisper_module': whisper,
        'faster_whisper_model_class': FasterWhisperModel,
    }


def create_app():
    """
    Создает и настраивает приложение.
    
    Returns:
        dict: Словарь с инициализированными компонентами приложения
              (адаптеры, сервисы и т.д.)
    """
    # TODO: Будет реализовано при рефакторинге
    # Здесь будет создание адаптеров и сервисов
    pass


# Примечание: Фабричные методы находятся в app.factories.*
# - app.factories.transcription_factory: create_transcription_adapter(), create_transcription_service()
# - Будущие фабрики будут добавлены в app.factories.*

