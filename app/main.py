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
    import whisper  # type: ignore[import]
    from faster_whisper import WhisperModel as FasterWhisperModel  # type: ignore[import]
    
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
    import pymorphy3  # type: ignore[import]
    from app.adapters.input.cli import (
        ScribeCommandHandler,
        ProtocolCommandHandler,
        TagCommandHandler,
    )
    from app.factories import (
        create_transcription_adapter,
        create_transcription_service,
        create_protocol_client,
        create_protocol_service,
        create_word_analysis_service,
    )

    morph_analyzer = pymorphy3.MorphAnalyzer(lang="ru")

    handlers = {
        "scribe": ScribeCommandHandler(),
        "protocol": ProtocolCommandHandler(),
        "tag": TagCommandHandler(
            analysis_service_factory=lambda: create_word_analysis_service(
                dependencies={"morph": morph_analyzer}
            )
        ),
    }

    services = {
        "word_analysis": create_word_analysis_service(
            dependencies={"morph": morph_analyzer}
        ),
    }

    factories = {
        "transcription": create_transcription_adapter,
        "transcription_service": create_transcription_service,
        "protocol_client": create_protocol_client,
        "protocol_service": create_protocol_service,
        "word_analysis_service": create_word_analysis_service,
    }

    return {
        "handlers": handlers,
        "services": services,
        "factories": factories,
    }


# Примечание: Фабричные методы находятся в app.factories.*
# - app.factories.transcription_factory: create_transcription_adapter(), create_transcription_service()
# - app.factories.protocol_factory: create_protocol_client(), create_protocol_service()
# - app.factories.tag_factory: create_word_analysis_service()

