"""Временный скрипт для проверки фабрик."""

from app.factories.transcription_factory import (
    create_transcription_adapter,
    create_transcription_service,
)
import whisper

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_AVAILABLE = True
except ImportError:
    FASTER_AVAILABLE = False

dependencies = {
    "whisper_module": whisper,
    "faster_whisper_model_class": FasterWhisperModel if FASTER_AVAILABLE else None,
}

# Тест создания Whisper адаптера
adapter, model = create_transcription_adapter(
    "base",
    dependencies=dependencies,
)
service = create_transcription_service(adapter)
print("Whisper factory OK")

if FASTER_AVAILABLE:
    adapter2, model2 = create_transcription_adapter(
        "faster:base",
        dependencies=dependencies,
    )
    service2 = create_transcription_service(adapter2)
    print("FasterWhisper factory OK")

print("All factory methods OK")




