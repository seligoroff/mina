"""Временный скрипт для проверки фабрик."""
from app.main import create_transcription_adapter, create_transcription_service
import whisper

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_AVAILABLE = True
except ImportError:
    FASTER_AVAILABLE = False

# Тест создания Whisper адаптера
adapter, model = create_transcription_adapter(
    'base', 
    whisper_module=whisper,
    faster_whisper_available=False
)
service = create_transcription_service(adapter)
print('Whisper factory OK')

if FASTER_AVAILABLE:
    adapter2, model2 = create_transcription_adapter(
        'faster:base',
        whisper_module=whisper,
        faster_whisper_model_class=FasterWhisperModel,
        faster_whisper_available=True
    )
    service2 = create_transcription_service(adapter2)
    print('FasterWhisper factory OK')

print('All factory methods OK')



