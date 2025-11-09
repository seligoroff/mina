import pytest

from app.adapters.output.whisper import FasterWhisperAdapter, WhisperAdapter
from app.factories.transcription_factory import (
    _create_transcription_adapter_internal,
    create_transcription_adapter,
)


@pytest.mark.integration
def test_internal_method_behavior_matches_public_api():
    """Сравниваем _create_transcription_adapter_internal с публичной фабрикой."""
    from app.main import get_transcription_dependencies

    deps = get_transcription_dependencies()

    public_adapter, public_model_name = create_transcription_adapter(
        "faster:base",
        compute_type="float16",
    )

    internal_adapter, internal_model_name = _create_transcription_adapter_internal(
        model="faster:base",
        whisper_module=deps["whisper_module"],
        faster_whisper_model_class=deps["faster_whisper_model_class"],
        compute_type="float16",
    )

    assert type(public_adapter) == type(internal_adapter)
    assert public_model_name == internal_model_name
    assert isinstance(public_adapter, FasterWhisperAdapter)
    assert isinstance(internal_adapter, FasterWhisperAdapter)
    assert public_adapter._compute_type == internal_adapter._compute_type


@pytest.mark.integration
def test_internal_method_with_real_dependencies():
    """Проверяем, что внутренний метод корректно работает с реальными зависимостями."""
    from app.main import get_transcription_dependencies

    deps = get_transcription_dependencies()

    adapter, model_name = _create_transcription_adapter_internal(
        model="small",
        whisper_module=deps["whisper_module"],
        faster_whisper_model_class=deps["faster_whisper_model_class"],
        compute_type="int8",
    )

    assert isinstance(adapter, WhisperAdapter)
    assert model_name == "small"
    assert adapter._whisper is deps["whisper_module"]

    adapter2, model_name2 = _create_transcription_adapter_internal(
        model="faster:base",
        whisper_module=deps["whisper_module"],
        faster_whisper_model_class=deps["faster_whisper_model_class"],
        compute_type="int8",
    )

    assert isinstance(adapter2, FasterWhisperAdapter)
    assert model_name2 == "base"
    assert adapter2._faster_whisper_model_class is deps["faster_whisper_model_class"]
    assert adapter2._compute_type == "int8"

