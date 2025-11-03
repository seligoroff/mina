"""Тесты для app.main."""

import pytest
from app.main import get_transcription_dependencies


@pytest.mark.unit
class TestGetTranscriptionDependencies:
    """Тесты для get_transcription_dependencies()."""
    
    def test_get_transcription_dependencies_structure(self):
        """Тест структуры возвращаемого словаря зависимостей."""
        deps = get_transcription_dependencies()
        
        assert isinstance(deps, dict)
        assert 'whisper_module' in deps
        assert 'faster_whisper_model_class' in deps
        
        # Проверяем типы (не можем проверить точные классы без импорта)
        assert deps['whisper_module'] is not None
        assert deps['faster_whisper_model_class'] is not None
    
    def test_get_transcription_dependencies_imports(self):
        """Тест импорта зависимостей."""
        deps = get_transcription_dependencies()
        
        # Проверяем, что модуль whisper импортирован
        assert hasattr(deps['whisper_module'], 'load_model')
        
        # Проверяем, что класс FasterWhisperModel импортирован
        assert deps['faster_whisper_model_class'] is not None
    
    def test_get_transcription_dependencies_whisper_module(self):
        """Тест структуры модуля whisper."""
        deps = get_transcription_dependencies()
        whisper_module = deps['whisper_module']
        
        # Проверяем наличие метода load_model
        assert hasattr(whisper_module, 'load_model')
        assert callable(whisper_module.load_model)
    
    def test_get_transcription_dependencies_faster_whisper_class(self):
        """Тест класса FasterWhisperModel."""
        deps = get_transcription_dependencies()
        faster_whisper_model_class = deps['faster_whisper_model_class']
        
        # Проверяем, что это класс
        assert faster_whisper_model_class is not None
        # Проверяем, что это вызываемый объект (класс)
        assert callable(faster_whisper_model_class)
    
    def test_get_transcription_dependencies_no_exceptions(self):
        """Тест отсутствия исключений при вызове."""
        # Функция не должна выбрасывать исключения
        try:
            deps = get_transcription_dependencies()
            assert deps is not None
        except Exception as e:
            pytest.fail(f"get_transcription_dependencies() raised {type(e).__name__}: {e}")
    
    def test_get_transcription_dependencies_returns_same_instance(self):
        """Тест, что функция возвращает одинаковые зависимости при повторном вызове."""
        deps1 = get_transcription_dependencies()
        deps2 = get_transcription_dependencies()
        
        # Проверяем, что это те же объекты (модули)
        assert deps1['whisper_module'] is deps2['whisper_module']
        assert deps1['faster_whisper_model_class'] is deps2['faster_whisper_model_class']
    
    def test_get_transcription_dependencies_dict_keys(self):
        """Тест правильности ключей словаря."""
        deps = get_transcription_dependencies()
        
        # Проверяем, что словарь содержит только ожидаемые ключи
        expected_keys = {'whisper_module', 'faster_whisper_model_class'}
        actual_keys = set(deps.keys())
        
        assert actual_keys == expected_keys



