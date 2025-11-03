"""Тесты для фабрик транскрипции."""

import pytest
from unittest.mock import Mock, patch
from app.factories.transcription_factory import (
    create_transcription_adapter,
    create_transcription_service,
    _create_transcription_adapter_internal,
)
from app.adapters.output.whisper import WhisperAdapter, FasterWhisperAdapter
from app.application.ports import ITranscriptionEngine
from app.application.services import TranscriptionService


@pytest.mark.unit
class TestCreateTranscriptionAdapter:
    """Тесты для create_transcription_adapter()."""
    
    def test_create_transcription_adapter_uses_default_dependencies(self):
        """Тест автоматического разрешения зависимостей."""
        adapter, model_name = create_transcription_adapter('small')
        
        assert model_name == 'small'
        assert isinstance(adapter, WhisperAdapter)
    
    def test_create_transcription_adapter_with_explicit_dependencies(self):
        """Тест создания адаптера с явными зависимостями."""
        mock_whisper = Mock()
        mock_faster_whisper = Mock()
        
        deps = {
            'whisper_module': mock_whisper,
            'faster_whisper_model_class': mock_faster_whisper,
        }
        
        adapter, model_name = create_transcription_adapter(
            'small',
            dependencies=deps
        )
        
        assert isinstance(adapter, WhisperAdapter)
        assert adapter._whisper is mock_whisper
    
    def test_create_transcription_adapter_faster_whisper(self):
        """Тест создания адаптера для faster-whisper."""
        adapter, model_name = create_transcription_adapter('faster:base')
        
        assert model_name == 'base'
        assert isinstance(adapter, FasterWhisperAdapter)
    
    def test_create_transcription_adapter_compute_type(self):
        """Тест передачи compute_type в адаптер."""
        adapter, _ = create_transcription_adapter(
            'faster:base',
            compute_type='float16'
        )
        
        assert isinstance(adapter, FasterWhisperAdapter)
        assert adapter._compute_type == 'float16'
    
    def test_create_transcription_adapter_default_compute_type(self):
        """Тест значения по умолчанию для compute_type."""
        adapter, _ = create_transcription_adapter('faster:base')
        
        assert isinstance(adapter, FasterWhisperAdapter)
        assert adapter._compute_type == 'int8'
    
    def test_create_transcription_adapter_different_compute_types(self):
        """Тест различных значений compute_type."""
        compute_types = ['int8', 'float16', 'float32']
        
        for compute_type in compute_types:
            adapter, _ = create_transcription_adapter(
                'faster:base',
                compute_type=compute_type
            )
            
            assert isinstance(adapter, FasterWhisperAdapter)
            assert adapter._compute_type == compute_type
    
    def test_create_transcription_adapter_whisper_models(self):
        """Тест создания адаптера для различных моделей Whisper."""
        models = ['tiny', 'base', 'small', 'medium', 'large']
        
        for model in models:
            adapter, model_name = create_transcription_adapter(model)
            
            assert model_name == model
            assert isinstance(adapter, WhisperAdapter)
    
    def test_create_transcription_adapter_faster_whisper_models(self):
        """Тест создания адаптера для различных моделей faster-whisper."""
        models = ['tiny', 'base', 'small', 'medium', 'large']
        
        for model in models:
            faster_model = f'faster:{model}'
            adapter, model_name = create_transcription_adapter(faster_model)
            
            assert model_name == model
            assert isinstance(adapter, FasterWhisperAdapter)
    
    def test_create_transcription_adapter_extracts_model_name(self):
        """Тест извлечения имени модели из 'faster:model_name'."""
        test_cases = [
            ('faster:base', 'base'),
            ('faster:small', 'small'),
            ('faster:medium', 'medium'),
            ('faster:large-v2', 'large-v2'),
        ]
        
        for input_model, expected_name in test_cases:
            adapter, model_name = create_transcription_adapter(input_model)
            
            assert model_name == expected_name
            assert isinstance(adapter, FasterWhisperAdapter)
    
    def test_create_transcription_adapter_calls_get_transcription_dependencies(self):
        """Тест вызова get_transcription_dependencies() при отсутствии dependencies."""
        with patch('app.main.get_transcription_dependencies') as mock_get_deps:
            mock_deps = {
                'whisper_module': Mock(),
                'faster_whisper_model_class': Mock(),
            }
            mock_get_deps.return_value = mock_deps
            
            create_transcription_adapter('small')
            
            # Проверяем, что функция была вызвана
            mock_get_deps.assert_called_once()
    
    def test_create_transcription_adapter_with_explicit_deps_does_not_call_get_deps(self):
        """Тест, что при явных зависимостях get_transcription_dependencies() не вызывается."""
        mock_deps = {
            'whisper_module': Mock(),
            'faster_whisper_model_class': Mock(),
        }
        
        with patch('app.main.get_transcription_dependencies') as mock_get_deps:
            create_transcription_adapter('small', dependencies=mock_deps)
            
            # Проверяем, что функция не была вызвана
            mock_get_deps.assert_not_called()
    
    def test_create_transcription_adapter_returns_correct_type(self):
        """Тест возврата правильного типа адаптера."""
        whisper_adapter, _ = create_transcription_adapter('small')
        faster_adapter, _ = create_transcription_adapter('faster:small')
        
        assert isinstance(whisper_adapter, ITranscriptionEngine)
        assert isinstance(faster_adapter, ITranscriptionEngine)
    
    def test_create_transcription_adapter_model_name_without_prefix(self):
        """Тест обработки модели без префикса 'faster:'."""
        adapter, model_name = create_transcription_adapter('base')
        
        assert model_name == 'base'
        assert isinstance(adapter, WhisperAdapter)
    
    def test_create_transcription_adapter_with_mock_dependencies(self):
        """Тест создания адаптера с мокированными зависимостями."""
        mock_whisper_module = Mock()
        mock_faster_whisper_class = Mock()
        
        deps = {
            'whisper_module': mock_whisper_module,
            'faster_whisper_model_class': mock_faster_whisper_class,
        }
        
        # Тест для WhisperAdapter
        adapter, model_name = create_transcription_adapter(
            'small',
            dependencies=deps
        )
        
        assert isinstance(adapter, WhisperAdapter)
        assert adapter._whisper is mock_whisper_module
        
        # Тест для FasterWhisperAdapter
        adapter2, model_name2 = create_transcription_adapter(
            'faster:base',
            compute_type='float16',
            dependencies=deps
        )
        
        assert isinstance(adapter2, FasterWhisperAdapter)
        assert adapter2._faster_whisper_model_class is mock_faster_whisper_class
        assert adapter2._compute_type == 'float16'


@pytest.mark.unit
class TestCreateTranscriptionService:
    """Тесты для create_transcription_service()."""
    
    def test_create_transcription_service_with_whisper_adapter(self):
        """Тест создания сервиса с WhisperAdapter."""
        adapter, _ = create_transcription_adapter('small')
        service = create_transcription_service(adapter)
        
        assert isinstance(service, TranscriptionService)
        assert service._engine is adapter
    
    def test_create_transcription_service_with_faster_adapter(self):
        """Тест создания сервиса с FasterWhisperAdapter."""
        adapter, _ = create_transcription_adapter('faster:base')
        service = create_transcription_service(adapter)
        
        assert isinstance(service, TranscriptionService)
        assert service._engine is adapter
    
    def test_create_transcription_service_with_mock_engine(self):
        """Тест создания сервиса с мокированным engine."""
        mock_engine = Mock(spec=ITranscriptionEngine)
        service = create_transcription_service(mock_engine)
        
        assert isinstance(service, TranscriptionService)
        assert service._engine is mock_engine


@pytest.mark.unit
class TestCreateTranscriptionAdapterInternal:
    """Тесты для внутреннего метода _create_transcription_adapter_internal()."""
    
    def test_create_transcription_adapter_internal_whisper(self):
        """Тест создания WhisperAdapter через внутренний метод."""
        mock_whisper_module = Mock()
        mock_faster_whisper_class = Mock()
        
        adapter, model_name = _create_transcription_adapter_internal(
            model='small',
            whisper_module=mock_whisper_module,
            faster_whisper_model_class=mock_faster_whisper_class,
            compute_type='int8'
        )
        
        assert isinstance(adapter, WhisperAdapter)
        assert adapter._whisper is mock_whisper_module
        assert model_name == 'small'
    
    def test_create_transcription_adapter_internal_faster_whisper(self):
        """Тест создания FasterWhisperAdapter через внутренний метод."""
        mock_whisper_module = Mock()
        mock_faster_whisper_class = Mock()
        
        adapter, model_name = _create_transcription_adapter_internal(
            model='faster:base',
            whisper_module=mock_whisper_module,
            faster_whisper_model_class=mock_faster_whisper_class,
            compute_type='float16'
        )
        
        assert isinstance(adapter, FasterWhisperAdapter)
        assert adapter._faster_whisper_model_class is mock_faster_whisper_class
        assert adapter._compute_type == 'float16'
        assert model_name == 'base'
    
    def test_create_transcription_adapter_internal_extracts_model_name(self):
        """Тест извлечения имени модели из 'faster:model_name'."""
        mock_whisper_module = Mock()
        mock_faster_whisper_class = Mock()
        
        test_cases = [
            ('faster:tiny', 'tiny'),
            ('faster:base', 'base'),
            ('faster:small', 'small'),
            ('faster:medium', 'medium'),
            ('faster:large-v2', 'large-v2'),
        ]
        
        for input_model, expected_name in test_cases:
            adapter, model_name = _create_transcription_adapter_internal(
                model=input_model,
                whisper_module=mock_whisper_module,
                faster_whisper_model_class=mock_faster_whisper_class,
                compute_type='int8'
            )
            
            assert model_name == expected_name
            assert isinstance(adapter, FasterWhisperAdapter)
    
    def test_create_transcription_adapter_internal_preserves_model_name(self):
        """Тест сохранения имени модели без префикса 'faster:'."""
        mock_whisper_module = Mock()
        mock_faster_whisper_class = Mock()
        
        test_cases = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2']
        
        for model in test_cases:
            adapter, model_name = _create_transcription_adapter_internal(
                model=model,
                whisper_module=mock_whisper_module,
                faster_whisper_model_class=mock_faster_whisper_class,
                compute_type='int8'
            )
            
            assert model_name == model
            assert isinstance(adapter, WhisperAdapter)
    
    def test_create_transcription_adapter_internal_compute_type(self):
        """Тест передачи compute_type в FasterWhisperAdapter."""
        mock_whisper_module = Mock()
        mock_faster_whisper_class = Mock()
        
        compute_types = ['int8', 'float16', 'float32']
        
        for compute_type in compute_types:
            adapter, _ = _create_transcription_adapter_internal(
                model='faster:base',
                whisper_module=mock_whisper_module,
                faster_whisper_model_class=mock_faster_whisper_class,
                compute_type=compute_type
            )
            
            assert isinstance(adapter, FasterWhisperAdapter)
            assert adapter._compute_type == compute_type
    
    def test_create_transcription_adapter_internal_default_compute_type(self):
        """Тест значения по умолчанию для compute_type."""
        mock_whisper_module = Mock()
        mock_faster_whisper_class = Mock()
        
        adapter, _ = _create_transcription_adapter_internal(
            model='faster:base',
            whisper_module=mock_whisper_module,
            faster_whisper_model_class=mock_faster_whisper_class
        )
        
        assert isinstance(adapter, FasterWhisperAdapter)
        assert adapter._compute_type == 'int8'
    
    def test_create_transcription_adapter_internal_returns_correct_types(self):
        """Тест возврата правильных типов адаптеров."""
        mock_whisper_module = Mock()
        mock_faster_whisper_class = Mock()
        
        whisper_adapter, _ = _create_transcription_adapter_internal(
            model='small',
            whisper_module=mock_whisper_module,
            faster_whisper_model_class=mock_faster_whisper_class,
            compute_type='int8'
        )
        
        faster_adapter, _ = _create_transcription_adapter_internal(
            model='faster:small',
            whisper_module=mock_whisper_module,
            faster_whisper_model_class=mock_faster_whisper_class,
            compute_type='int8'
        )
        
        assert isinstance(whisper_adapter, ITranscriptionEngine)
        assert isinstance(faster_adapter, ITranscriptionEngine)
        assert isinstance(whisper_adapter, WhisperAdapter)
        assert isinstance(faster_adapter, FasterWhisperAdapter)
    
    def test_create_transcription_adapter_internal_whisper_module_not_used_for_faster(self):
        """Тест, что whisper_module не используется для faster-whisper."""
        mock_whisper_module = Mock()
        mock_faster_whisper_class = Mock()
        
        adapter, _ = _create_transcription_adapter_internal(
            model='faster:base',
            whisper_module=mock_whisper_module,
            faster_whisper_model_class=mock_faster_whisper_class,
            compute_type='int8'
        )
        
        # Проверяем, что whisper_module не был использован
        assert isinstance(adapter, FasterWhisperAdapter)
        assert not hasattr(adapter, '_whisper')
        assert hasattr(adapter, '_faster_whisper_model_class')
    
    def test_create_transcription_adapter_internal_faster_class_not_used_for_whisper(self):
        """Тест, что faster_whisper_model_class не используется для Whisper."""
        mock_whisper_module = Mock()
        mock_faster_whisper_class = Mock()
        
        adapter, _ = _create_transcription_adapter_internal(
            model='small',
            whisper_module=mock_whisper_module,
            faster_whisper_model_class=mock_faster_whisper_class,
            compute_type='int8'
        )
        
        # Проверяем, что faster_whisper_model_class не был использован
        assert isinstance(adapter, WhisperAdapter)
        assert adapter._whisper is mock_whisper_module
        assert not hasattr(adapter, '_faster_whisper_model_class')
    
    def test_create_transcription_adapter_internal_edge_case_faster_colon_only(self):
        """Тест обработки граничного случая 'faster:'."""
        mock_whisper_module = Mock()
        mock_faster_whisper_class = Mock()
        
        adapter, model_name = _create_transcription_adapter_internal(
            model='faster:',
            whisper_module=mock_whisper_module,
            faster_whisper_model_class=mock_faster_whisper_class,
            compute_type='int8'
        )
        
        assert isinstance(adapter, FasterWhisperAdapter)
        assert model_name == ''  # Пустое имя после 'faster:'
    
    def test_create_transcription_adapter_internal_complex_model_names(self):
        """Тест обработки сложных имен моделей."""
        mock_whisper_module = Mock()
        mock_faster_whisper_class = Mock()
        
        complex_models = [
            'faster:base',
            'faster:small',
            'faster:medium',
            'faster:large-v2',
            'base',  # Без префикса
            'small',
        ]
        
        for model in complex_models:
            adapter, model_name = _create_transcription_adapter_internal(
                model=model,
                whisper_module=mock_whisper_module,
                faster_whisper_model_class=mock_faster_whisper_class,
                compute_type='int8'
            )
            
            assert isinstance(adapter, ITranscriptionEngine)
            
            if model.startswith('faster:'):
                assert isinstance(adapter, FasterWhisperAdapter)
                expected_name = model.split(':', 1)[1]
                assert model_name == expected_name
            else:
                assert isinstance(adapter, WhisperAdapter)
                assert model_name == model


@pytest.mark.integration
class TestCreateTranscriptionAdapterIntegration:
    """Интеграционные тесты для _create_transcription_adapter_internal()."""
    
    def test_internal_method_behavior_matches_public_api(self):
        """Тест, что внутренний метод работает идентично публичному API."""
        from app.main import get_transcription_dependencies
        
        deps = get_transcription_dependencies()
        
        # Создаем через публичный API
        public_adapter, public_model_name = create_transcription_adapter(
            'faster:base',
            compute_type='float16'
        )
        
        # Создаем через внутренний метод
        internal_adapter, internal_model_name = _create_transcription_adapter_internal(
            model='faster:base',
            whisper_module=deps['whisper_module'],
            faster_whisper_model_class=deps['faster_whisper_model_class'],
            compute_type='float16'
        )
        
        # Проверяем, что результаты идентичны
        assert type(public_adapter) == type(internal_adapter)
        assert public_model_name == internal_model_name
        assert isinstance(public_adapter, FasterWhisperAdapter)
        assert isinstance(internal_adapter, FasterWhisperAdapter)
        assert public_adapter._compute_type == internal_adapter._compute_type
    
    def test_internal_method_with_real_dependencies(self):
        """Тест внутреннего метода с реальными зависимостями."""
        from app.main import get_transcription_dependencies
        
        deps = get_transcription_dependencies()
        
        # Тест для Whisper
        adapter, model_name = _create_transcription_adapter_internal(
            model='small',
            whisper_module=deps['whisper_module'],
            faster_whisper_model_class=deps['faster_whisper_model_class'],
            compute_type='int8'
        )
        
        assert isinstance(adapter, WhisperAdapter)
        assert model_name == 'small'
        assert adapter._whisper is deps['whisper_module']
        
        # Тест для faster-whisper
        adapter2, model_name2 = _create_transcription_adapter_internal(
            model='faster:base',
            whisper_module=deps['whisper_module'],
            faster_whisper_model_class=deps['faster_whisper_model_class'],
            compute_type='int8'
        )
        
        assert isinstance(adapter2, FasterWhisperAdapter)
        assert model_name2 == 'base'
        assert adapter2._faster_whisper_model_class is deps['faster_whisper_model_class']
        assert adapter2._compute_type == 'int8'

