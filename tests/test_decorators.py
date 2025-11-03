"""Тесты для декораторов."""

import pytest
from unittest.mock import patch, Mock
from app.utils.decorators import require_ffmpeg


@pytest.mark.unit
class TestRequireFfmpeg:
    """Тесты для декоратора require_ffmpeg."""
    
    def test_require_ffmpeg_when_ffmpeg_available(self):
        """Тест декоратора, когда ffmpeg доступен."""
        @require_ffmpeg
        def test_func():
            return "success"
        
        # Мокируем shutil.which для возврата пути к ffmpeg
        with patch('app.utils.decorators.shutil.which', return_value='/usr/bin/ffmpeg'):
            result = test_func()
            assert result == "success"
    
    def test_require_ffmpeg_when_ffmpeg_not_available(self):
        """Тест декоратора, когда ffmpeg недоступен."""
        @require_ffmpeg
        def test_func():
            return "success"
        
        # Мокируем shutil.which для возврата None
        with patch('app.utils.decorators.shutil.which', return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg не найден"):
                test_func()
    
    def test_require_ffmpeg_preserves_function_metadata(self):
        """Тест сохранения метаданных функции."""
        @require_ffmpeg
        def test_func(param1, param2=None):
            """Тестовая функция."""
            return f"{param1}:{param2}"
        
        # Проверяем сохранение имени и документации
        assert test_func.__name__ == "test_func"
        assert "Тестовая функция" in test_func.__doc__
    
    def test_require_ffmpeg_passes_arguments(self):
        """Тест передачи аргументов в декорируемую функцию."""
        @require_ffmpeg
        def test_func(a, b, c=10):
            return a + b + c
        
        # Мокируем shutil.which для возврата пути к ffmpeg
        with patch('app.utils.decorators.shutil.which', return_value='/usr/bin/ffmpeg'):
            result = test_func(1, 2, c=3)
            assert result == 6
    
    def test_require_ffmpeg_passes_kwargs(self):
        """Тест передачи kwargs в декорируемую функцию."""
        @require_ffmpeg
        def test_func(**kwargs):
            return kwargs
        
        # Мокируем shutil.which для возврата пути к ffmpeg
        with patch('app.utils.decorators.shutil.which', return_value='/usr/bin/ffmpeg'):
            result = test_func(x=1, y=2, z=3)
            assert result == {'x': 1, 'y': 2, 'z': 3}
    
    def test_require_ffmpeg_with_method(self):
        """Тест декоратора на методе класса."""
        class TestClass:
            @require_ffmpeg
            def test_method(self, value):
                return value * 2
        
        obj = TestClass()
        
        # Мокируем shutil.which для возврата пути к ffmpeg
        with patch('app.utils.decorators.shutil.which', return_value='/usr/bin/ffmpeg'):
            result = obj.test_method(5)
            assert result == 10
    
    def test_require_ffmpeg_error_message(self):
        """Тест сообщения об ошибке при отсутствии ffmpeg."""
        @require_ffmpeg
        def test_func():
            pass
        
        # Мокируем shutil.which для возврата None
        with patch('app.utils.decorators.shutil.which', return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                test_func()
            
            assert "ffmpeg не найден" in str(exc_info.value)
            assert "sudo apt install ffmpeg" in str(exc_info.value)



