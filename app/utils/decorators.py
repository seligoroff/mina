"""Декораторы для валидации зависимостей."""

import shutil
from functools import wraps
from typing import Callable, Any


def require_ffmpeg(func: Callable) -> Callable:
    """Декоратор для проверки наличия ffmpeg перед выполнением функции.
    
    Проверяет наличие ffmpeg в системе перед вызовом декорируемой функции.
    Если ffmpeg не найден, выбрасывает RuntimeError с инструкцией по установке.
    
    Args:
        func: Функция или метод, который требует наличия ffmpeg
        
    Returns:
        Обернутая функция с проверкой ffmpeg
        
    Raises:
        RuntimeError: Если ffmpeg не найден в системе
        
    Example:
        @require_ffmpeg
        def transcribe(self, ...):
            # Бизнес-логика транскрипции
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg не найден. Установите его: sudo apt install ffmpeg"
            )
        return func(*args, **kwargs)
    return wrapper




