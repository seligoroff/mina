"""Утилиты для работы с конфигурацией."""

import yaml
import click


def load_config(config_path: str) -> dict:
    """Загружает конфигурацию из YAML файла.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        dict: Загруженная конфигурация
        
    Raises:
        click.ClickException: Если файл не найден или произошла ошибка при чтении
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise click.ClickException(f"Файл конфигурации не найден: {config_path}")
    except yaml.YAMLError as e:
        raise click.ClickException(f"Ошибка при чтении YAML конфига {config_path}: {e}")
    except Exception as e:
        raise click.ClickException(f"Ошибка при загрузке конфига {config_path}: {e}")


def get_telegram_config(raw_config: dict) -> dict:
    """Возвращает и валидирует секцию telegram из конфига.

    Требуемые поля: api_id, api_hash, session_path.
    Не падает, если поле peer отсутствует (оно может задаваться через CLI).
    """
    telegram = (raw_config or {}).get("telegram") or {}

    api_id = telegram.get("api_id")
    api_hash = telegram.get("api_hash")
    session_path = telegram.get("session_path")

    missing = [name for name, val in (("api_id", api_id), ("api_hash", api_hash), ("session_path", session_path)) if not val]
    if missing:
        raise click.ClickException(
            "Неполная конфигурация telegram: отсутствуют поля "
            + ", ".join(missing)
            + ". Заполните их в config.yaml или передайте через переменные окружения."
        )

    # Нормализуем api_id (может приходить со знаком ':' в начале)
    try:
        telegram["api_id"] = int(str(api_id).lstrip(":").strip())
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(f"Некорректное значение telegram.api_id: {api_id!r}") from e

    return telegram



