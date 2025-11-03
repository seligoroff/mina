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



