import re
import os
import click
import shutil
import whisper
import requests
from collections import Counter
from utils import write_transcript

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import pymorphy3
    PYMORPHY3_AVAILABLE = True
except ImportError:
    pymorphy3 = None
    PYMORPHY3_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

# Регулярки для tag
# Используем явный паттерн для кириллицы и латиницы
WORD_PATTERN = re.compile(r'\b[а-яА-ЯёЁa-zA-Z]{3,}\b', re.UNICODE)
TIMESTAMP_PATTERN = re.compile(r'^\[\d{1,3}\.\d{2}\s*-\s*\d{1,3}\.\d{2}\]')

# Части речи, которые считаем шумом
POS_TO_EXCLUDE = {'NPRO', 'ADVB', 'PRCL', 'CONJ', 'PREP', 'INTJ'}


def load_config(config_path):
    """Загружает конфигурацию из YAML файла."""
    if not YAML_AVAILABLE:
        raise click.ClickException("Для работы с конфигом требуется установить pyyaml: pip install pyyaml")
    
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


@click.group()
def cli():
    """Протокольный ассистент - утилиты для транскрипции и анализа аудио."""
    pass


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Путь к аудиофайлу')
@click.option('--output', '-o', required=True, type=click.Path(), help='Путь к выходному текстовому файлу')
@click.option('--model', '-m', default='small', show_default=True, 
              help='Название модели: tiny, base, small, medium, large. Для faster-whisper используйте формат "faster:model" (например, "faster:base")')
@click.option('--language', '--lang', default='ru', show_default=True, 
              help='Язык транскрипции (код ISO 639-1, например: ru, en, es, de)')
@click.option('--compute-type', default='int8', show_default=True, 
              help='Тип вычислений для faster-whisper (int8, float16, float32)')
def scribe(input, output, model, language, compute_type):
    """Распознавание речи с таймингами с помощью OpenAI Whisper или faster-whisper."""

    # Проверка наличия ffmpeg (нужен для обоих движков)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg не найден. Установите его: sudo apt install ffmpeg")

    # Определяем движок по формату модели
    if model.startswith('faster:'):
        # Используем faster-whisper
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper не установлен. Установите его: pip install faster-whisper")
        
        _, model_name = model.split(':', 1)  # Разделяем "faster:model_name"
        print(f"Загрузка модели faster-whisper: {model_name}")
        working_model = FasterWhisperModel(model_name, compute_type=compute_type)
        
        print(f"Распознавание: {input}")
        segments, _ = working_model.transcribe(input, beam_size=5, language=language)
        
        # faster-whisper возвращает генератор - выводим сегменты в реальном времени
        write_transcript(segments, output, verbose=True)
    else:
        # Используем оригинальный Whisper
        print(f"Загрузка модели Whisper: {model}")
        working_model = whisper.load_model(model)
        
        print(f"Распознавание: {input}")
        result = working_model.transcribe(input, language=language, verbose=True)
        
        write_transcript(result['segments'], output)

    print(f"Готово! Стенограмма сохранена в: {output}")


@cli.command()
@click.option('--input', '-i', required=True, help='Путь к файлу с расшифровкой.')
@click.option('--output', '-o', required=False, help='Путь к выходному файлу (опционально).')
@click.option('--limit', '-l', default=50, show_default=True, help='Сколько слов вывести в итоговой статистике.')
@click.option('--lemmatize', is_flag=True, default=False, help='Включить лемматизацию (требуется pymorphy3).')
@click.option('--stopwords', required=False, type=click.Path(), help='Путь к файлу со стоп-словами (по одному слову на строку).')
@click.option('--no-names', is_flag=True, default=False, help='Исключать имена собственные (Name-граммема).')
def tag(input, output, limit, lemmatize, stopwords, no_names):
    """Генерация облака слов (частотный список) из текста расшифровки митапа."""

    if lemmatize and not PYMORPHY3_AVAILABLE:
        raise RuntimeError("Для лемматизации требуется установить pymorphy3: pip install pymorphy3")

    morph = pymorphy3.MorphAnalyzer(lang='ru') if lemmatize else None

    # Загружаем стоп-слова (если указаны)
    stopword_set = set()
    if stopwords:
        with open(stopwords, 'r', encoding='utf-8') as f:
            stopword_set = {line.strip().lower() for line in f if line.strip()}

    # Загружаем текст
    try:
        with open(input, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise click.ClickException(f"Файл не найден: {input}")
    except Exception as e:
        raise click.ClickException(f"Ошибка при чтении файла {input}: {e}")

    # Извлекаем текст из строк, убирая таймкоды
    text_lines = []
    for line in lines:
        stripped = line.strip()
        # Проверяем, есть ли таймкод в начале строки
        match = TIMESTAMP_PATTERN.match(stripped)
        if match:
            # Извлекаем текст после таймкода
            text_after_timestamp = stripped[match.end():].strip()
            if text_after_timestamp:
                text_lines.append(text_after_timestamp)
        else:
            # Если таймкода нет, берем всю строку
            if stripped:
                text_lines.append(stripped)
    
    full_text = ' '.join(text_lines).lower()

    # Извлекаем слова (длиной >= 3)
    words = WORD_PATTERN.findall(full_text)
    
    if not words:
        click.echo(f"Внимание: в файле {input} не найдено слов (длиной >= 3 символов).", err=True)
        return

    # Лемматизация и грамматическая фильтрация
    if lemmatize:
        filtered = []
        for word in words:
            parsed = morph.parse(word)[0]
            lemma = parsed.normal_form

            if parsed.tag.POS in POS_TO_EXCLUDE:
                continue

            if no_names and 'Name' in parsed.tag.grammemes:
                continue

            filtered.append(lemma)
        words = filtered

    # Удаляем стоп-слова
    if stopword_set:
        words = [word for word in words if word not in stopword_set]

    # Частотный анализ
    word_counts = Counter(words)
    most_common = word_counts.most_common(limit)
    
    if not most_common:
        click.echo(f"Внимание: после фильтрации не осталось слов для анализа.", err=True)
        return
    
    output_lines = [f'{word}: {count}' for word, count in most_common]
    result_text = '\n'.join(output_lines)

    # Вывод
    if output:
        try:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(result_text)
            click.echo(f'Результат записан в файл: {output}')
        except Exception as e:
            raise click.ClickException(f"Ошибка при записи в файл {output}: {e}")
    else:
        click.echo(result_text)


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Путь к файлу с расшифровкой.')
@click.option('--output', '-o', default=None, type=click.Path(), help='Путь к выходному файлу (опционально, если не указан - вывод в консоль).')
@click.option('--config', '-c', default=None, type=click.Path(exists=True),
              help='Путь к файлу конфигурации (по умолчанию: config.yaml в директории скрипта).')
def protocol(input, output, config):
    """Создает структурированный протокол из расшифровки с помощью DeepSeek API."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Определяем путь к конфигу
    if config is None:
        config = os.path.join(script_dir, 'config.yaml')
        if not os.path.exists(config):
            raise click.ClickException(
                f"Файл конфигурации не найден: {config}\n"
                "Создайте config.yaml в директории скрипта или укажите путь через --config"
            )
    
    # Загружаем конфиг
    config_data = load_config(config)
    
    # Извлекаем API ключ, модель и инструкции из конфига
    if 'deepseek' not in config_data:
        raise click.ClickException(
            f"В конфиге {config} отсутствует секция 'deepseek'"
        )
    
    deepseek_config = config_data['deepseek']
    api_key = deepseek_config.get('api_key')
    model = deepseek_config.get('model', 'deepseek-chat')
    instructions_path = deepseek_config.get('instructions', 'deepseek-protocol-instructions.md')
    
    if not api_key:
        raise click.ClickException(
            f"В конфиге {config} отсутствует 'deepseek.api_key'"
        )

    # Определяем путь к файлу с инструкциями из конфига
    if not os.path.isabs(instructions_path):
        # Если путь относительный, берем относительно директории скрипта
        instructions = os.path.join(script_dir, instructions_path)
    else:
        instructions = instructions_path
    
    if not os.path.exists(instructions):
        raise click.ClickException(
            f"Файл с инструкциями не найден: {instructions}\n"
            "Укажите путь к файлу с инструкциями в конфиге"
        )

    # Читаем инструкции
    try:
        with open(instructions, 'r', encoding='utf-8') as f:
            instructions_text = f.read()
    except Exception as e:
        raise click.ClickException(f"Ошибка при чтении файла с инструкциями {instructions}: {e}")

    # Читаем расшифровку
    try:
        with open(input, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
    except Exception as e:
        raise click.ClickException(f"Ошибка при чтении файла с расшифровкой {input}: {e}")

    # Формируем промпт
    prompt = f"""{instructions_text}

**Расшифровка:**

{transcript_text}
"""

    # Отправляем запрос в DeepSeek API
    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    }

    click.echo("Отправка запроса в DeepSeek API...", err=True)

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        
        # Извлекаем текст ответа
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            
            # Вывод результата
            if output:
                try:
                    with open(output, 'w', encoding='utf-8') as f:
                        f.write(content)
                    click.echo(f'Протокол сохранен в файл: {output}')
                except Exception as e:
                    raise click.ClickException(f"Ошибка при записи в файл {output}: {e}")
            else:
                click.echo(content)
        else:
            raise click.ClickException(f"Неожиданный формат ответа от API: {result}")
            
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                raise click.ClickException(f"Ошибка API DeepSeek: {error_detail}")
            except:
                raise click.ClickException(f"Ошибка API DeepSeek: {e.response.text}")
        else:
            raise click.ClickException(f"Ошибка при отправке запроса: {e}")


if __name__ == '__main__':
    cli()

