import os
import click
import shutil
import requests
from app.factories import create_transcription_adapter, create_transcription_service
from app.utils.config import load_config
from app.adapters.output import FileOutputWriter
from app.application.services.word_analysis import WordAnalysisService


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

    # Создаем адаптер через фабричный метод (зависимости разрешаются автоматически)
    adapter, model_name = create_transcription_adapter(
        model=model,
        compute_type=compute_type
    )

    # Создаем сервис через фабричный метод
    service = create_transcription_service(engine=adapter)
    
    # Создаем адаптер для записи результатов
    output_writer = FileOutputWriter(output_path=output, verbose=True)

    # Выполняем транскрипцию через сервис
    try:
        list(service.transcribe(
            input_path=input,
            output_writer=output_writer,
            model_name=model_name,
            language=language,
            verbose=True,
            beam_size=5
        ))
        click.echo(f'Готово! Стенограмма сохранена в: {output}')
    except RuntimeError as e:
        raise click.ClickException(str(e))


@cli.command()
@click.option('--input', '-i', required=True, help='Путь к файлу с расшифровкой.')
@click.option('--output', '-o', required=False, help='Путь к выходному файлу (опционально).')
@click.option('--limit', '-l', default=50, show_default=True, help='Сколько слов вывести в итоговой статистике.')
@click.option('--lemmatize', is_flag=True, default=False, help='Включить лемматизацию (требуется pymorphy3).')
@click.option('--stopwords', required=False, type=click.Path(), help='Путь к файлу со стоп-словами (по одному слову на строку).')
@click.option('--no-names', is_flag=True, default=False, help='Исключать имена собственные (Name-граммема).')
def tag(input, output, limit, lemmatize, stopwords, no_names):
    """Генерация облака слов (частотный список) из текста расшифровки митапа."""
    
    service = WordAnalysisService()
    
    try:
        # Выполняем анализ через сервис
        word_frequency = service.analyze_word_frequency(
            file_path=input,
            stopwords=stopwords,
            lemmatize=lemmatize,
            no_names=no_names,
            limit=limit
        )
        
        # Форматируем результат
        result_text = service.format_result(word_frequency)
        
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
            
    except (FileNotFoundError, IOError) as e:
        raise click.ClickException(str(e))
    except ValueError as e:
        click.echo(f"Внимание: {e}", err=True)


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

