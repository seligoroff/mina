import click
from app.adapters.input.cli import (
    ScribeCommandHandler,
    ScribeCommandOptions,
    ProtocolCommandHandler,
    ProtocolCommandOptions,
    TagCommandHandler,
    TagCommandOptions,
)
from app.domain.exceptions import ProtocolClientError


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
    handler = ScribeCommandHandler()
    options = ScribeCommandOptions(
        input_path=input,
        output_path=output,
        model=model,
        language=language,
        compute_type=compute_type,
        verbose=True,
    )
    try:
        handler.execute(options)
        click.echo(f"Готово! Стенограмма сохранена в: {output}")
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
    handler = TagCommandHandler()
    options = TagCommandOptions(
        transcript_path=input,
        output_path=output,
        limit=limit,
        lemmatize=lemmatize,
        stopwords_path=stopwords,
        exclude_names=no_names,
    )
    try:
        handler.execute(options)
        if output:
            click.echo(f"Результат записан в файл: {output}")
    except (FileNotFoundError, IOError) as e:
        raise click.ClickException(str(e))
    except ValueError as e:
        click.echo(f"Внимание: {e}", err=True)


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Путь к файлу с расшифровкой.')
@click.option('--output', '-o', default=None, type=click.Path(), help='Путь к выходному файлу (опционально, если не указан - вывод в консоль).')
@click.option('--config', '-c', default=None, type=click.Path(),
              help='Путь к файлу конфигурации (по умолчанию: config.yaml в директории проекта).')
def protocol(input, output, config):
    """Создает структурированный протокол из расшифровки."""
    handler = ProtocolCommandHandler(
        output_writer=_write_protocol_output
    )
    options = ProtocolCommandOptions(
        transcript_path=input,
        output_path=output,
        config_path=config,
    )
    try:
        handler.execute(options)
        if output:
            click.echo(f"Протокол сохранен в файл: {output}")
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    except ProtocolClientError as e:
        raise click.ClickException(str(e))
    except OSError as e:
        raise click.ClickException(f"Ошибка при записи файла: {e}")


def _write_protocol_output(target_path: str, content: str) -> None:
    """Записывает протокол в файл или выводит в консоль."""
    if target_path:
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            raise exc
    else:
        click.echo(content)


if __name__ == '__main__':
    cli()

