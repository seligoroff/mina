"""Общие утилиты для транскрипции."""


def write_transcript(segments, output_path, verbose=False):
    """
    Записывает транскрипцию сегментов в файл с таймингами.
    
    Поддерживает два формата сегментов:
    - faster-whisper: объекты с атрибутами (segment.start, segment.end, segment.text)
    - openai-whisper: словари с ключами (segment['start'], segment['end'], segment['text'])
    
    Args:
        segments: Итератор или список сегментов (объекты или словари)
        output_path: Путь к выходному файлу
        verbose: Если True, выводит сегменты на экран в реальном времени (для генераторов)
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            # Поддержка обоих форматов: объект с атрибутами или словарь
            if hasattr(segment, 'start'):
                # faster-whisper формат (объект)
                start = segment.start
                end = segment.end
                text = segment.text.strip()
            elif isinstance(segment, dict):
                # openai-whisper формат (словарь)
                start = segment['start']
                end = segment['end']
                text = segment['text'].strip()
            else:
                # Fallback: попытка использовать getattr
                start = getattr(segment, 'start', segment.get('start', 0.0))
                end = getattr(segment, 'end', segment.get('end', 0.0))
                text = getattr(segment, 'text', segment.get('text', '')).strip()
            
            line = f"[{start:.2f} - {end:.2f}] {text}\n"
            
            # Выводим на экран в реальном времени, если verbose=True
            if verbose:
                print(line.strip())
            
            # Записываем в файл
            f.write(line)

