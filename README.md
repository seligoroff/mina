# Mina - утилита для транскрипции и анализа аудио/видео

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CLI](https://img.shields.io/badge/CLI-Click-green.svg)
![Whisper](https://img.shields.io/badge/Whisper-OpenAI-orange.svg)
![DeepSeek](https://img.shields.io/badge/DeepSeek-API-purple.svg)

Утилита для транскрипции аудиофайлов с помощью моделей Whisper и анализа полученных текстов.

## 🧭 Почему Mina

<table>
<tr>

<td>
Название Mina — это отсылка к Мине Мюррей, героине романа «Дракула» Брэмa Стокера.
Мина — одна из первых «стенографисток» в литературе: она аккуратно собирала, расшифровывала и систематизировала записи, письма и заметки — превращая хаос событий в понятный рассказ. Подобно ей, Mina помогает из фрагментов речи и видео создать структурированный протокол встречи.
Она «слушает», «понимает» и превращает разговор в осмысленный документ.
</td>
<td width="35%">
<img src="mina.png" alt="Mina"/>
</td>
</tr>
</table>


## 📋 Компоненты

Mina предоставляет единую CLI утилиту `cli.py` с тремя командами:

1. **`transcribe`** - Транскрипция аудио с помощью OpenAI Whisper или faster-whisper
2. **`tagcloud`** - Анализ транскрипций и генерация облака тегов
3. **`protocol`** - Создание структурированного протокола из расшифровки с помощью DeepSeek API

---

## 📦 Зависимости

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/) (для декодирования аудио)
- pip-библиотеки:
  - `whisper` (от OpenAI) - для транскрипции
  - `faster-whisper` - опционально, для быстрой транскрипции
  - `pymorphy3` и `pymorphy3-dicts-ru` - опционально, для лемматизации в tagcloud
  - `pyyaml` - для работы с конфигом
  - `click`
  - `torch`
  - `requests`

---

## ⚙️ Установка

Создайте виртуальное окружение (рекомендуется):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Установите зависимости:

```bash
# Используйте requirements.txt для полной установки всех зависимостей
pip install -r requirements.txt

# Дополнительно установите pyyaml (если еще не установлен)
pip install pyyaml
```

### ⚠️ Обязательно: Установка ffmpeg

Whisper использует ffmpeg для декодирования аудио. Убедитесь, что он установлен и доступен в системе.

#### Установка в Linux / WSL:

```bash
sudo apt update
sudo apt install ffmpeg
```

Проверьте установку:

```bash
ffmpeg -version
```

### ⚙️ Настройка конфига для команды `protocol`

Скопируйте пример конфига и заполните своими данными:

```bash
cp config.yaml.example config.yaml
```

Затем отредактируйте `config.yaml` и укажите свой API ключ DeepSeek:

```yaml
deepseek:
  api_key: "ваш_api_ключ_здесь"
  model: "deepseek-chat"
  instructions: "deepseek-protocol-instructions.md"
```

---

## 🚀 Использование

### Общий синтаксис:

```bash
python cli.py <команда> [опции]
```

Просмотр доступных команд:

```bash
python cli.py --help
```

---

### 1. Транскрипция (`transcribe`)

Распознавание речи с таймингами с помощью OpenAI Whisper или faster-whisper.

```bash
python cli.py transcribe -i <аудиофайл> -o <текстовый_вывод> [-m <модель>]
```

**Примеры:**

**С оригинальным Whisper (по умолчанию):**
```bash
python cli.py transcribe -i meeting.mp3 -o transcript.txt
python cli.py transcribe -i interview.wav -o output.txt -m medium
```

**С faster-whisper:**
```bash
python cli.py transcribe -i meeting.mp3 -o transcript.txt -m faster:base
python cli.py transcribe -i meeting.mp3 -o transcript.txt -m faster:small --compute-type float16
```

**Аргументы:**
| Опция | Описание |
|-------|----------|
| `--input, -i` | Путь к входному аудиофайлу (обязательно) |
| `--output, -o` | Путь к выходному .txt файлу (обязательно) |
| `--model, -m` | Модель: tiny, base, small, medium, large. Для faster-whisper используйте формат "faster:model" (например, "faster:base") |
| `--compute-type` | Тип вычислений для faster-whisper (int8, float16, float32) |

**Сравнение моделей Whisper:**

**OpenAI Whisper:**
| Модель   | Размер    | Точность     | Скорость      | Память        |
| -------- | --------- | ------------ | ------------- | ------------- |
| `tiny`   | ~39 MB    | низкая       | очень быстрая | минимальная   |
| `base`   | ~74 MB    | ниже средней | очень быстрая | малая         |
| `small`  | ~244 MB   | средняя      | быстрая       | средняя       |
| `medium` | ~769 MB   | высокая      | медленнее     | большая       |
| `large`  | ~1550 MB  | максимальная | медленная     | очень большая |

**faster-whisper** (оптимизированная реализация, поддерживает те же модели):
| Модель   | Размер    | Точность     | Скорость           | Память        |
| -------- | --------- | ------------ | ------------------ | ------------- |
| `tiny`   | ~39 MB    | низкая       | очень быстрая (+++) | минимальная   |
| `base`   | ~74 MB    | ниже средней | очень быстрая (+++) | малая         |
| `small`  | ~244 MB   | средняя      | быстрая (++)       | средняя       |
| `medium` | ~769 MB   | высокая      | средняя (+)        | большая       |
| `large`  | ~1550 MB  | максимальная | медленная          | очень большая |

> ✅ Рекомендуется начать с `small` (OpenAI Whisper) или `faster:base` (faster-whisper) как сбалансированных по точности и скорости.
> 
> 🚀 faster-whisper работает в 2-4 раза быстрее оригинального Whisper при той же точности, особенно на GPU.
> 
> ❗ На CPU модели `medium` и `large` могут работать медленно для обоих вариантов.

---

### 2. Анализ транскрипций (`tagcloud`)

Генерация облака тегов (частотного списка слов) из текста транскрипции.

```bash
python cli.py tagcloud -i <файл_транскрипции> [-o <выходной_файл>] [--lemmatize] [--stopwords <файл>] [--limit <N>] [--no-names]
```

**Примеры:**
```bash
# Базовый анализ
python cli.py tagcloud -i transcript.txt -o tags.txt

# С лемматизацией и стоп-словами
python cli.py tagcloud -i transcript.txt -o tags.txt --lemmatize --stopwords stopwords.txt

# Исключить имена собственные и показать топ-100 слов
python cli.py tagcloud -i transcript.txt -o tags.txt --lemmatize --no-names --limit 100
```

**Аргументы:**
| Опция | Описание |
|-------|----------|
| `--input, -i` | Путь к файлу с транскрипцией (обязательно) |
| `--output, -o` | Путь к выходному файлу (опционально, если не указан - вывод в консоль) |
| `--limit, -l` | Сколько слов вывести в итоговой статистике (по умолчанию: 50) |
| `--lemmatize` | Включить лемматизацию (требуется pymorphy3) |
| `--stopwords` | Путь к файлу со стоп-словами (по одному слову на строку) |
| `--no-names` | Исключать имена собственные (Name-граммема) |

**Что делает команда:**
- Извлекает текст из строк с таймкодами (убирает таймкоды, но сохраняет текст)
- Извлекает слова (длиной >= 3 символов, кириллица и латиница)
- Опционально: лемматизирует слова, фильтрует по частям речи
- Удаляет стоп-слова (если указан файл)
- Выводит частотный список топ-N слов

---

### 3. Создание протокола (`protocol`)

Создает структурированный протокол из расшифровки с помощью DeepSeek API.

```bash
python cli.py protocol -i <файл_расшифровки> [-o <выходной_файл>] [--config <путь_к_конфигу>]
```

**Примеры:**
```bash
# Все настройки берутся из config.yaml, вывод в консоль
python cli.py protocol -i transcript.txt

# Сохранить результат в файл
python cli.py protocol -i transcript.txt -o protocol.md

# С указанием другого конфига
python cli.py protocol -i transcript.txt -o protocol.md --config custom-config.yaml
```

**Аргументы:**
| Опция | Описание |
|-------|----------|
| `--input, -i` | Путь к файлу с расшифровкой (обязательно) |
| `--output, -o` | Путь к выходному файлу (опционально, если не указан - вывод в консоль) |
| `--config, -c` | Путь к файлу конфигурации (по умолчанию: config.yaml в директории скрипта) |

**Конфигурация (config.yaml):**
```yaml
deepseek:
  api_key: "ваш_api_ключ"
  model: "deepseek-chat"
  instructions: "deepseek-protocol-instructions.md"
```

**Что делает команда:**
- Читает расшифровку из файла
- Читает инструкции из файла (по умолчанию: `deepseek-protocol-instructions.md`)
- Отправляет запрос в DeepSeek API с инструкциями и расшифровкой
- Выводит структурированный протокол (резюме, темы, решения, action items)

---

## 📊 Формат вывода транскрипций

Команда `transcribe` выводит текст в следующем формате:

```plaintext
[0.00 - 5.43] Добро пожаловать на встречу по проекту.
[5.43 - 10.72] Сегодня обсудим план по внедрению.
[10.72 - 15.30] Давайте начнем с обсуждения задач.
```

Формат: `[начало - конец] текст`, где время указано в секундах.

---

## 🔄 Типичный workflow

1. **Транскрипция аудио:**
   ```bash
   python cli.py transcribe -i meeting.mp3 -o transcript.txt
   ```
   или с faster-whisper:
   ```bash
   python cli.py transcribe -i meeting.mp3 -o transcript.txt -m faster:base
   ```

2. **Анализ транскрипции:**
   ```bash
   python cli.py tagcloud -i transcript.txt -o tags.txt --lemmatize --stopwords stopwords.txt
   ```

3. **Создание структурированного протокола:**
   ```bash
   python cli.py protocol -i transcript.txt
   ```

---

## 📌 Настройки

### `transcribe`:
- Язык задан явно: `language='ru'`
- Модель по умолчанию: `small`
- Тайминг выводится по сегментам

### `tagcloud`:
- Минимальная длина слова: 3 символа
- Поддержка кириллицы и латиницы
- Части речи, исключаемые при лемматизации: NPRO, ADVB, PRCL, CONJ, PREP, INTJ

---

## 🛠️ Структура проекта

```
src/
├── cli.py                           # Основной CLI файл с командами
├── utils.py                         # Общие утилиты
├── config.yaml                      # Конфигурация (создается из config.yaml.example)
├── config.yaml.example              # Пример конфигурации
├── deepseek-protocol-instructions.md  # Инструкции для создания протокола
├── stopwords.txt                    # Список стоп-слов (для tagcloud)
└── requirements.txt                 # Зависимости проекта
```

---

## 💡 Рекомендации

- Для быстрой транскрипции используйте faster-whisper: `-m faster:base`
- Для максимальной точности используйте оригинальный Whisper с моделью `medium` или `large`
- Для анализа используйте `tagcloud` с лемматизацией для лучших результатов
- Файл `stopwords.txt` можно настроить под свои нужды
- Для команды `protocol` настройте `config.yaml` с вашим API ключом DeepSeek
