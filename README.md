# UVT — Universal Voice Translator

**Закадровый перевод любого видео и звука — кнопкой в браузере, локально, бесплатно.**
*Real-time AI voice-over translation for any video or audio: browser button, offline dubbing, live system-audio mode.*

![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)
![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

UVT не привязан к конкретным сайтам и браузерам. Три режима работы:

| Режим | Что делает | Для чего |
| --- | --- | --- |
| 🖱 **Кнопка в браузере** | как [voice-over-translation](https://github.com/ilyhalight/voice-over-translation), но перевод делает ваш движок | YouTube и почти любой сайт с видео |
| 🎬 **`uvt dub`** | дубляж файла или ссылки: голос заменён, видео не перекодируется, субтитры рядом | локальные видео, ролики по ссылке |
| 🎧 **`uvt run`** | живой перевод системного звука | Telegram, Zoom, плееры, игры |

```text
Видео/звук → STT (Whisper) → перевод (LLM) → TTS → озвучка поверх приглушённого
оригинала, точно по таймкодам реплик + субтитры (srt/vtt/json)
```

## Возможности

- **Пофразовая синхронизация**: пословные таймкоды Whisper, реплики ложатся на
  свои места; длинные фразы автоматически ускоряются, чтобы влезть в тайминг.
- **Автоопределение мужского/женского голоса** по высоте тона: диалог озвучивается
  двумя голосами, а переводчик получает род говорящего («я готова», а не «я готов»).
- **Выравнивание громкости**: озвучка подгоняется под уровень оригинальной речи.
- **Полностью бесплатный режим**: Whisper локально + LLM через Ollama (или
  Google-перевод без ключа) + Edge TTS — ноль облачных платежей.
- **Любой LLM для перевода**: OpenAI, DeepSeek, Groq, OpenRouter, **Ollama**,
  **LM Studio**, llama.cpp — один OpenAI-совместимый движок; настраиваемый
  промпт и глоссарий.
- **STT на выбор**: faster-whisper (CPU), mlx-whisper (GPU Apple Silicon),
  облачный Whisper одним запросом (OpenAI/Groq; у Groq бесплатно и за секунды).
- **TTS на выбор**: Edge (бесплатно), OpenAI (естественнее), Kokoro (локально).
- **Плагины**: свой STT/переводчик/TTS/захват — один `.py`-файл в
  [plugins/](plugins/README.md), ядро не меняется.
- **Живой режим**: захват системного звука (BlackHole/VB-Cable/PipeWire/WASAPI),
  субтитры + озвучка на лету.

## Быстрый старт

Нужны Python 3.10+ и ffmpeg (`brew install ffmpeg` / `apt install ffmpeg`).

```bash
git clone <ваш-репозиторий> uvt && cd uvt
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[recommended,server,web]"
uvt run --profile demo        # самопроверка без моделей и сети
```

### Кнопка в браузере (главный сценарий)

1. `pip install` уже сделан выше; запустите сервер:

   ```bash
   uvt serve -p free       # бесплатный конвейер (нужна Ollama: см. profiles/free.yaml)
   # или: uvt serve -p cloud   (GPT-перевод + голоса OpenAI, ключ в OPENAI_API_KEY)
   ```

2. Установите [Tampermonkey](https://www.tampermonkey.net/) и добавьте скрипт
   [browser/uvt.user.js](browser/uvt.user.js) ([инструкция](browser/README.md)).
3. Откройте видео → на плеере появится кнопка **UVT** → нажмите. Рядом — выбор
   языков и голоса, ползунки громкостей, крестик отмены.

Кнопка сама достаёт поток: yt-dlp для известных сайтов, а для остальных —
манифесты прямо из ресурсов страницы (ролики-превью отсекаются по длительности).
Сайты с DRM (Netflix и т.п.) не поддерживаются.

### Дубляж файла или ссылки

```bash
uvt dub фильм.mp4 -p cloud                 # → фильм.dub.mkv + .srt + .json
uvt dub "https://youtube.com/watch?v=…"    # ролик по ссылке
uvt dub файл.mp4 --duck-db -60             # почти убрать оригинальный голос
```

### Живой перевод приложений (Telegram, Zoom, плееры)

Один раз заверните системный звук в виртуальное устройство
(macOS: `brew install blackhole-2ch` + Multi-Output Device; Windows: встроенный
WASAPI loopback; Linux: monitor-источник PipeWire), затем:

```bash
uvt run -p live
```

Перевод звучит после конца фразы (~2–4 с) — это честная физика реального времени.

## Бесплатно или качественнее

| | Профиль `free` | Профиль `cloud` |
| --- | --- | --- |
| STT | Whisper локально | + Groq одним запросом (бесплатный ключ, секунды на ролик) |
| Перевод | Ollama / Google без ключа | GPT (`OPENAI_API_KEY`) |
| Озвучка | Edge (бесплатно) | голоса OpenAI |
| Цена часа видео | 0 | ~$0.3–0.5 |

Все параметры — обычный YAML в [profiles/](profiles/), полный справочник —
[profiles/default.yaml](profiles/default.yaml).

## Архитектура

Каждый этап — независимый asyncio-сервис, обмен через шину сообщений; движки —
плагины с ленивыми зависимостями. Подробности, бюджет задержек и Plugin API —
в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Дорожная карта

- ✅ пофразовый дубляж, кнопка в браузере, авто-пол голоса, бесплатный режим,
  выравнивание громкости, отмена задач, кэш
- 🔜 куки для капризных CDN, автоперевод при открытии видео, горячие клавиши,
  passthrough-микшер для живой замены голоса, стабильный Windows (WASAPI/GUI)
- 🔭 полная диаризация спикеров, клонирование голоса, память перевода

## Дисклеймер

UVT — инструмент для личного использования: переводите контент, к которому у
вас есть законный доступ, и соблюдайте условия сайтов и API-провайдеров.
Движок `google-free` использует неофициальный публичный endpoint и может
ограничиваться; для стабильной работы используйте Ollama или ключевые API.

## Лицензия

MIT — см. [LICENSE](LICENSE).
