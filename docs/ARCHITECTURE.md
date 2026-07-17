# Архитектура UVT

## Два независимых продукта в одном репозитории

UVT использует общие движки и `Segmenter`, но **не смешивает** batch-дубляж с Live-переводом:

| Путь | Точка входа | Временная модель | Результат |
| --- | --- | --- | --- |
| Batch | `uvt dub`, `uvt serve` + userscript, вкладка Batch GUI | Весь источник уже известен; можно положить реплики на его таймлайн | Готовая дорожка, SRT/VTT/JSON и реальные границы TTS |
| Live | `uvt run`, вкладка Live GUI | Следующая фраза ещё неизвестна; отсчёт идёт от source clock захвата | Субтитры и/или voice-over после окончания фразы |

Браузерный userscript — потребитель **batch**-сервера. Он не отправляет потоковый звук в Live-конвейер: сервер скачивает/читает источник, готовит дорожку и только потом script синхронизирует её с `<video>`.

## Ключевые решения

| Решение | Почему |
| --- | --- |
| **Python 3.10+ / asyncio** | ML-экосистема Python-first; тяжёлая работа уходит в CTranslate2, ONNX Runtime, PortAudio и фоновые потоки. |
| **Сервисы + шина сообщений** | Этапы независимо запускаются/останавливаются и общаются immutable-подобными событиями через fan-out очереди. Ошибка одной реплики не валит весь Live. |
| **Реестр движков + ленивые импорты** | Базовая установка не тянет Whisper/GUI/TTS; движки импортируются при выборе. Плагины не меняют ядро. |
| **Единый OpenAI-совместимый переводчик** | Покрывает OpenAI, Groq, DeepSeek, Ollama, LM Studio и другие endpoints через `base_url`/модель/ключ. URL вне localhost — это сетевой маршрут. |
| **16 кГц float32 mono внутри Live** | Общий формат VAD/STT; CaptureService ресемплит нативное устройство один раз. |
| **Source clock и deadline** | Live-перевод измеряется от момента исходного звука, а не от удобного момента завершения Python-кода. |
| **Latest wins, не бесконечная очередь** | Для разговора просроченная реплика хуже пропущенной: тяжёлые стадии берут свежий сегмент, а drops отдельно измеряются. |

## Live поток данных и тайминг

```text
CaptureEngine
  │ native samples + source clock
  ▼
CaptureService ──[audio 16 kHz]──► VADService ──[speech + target/deadline]──► STTService
                                                                                │
                                                                                ▼
TranslationService ──[translation]──┬──► OverlayService / HistoryService
                                     │
                                     ▼
                         TTSService + SpeakerService ──[tts]──► OutputService
                                                                      │
                                                          target wait / latest-wins
                                                                      ▼
                                                        write to selected device

Все сервисы ──[status]──► CLI / GUI
```

`CaptureService` строит непрерывные source-часы из длительности полученных сэмплов и монотонного времени; `VADService` переносит их из sample offsets Segmenter в границы конкретной реплики. Для каждой реплики задаются:

```text
target_ts   = source_end_ts + output.target_delay_s
deadline_ts = target_ts + output.max_backlog_s
```

STT, перевод, TTS и output проверяют дедлайн до тяжёлой работы. При `output.latest_wins: true` накопившаяся очередь сливается до самого свежего сегмента. Уже готовый текст всё ещё доступен оверлею и истории, даже если его звук уже бессмысленно воспроизводить.

`OutputService` ждёт `target_ts` и умеет уступить ещё не начатую запись более свежей готовой реплике. После старта записи он не прерывает звук посередине, чтобы не создавать артефакты.

### Метрики, которые не следует путать

`Trace` хранит отдельные группы данных:

- **processing spans:** `speech_end → stt → translate → tts → play → write_end`;
- **source latency:** `source_end → play` и `source_end → write_end`;
- **sync drift:** отклонение `play`/`write_end` от `target_ts`;
- **counters:** `dropped`, `dropped.<stage>.<reason>`, `output_errors`.

`write_end` означает, что PortAudio/устройство приняло последнюю порцию данных, а не физическое время, когда динамик закончил играть свой аппаратный буфер. GUI показывает эти значения как наблюдаемую телеметрию, не как гарантию абсолютной синхронности.

## Вход и выход по платформам

| ОС | Захват системного звука сейчас | Подсказка |
| --- | --- | --- |
| macOS | `sounddevice` + BlackHole/Multi-Output Device | BlackHole должен быть **input** для UVT; системный output остаётся в Multi-Output. |
| Windows | `wasapi-loopback` (PyAudioWPatch) либо VB-Cable | Можно выбрать конкретный output для loopback или виртуальный кабель. |
| Linux | `sounddevice` + monitor-источник PipeWire/PulseAudio | Выберите `monitor`, а не обычный микрофон. |

`capture.device` и `output.device` независимы. Выход можно направить в наушники или второй виртуальный кабель для OBS/Discord/Zoom. GUI отмечает типичные virtual/monitor имена, но пользователь обязан проверить фактический маршрут — «вход по умолчанию» почти всегда оказывается микрофоном.

## Batch-дубляж

`uvt/dub.py` использует те же STT/перевод/TTS контракты, но не шину Live:

```text
файл/URL → decode → long STT с word timestamps → фразы → batch translation
        → параллельный TTS → time-aligned mix → ffmpeg + SRT/JSON
```

Для каждой TTS-реплики сохраняются фактические `tts_start`/`tts_end`. Оригинал приглушается envelope с короткими рампами только в этих окнах, а не по эвристике длины строки. Для видео `ffmpeg` сохраняет видеопоток (`-c:v copy`) и, по умолчанию, оставляет оригинальную дорожку в контейнере.

Это пофразовый voice-over с ducking, а не отделение голоса от музыки/эффектов и не lip-sync. Ссылка скачивается через yt-dlp до начала обработки; DRM-потоки не поддерживаются.

## Голоса и speaker layer

В Live `SpeakerService` присваивает сессии нейтральные `speaker-N` по лёгким тембровым/F0-признакам, хранит confidence и выбирает роль/явный голос через `speaker.voice_map`. Состояние живёт в памяти процесса и не является биометрической идентификацией.

Автоматический тембр может ошибиться на шуме, музыке, коротких репликах, детских или похожих голосах. `voice: auto` не означает точное определение пола, личности или полноценную диаризацию. Явный `tts.voice` или `tts.voice_gender` имеет приоритет над автоматическим выбором.

В batch сейчас используется F0-эвристика до перевода, чтобы помочь грамматическому роду и выбору тембра. Это также не надёжная диаризация. Клонирование голоса и полноценная diarization требуют отдельной оценки качества, лицензии моделей и согласия говорящих.

## Маршруты данных и профили

Профиль определяет не только скорость/цену, но и куда уходят данные:

| Профиль | STT | Перевод | TTS | Сеть во время работы |
| --- | --- | --- | --- | --- |
| `local` | faster-whisper | локальный Ollama | Piper | нет после установки моделей |
| `free` | faster-whisper | локальный Ollama | Microsoft Edge | Edge получает текст |
| `cloud-fast` | OpenAI-compatible | OpenAI | OpenAI | аудио и текст у выбранного облака |
| `cloud-quality` | OpenAI-compatible | OpenAI | OpenAI | аудио и текст у выбранного облака |

Локальный сервер браузерной кнопки (`127.0.0.1`) не меняет этот факт: он защищает HTTP-соединение между userscript и UVT, а не превращает внешний STT/TTS в локальный.

## Конфигурация и Plugin API

`AppConfig` содержит секции `capture`, `vad`, `stt`, `translation`, `tts`, `speaker`, `output`, `overlay`, `history`, `latency`. Секции разрешают extra-ключи: пользовательский движок получает собственные параметры без изменения схемы.

Контракты в `uvt/interfaces.py`:

- `CaptureEngine.stream()`;
- `VADEngine.prob(frame)`;
- `STTEngine.transcribe()` / `transcribe_long()`;
- `TranslationEngine.translate()` / `translate_batch()`;
- `TTSEngine.synthesize()`.

Движок регистрируется через `@register(kind, name)`. Встроенные модули и `plugins/*.py` подгружаются лениво; `warmup()`/`close()` управляют тяжёлыми ресурсами. Пример — [plugins/README.md](../plugins/README.md).

## GUI и проверка

GUI живёт в основном Qt-потоке, а Live/Batch worker — в daemon threads; сообщения проходят через queued Qt signals. Он намеренно разделяет вкладки Live и Batch, чтобы не показывать `replace`/`dual` как работающие возможности. Старые CLI aliases нормализуются к `voiceover`.

GUI и физические устройства требуют ручной проверки на целевой ОС. Автотесты покрывают core через dummy-движки; перед выпуском нужно проверить хотя бы:

1. `uvt run -p demo`;
2. выбор virtual input и отдельного output в GUI;
3. `uvt dub` на коротком ролике и соответствие `tts_start`/`tts_end`;
4. Live метрики и drops под искусственной нагрузкой;
5. чистый `local` маршрут с отключённой сетью после подготовки моделей.
