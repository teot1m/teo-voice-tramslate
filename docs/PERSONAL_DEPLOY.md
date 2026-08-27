# Личный VPS: Free + GPT + ElevenLabs одновременно

Это запуск для одного личного пользователя, а не публичный SaaS. Одна команда
поднимает три маршрута UVT и не выводит API-ключи за пределы VPS.

## Что нужно

- Linux VPS: для Oracle Always Free достаточно Ampere A1 с 2 OCPU / 12 GB RAM
  и 50+ GB диска; `free-vps` обрабатывает только одну задачу одновременно.
- Домен с тремя поддоменами, например `free.uvt.example`, `cloud.uvt.example`
  и `eleven.uvt.example`, если браузер будет обращаться к VPS по HTTPS.
- Учётная запись OpenAI для GPT-маршрутов и ElevenLabs для третьего TTS.

`free-vps` не является полностью офлайн: Edge TTS отправляет текст Microsoft.
MLX Whisper намеренно не используется, поскольку он предназначен для Apple
Silicon, а VPS работает на Linux.

## Установка на Ubuntu

```bash
sudo apt update
sudo apt install -y ffmpeg git python3 python3-venv curl
curl -fsSL https://ollama.com/install.sh | sh
sudo mkdir -p /opt/uvt
sudo chown "$USER":"$(id -gn)" /opt/uvt
git clone <URL-вашего-репозитория> /opt/uvt
cd /opt/uvt
python3 -m venv .venv
.venv/bin/pip install -e '.[recommended,server,web]'
ollama pull qwen2.5:3b
cp .env.example .env
chmod 600 .env
```

В `.env` задайте:

```dotenv
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=sk_...
UVT_API_TOKEN=длинный-случайный-секрет
UVT_FREE_PUBLIC_URL=https://free.uvt.example
UVT_GPT_PUBLIC_URL=https://cloud.uvt.example
UVT_ELEVEN_PUBLIC_URL=https://eleven.uvt.example
```

Замените примерные домены на свои. `UVT_*_PUBLIC_URL` принимают
только origin, например `https://free.uvt.example`, без пути, query и
fragment. Они нужны web-dashboard, чтобы вкладки Free/GPT/ElevenLabs
обращались к правильным HTTPS-поддоменам. Без этих переменных
межмаршрутные вкладки за reverse proxy не настроены.

Запуск и проверка:

```bash
uvt serve-personal
# Без X-UVT-Token каждый /meta должен вернуть HTTP 401.
curl -i http://127.0.0.1:8765/meta
# С токеном каждый /meta должен вернуть HTTP 200.
curl -H 'X-UVT-Token: длинный-случайный-секрет' http://127.0.0.1:8765/meta
curl -H 'X-UVT-Token: длинный-случайный-секрет' http://127.0.0.1:8766/meta
curl -H 'X-UVT-Token: длинный-случайный-секрет' http://127.0.0.1:8767/meta
```

Не публикуйте порты 8765–8767 через firewall: они остаются на loopback.
Ключ `--open-browser` полезен для локального Mac/desktop-запуска:
`uvt serve-personal --open-browser`. На VPS, в headless-сессии или при non-loopback
`--host` UVT не открывает браузер автоматически. Локальная панель с
`UVT_API_TOKEN` откроется и попросит ввести токен.

## Web-панель и настройки

Корень каждого маршрута — web-dashboard, а не JSON `/meta`:

- Free/Local: `http://127.0.0.1:8765/`;
- GPT: `http://127.0.0.1:8766/`;
- ElevenLabs: `http://127.0.0.1:8767/`.

В локальной/LAN-панели, а на VPS при заданных
`UVT_FREE_PUBLIC_URL`, `UVT_GPT_PUBLIC_URL` и `UVT_ELEVEN_PUBLIC_URL`, вкладки
**Free**, **GPT** и **ElevenLabs** дают доступ к настройкам всех трёх
маршрутов:

- Free: языки, локальный Fast/Balanced/Quality и точный Piper-голос;
- GPT: OpenAI-модели STT, перевода и TTS и голос OpenAI;
- ElevenLabs: OpenAI STT/перевод, ElevenLabs TTS-модель и голос
  из аккаунта или точный Voice ID.

Для Mac M4 с 16 ГБ локальный `local-balanced` — рекомендуемый
баланс скорости и качества; Fast быстрее, Quality требовательнее.
Локальный Piper-набор включает Ирину/Дмитрия для русского и
Тетяну/Микиту для украинского. Точные каталоги облачных моделей и
голосов описаны в [browser/README.md](../browser/README.md).

Кнопка **Сохранить** применяет выбор только к следующим новым
задачам; уже запущенный перевод не изменяется. **Вернуть YAML по
умолчанию** удаляет сохранённое переопределение. Настройки хранятся в
`~/.config/uvt/server-settings.json`, где `~` — home пользователя сервиса;
для systemd-unit из этой инструкции это обычно
`/home/uvt/.config/uvt/server-settings.json`.
Другой путь задаётся через `UVT_SETTINGS_PATH`. API-ключи и `UVT_API_TOKEN`
в этот файл не попадают; dashboard показывает лишь их статус.

Проба голоса Free/Piper выполняется локально. Проба GPT или ElevenLabs
отправляет введённый текст в cloud TTS и расходует API-квоту;
запускайте её только явной кнопкой **Прослушать**.

## HTTPS reverse proxy

Поставьте Caddy и направьте три поддомена на loopback-порты:

Скопируйте [`deploy/Caddyfile.personal.example`](../deploy/Caddyfile.personal.example)
в `/etc/caddy/Caddyfile`, замените домены и перезагрузите Caddy.

После выпуска сертификатов можно открыть HTTPS-корень маршрута в браузере.
При включённом `UVT_API_TOKEN` dashboard попросит тот же токен и сохранит
его только в `sessionStorage` до закрытия вкладки. Затем откройте панель UVT на видео →
**Адрес текущего маршрута** и сохраните HTTPS URL для Free, GPT и ElevenLabs.
Адреса — несекретные настройки; они хранятся в Tampermonkey.

Токен не попадает в DOM видеосайта. Для личной сетевой установки
задайте то же значение в константе `UVT_API_TOKEN` в личной копии
`browser/uvt.user.js`:

```js
const UVT_API_TOKEN = "то-же-значение-что-на-VPS";
```

Не добавляйте `OPENAI_API_KEY` или `ELEVENLABS_API_KEY` в userscript.
Не публикуйте и не экспортируйте личную копию userscript со встроенным токеном.
Userscript 0.15 по умолчанию берёт модель, языки и голос из web-dashboard.
Выбор **Свои настройки для этого видео** включает ручное переопределение;
**Вернуться к настройкам web-панели** снова включает серверные значения.
Переключатель **Free / GPT / ElevenLabs** выбирает маршрут до нажатия
**перевести**; начатая задача остаётся на исходном сервере.

## Автозапуск

Создайте отдельного пользователя `uvt`, затем подключите готовый unit:

```bash
sudo useradd --system --create-home --shell /bin/bash uvt
sudo chown -R uvt:uvt /opt/uvt
sudo cp /opt/uvt/deploy/uvt-personal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now uvt-personal
sudo systemctl status uvt-personal
```

Не запускайте UVT от root и не открывайте напрямую порты процессов.
