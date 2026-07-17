# Личный VPS: Free + GPT Cloud одновременно

Это запуск для одного личного пользователя, а не публичный SaaS. Он поднимает
два процесса UVT и не выводит OpenAI-ключ за пределы VPS.

## Что нужно

- Linux VPS: для Oracle Always Free достаточно Ampere A1 с 2 OCPU / 12 GB RAM
  и 50+ GB диска; `free-vps` обрабатывает только одну задачу одновременно.
- Домен с двумя поддоменами, например `free.uvt.example` и
  `cloud.uvt.example`, если браузер будет обращаться к VPS по HTTPS.
- Учётная запись OpenAI только для cloud-версии.

`free-vps` не является полностью офлайн: Edge TTS отправляет текст Microsoft.
MLX Whisper намеренно не используется, поскольку он предназначен для Apple
Silicon, а VPS работает на Linux.

## Установка на Ubuntu

```bash
sudo apt update
sudo apt install -y ffmpeg git python3 python3-venv curl
curl -fsSL https://ollama.com/install.sh | sh
git clone <URL-вашего-репозитория> /opt/uvt
cd /opt/uvt
python3 -m venv .venv
.venv/bin/pip install -e '.[recommended,server,web]'
ollama pull qwen2.5:3b
cp .env.example .env
```

В `.env` задайте:

```dotenv
OPENAI_API_KEY=sk-...
UVT_API_TOKEN=длинный-случайный-секрет
```

Запуск и проверка:

```bash
bash scripts/serve-personal.sh
curl -H 'X-UVT-Token: длинный-случайный-секрет' http://127.0.0.1:8765/meta
curl -H 'X-UVT-Token: длинный-случайный-секрет' http://127.0.0.1:8766/meta
```

Не публикуйте порты 8765 и 8766 через firewall: они остаются на loopback.

## HTTPS reverse proxy

Поставьте Caddy и направьте два поддомена на loopback-порты:

Скопируйте [`deploy/Caddyfile.personal.example`](../deploy/Caddyfile.personal.example)
в `/etc/caddy/Caddyfile`, замените домены и перезагрузите Caddy.

После выпуска сертификатов замените в начале `browser/uvt.user.js` только URL:

```js
free:  { url: "https://free.uvt.example",  label: "Бесплатный — Whisper + Qwen" },
cloud: { url: "https://cloud.uvt.example", label: "GPT Cloud — OpenAI" },
const UVT_API_TOKEN = "то-же-значение-что-на-VPS";
```

Не добавляйте `OPENAI_API_KEY` в userscript. Переключатель «Модель» выбирает
маршрут до нажатия «перевести»; начатая задача остаётся на исходном сервере.

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
