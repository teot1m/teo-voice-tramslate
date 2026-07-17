#!/usr/bin/env bash
# Запускает два независимых batch-сервера UVT для личного использования.
# Free держит Whisper/Ollama, Cloud использует OpenAI и не делит с ним очередь.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FREE_PORT="${UVT_FREE_PORT:-8765}"
CLOUD_PORT="${UVT_CLOUD_PORT:-8766}"
UVT_HOST="${UVT_HOST:-127.0.0.1}"

cd "$PROJECT_DIR"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! command -v uvt >/dev/null 2>&1; then
  echo "UVT не найден. Установите проект: pip install -e '.[recommended,server,web]'" >&2
  exit 1
fi

cleanup() {
  trap - INT TERM EXIT
  kill "${free_pid:-}" "${cloud_pid:-}" 2>/dev/null || true
  wait "${free_pid:-}" "${cloud_pid:-}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

uvt serve -p free-vps --host "$UVT_HOST" --port "$FREE_PORT" &
free_pid=$!
uvt serve -p cloud-fast --host "$UVT_HOST" --port "$CLOUD_PORT" &
cloud_pid=$!

echo "UVT Free:  http://${UVT_HOST}:${FREE_PORT}"
echo "UVT Cloud: http://${UVT_HOST}:${CLOUD_PORT}"
echo "Остановить: Ctrl+C"
wait "$free_pid" "$cloud_pid"
