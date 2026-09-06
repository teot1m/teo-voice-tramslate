#!/bin/bash
# Desktop Live UI only. Audio capture starts after the user presses Start.
set -euo pipefail
UVT_PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$UVT_PROJECT_DIR"
export PYTHONPATH="$UVT_PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -x "$UVT_PROJECT_DIR/.venv/bin/python" ]]; then
  echo "Не найдено окружение UVT. Откройте README_START.md для подготовки проекта." >&2
  exit 1
fi
echo "Открываю локальный диалог. Выберите источник звука и наушники, затем нажмите «Запустить Live перевод»."
exec bash "$UVT_PROJECT_DIR/scripts/with-local-runtime.sh" "$UVT_PROJECT_DIR/.venv/bin/python" -m uvt gui -p local-dialogue "$@"
