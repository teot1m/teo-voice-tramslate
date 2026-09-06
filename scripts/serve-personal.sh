#!/usr/bin/env bash
# Запускает три независимых batch-маршрута UVT для личного использования:
# Free, GPT/OpenAI и GPT-перевод с ElevenLabs TTS.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
# Resolve this checkout even if the environment was installed from another path.
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! command -v uvt >/dev/null 2>&1; then
  echo "UVT не найден. Установите проект: pip install -e '.[recommended,server,web]'" >&2
  exit 1
fi

# Native TorchCodec also needs the library path before Python starts.
exec bash "$PROJECT_DIR/scripts/with-local-runtime.sh" "$(command -v uvt)" serve-personal "$@"
