#!/bin/bash
# Double-click in Finder to start this checkout and its three routes.
set -euo pipefail
UVT_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "UVT: запускаю Local / Free, GPT и ElevenLabs."
echo "Панель откроется в браузере. Для остановки нажмите Ctrl+C в этом терминале."
exec bash "$UVT_PROJECT_DIR/scripts/serve-personal.sh" --open-browser "$@"
