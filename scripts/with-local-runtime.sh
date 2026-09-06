#!/usr/bin/env bash
# Run a UVT command with a task-local Homebrew dylib fallback when needed.
# No Homebrew symlinks, libraries, or system settings are changed.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Использование: scripts/with-local-runtime.sh <команда> [аргументы...]" >&2
  exit 2
fi

UVT_FFMPEG="$(command -v ffmpeg || true)"
UVT_FFPROBE="$(command -v ffprobe || true)"
if [[ -n "$UVT_FFMPEG" && "$(uname -s)" == "Darwin" ]] && ! { "$UVT_FFMPEG" -version >/dev/null 2>&1; } 2>/dev/null; then
  # A Homebrew upgrade can point opt/x265 to a new ABI while an older ffmpeg
  # still needs a library that remains in the Cellar. Probe real installations;
  # never pretend a new ABI is compatible by renaming/symlinking its library.
  UVT_RUNTIME_FIXED=0
  for UVT_X265_LIBRARY in /opt/homebrew/Cellar/x265/*/lib/libx265.*.dylib; do
    [[ -f "$UVT_X265_LIBRARY" ]] || continue
    UVT_LIBRARY_DIR="${UVT_X265_LIBRARY%/*}"
    UVT_LIBRARY_FALLBACK="$UVT_LIBRARY_DIR${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
    if DYLD_FALLBACK_LIBRARY_PATH="$UVT_LIBRARY_FALLBACK" "$UVT_FFMPEG" -version >/dev/null 2>&1 &&
       [[ -n "$UVT_FFPROBE" ]] &&
       DYLD_FALLBACK_LIBRARY_PATH="$UVT_LIBRARY_FALLBACK" "$UVT_FFPROBE" -version >/dev/null 2>&1; then
      export DYLD_FALLBACK_LIBRARY_PATH="$UVT_LIBRARY_FALLBACK"
      UVT_RUNTIME_FIXED=1
      echo "UVT: совместимая библиотека FFmpeg подключена только для этого запуска." >&2
      break
    fi
  done
  if [[ "$UVT_RUNTIME_FIXED" == 0 ]]; then
    echo "FFmpeg не запускается. Исправьте его установку перед обработкой видео." >&2
    "$UVT_FFMPEG" -version
    exit 1
  fi
fi

exec "$@"
