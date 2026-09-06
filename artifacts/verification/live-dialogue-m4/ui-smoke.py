import json
from pathlib import Path

import sounddevice as sd
sd.query_devices = lambda *args, **kwargs: [
    {'name': 'Test microphone', 'max_input_channels': 1, 'max_output_channels': 0},
    {'name': 'Test headphones', 'max_input_channels': 0, 'max_output_channels': 2},
]

from PySide6.QtWidgets import QApplication
from uvt.config import load_config
from uvt.gui.main_window import MainWindow

root = Path(__file__).resolve().parents[3]
out = root / 'artifacts/verification/live-dialogue-m4'
app = QApplication([])
window = MainWindow(load_config('local-dialogue'), profile_name='local-dialogue')
window.show()
app.processEvents()
assert window.profile_combo.currentData() == 'local-dialogue'
assert window.source_lang.currentText() == 'en'
assert window.target_lang.currentText() == 'ru'
assert window.translate_combo.currentText() == 'nllb-ct2'
assert window.tts_combo.currentText() == 'piper'
assert window.live_mode_combo.currentData() == 'voiceover'
assert window.worker is None
window.grab().save(str(out / 'desktop-live.png'))
window.profile_combo.setCurrentIndex(window.profile_combo.findData('local-meeting'))
app.processEvents()
assert window.base_cfg.translation.engine == 'translategemma-mlx'
assert window.live_mode_combo.currentData() == 'subtitles'
assert window.worker is None
window.profile_combo.setCurrentIndex(window.profile_combo.findData('local-dialogue'))
app.processEvents()
assert window.translate_combo.currentText() == 'nllb-ct2'
assert window.live_mode_combo.currentData() == 'voiceover'
window.close()
report = {
    'status': 'passed',
    'scope': 'Qt offscreen UI: profile selection, EN/RU, engines, voice/subtitle switch, no auto-start',
    'audio_devices': 'stubbed; no microphone, sound output or models started',
    'live_call_tested': False,
}
(out / 'ui-result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False))
