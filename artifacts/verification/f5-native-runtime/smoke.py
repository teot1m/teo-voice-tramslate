import asyncio
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from uvt.config import load_config
from uvt.engines.tts_f5 import F5TTSEngine, check_f5_audio_runtime
from uvt.interfaces import VoiceReference

root = Path(__file__).resolve().parents[3]
output = root / 'artifacts/verification/f5-native-runtime'

async def main():
    cfg = load_config('local-natural')
    engine = F5TTSEngine(cfg.tts)
    samples, rate = sf.read(root / 'artifacts/benchmark/source-en.wav', dtype='float32', always_2d=True)
    reference = VoiceReference(
        samples.mean(axis=1), rate, label='neutral-local-fixture',
        text=(root / 'artifacts/benchmark/source-en.txt').read_text().strip(),
    )
    start = time.perf_counter()
    check_f5_audio_runtime()
    decoded = time.perf_counter()
    print('Native reference decoder passed.', flush=True)
    try:
        await engine.warmup()
        warmed = time.perf_counter()
        print('Cached F5 model loaded.', flush=True)
        audio, audio_rate = await engine.synthesize_slot(
            'Проверка локальной озвучки. Завтра обсудим планы проекта.',
            'ru', reference=reference,
        )
        completed = time.perf_counter()
        assert audio_rate == 24000 and len(audio) > audio_rate
        assert np.isfinite(audio).all() and np.max(np.abs(audio)) > .001
        output.mkdir(parents=True, exist_ok=True)
        sf.write(output / 'neutral-ru.wav', audio, audio_rate)
        report = {
            'profile': 'local-natural', 'tts': 'F5-TTS Russian', 'device': engine._device,
            'offline': True, 'nfe_step': engine._nfe_step,
            'decoder_seconds': round(decoded-start, 3),
            'warmup_seconds': round(warmed-decoded, 3),
            'synthesis_seconds': round(completed-warmed, 3),
            'audio_seconds': round(len(audio)/audio_rate, 3),
            'sample_rate': audio_rate, 'finite_non_silent': True,
            'reference': 'Existing neutral project-meeting fixture; no user video used.',
            'scope': 'Native decoder + cached F5 model + one Russian synthesis; not full-video or listening-quality validation.',
        }
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        await engine.close()

asyncio.run(main())
