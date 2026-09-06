# MOSS: four local voices

Neutral Russian probe: «Здравствуйте! Выберите удобный голос для перевода.»
ONNX CPU, 4 threads; HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.

| Voice | Official reference group | Synthesis | Audio duration | RTF |
|---|---|---:|---:|---:|
| [Adam](adam-ru.wav) | English Male | 4.850 s | 4.32 s | 1.123 |
| [Nathan](nathan-ru.wav) | English Male | 3.938 s | 5.36 s | 0.735 |
| [Ava](ava-ru.wav) | English Female | 3.189 s | 4.72 s | 0.676 |
| [Bella](bella-ru.wav) | English Female | 4.579 s | 6.80 s | 0.673 |

RTF = synthesis time / generated duration; below 1 means faster than playback.
Warmup: 4.449 s; peak RSS: 1488.3 MiB. Adam was synthesized first.
Model closed after verification. These are complete utterances, not streaming first-audio measurements.
Finite nonempty audio was verified; subjective quality and intelligibility were not separately evaluated.
The official voice references are English; all four probes request Russian synthesis supported by this multilingual model.

Sources: [official voice manifest](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX/blob/f52645cb467506d8e18e746ddd59482685b74e58/browser_poc_manifest.json), [official codec](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX/tree/ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae).

Details and manifest checksum: [results.json](results.json). Reproduction: [verify.py](verify.py), run from the project root with its Python environment and PYTHONPATH=src.
