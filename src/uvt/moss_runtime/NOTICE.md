# Vendored MOSS ONNX CPU runtime

`ort_cpu_runtime.py` is unmodified upstream source from
https://github.com/OpenMOSS/MOSS-TTS-Nano/blob/8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb/ort_cpu_runtime.py
(commit `8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb`), licensed under Apache 2.0; see LICENSE.

UVT imports only this NumPy/ONNX runtime. The UVT adapter supplies local model paths,
SentencePiece tokenization, soundfile/scipy PCM processing, and cancellation.
The upstream Torch, Transformers, WeTextProcessing and Gradio dependencies are not used.

Weights are downloaded separately from pinned revisions:
- OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX: f52645cb467506d8e18e746ddd59482685b74e58
- OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX: ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae
