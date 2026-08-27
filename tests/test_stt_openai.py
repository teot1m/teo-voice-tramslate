"""OpenAI transcription model capability guards."""
from __future__ import annotations

import numpy as np
import pytest

from uvt.config import STTConfig
from uvt.engines.stt_openai import OpenAICompatibleSTT


@pytest.mark.parametrize("model", ["gpt-4o-mini-transcribe", "gpt-4o-transcribe"])
async def test_gpt_transcribe_models_skip_unsupported_verbose_json(model):
    engine = OpenAICompatibleSTT(
        STTConfig(engine="openai-compatible", model=model, concurrency=8)
    )
    engine.concurrency_hint = 8

    class ClientMustNotBeCalled:
        async def post(self, *_args, **_kwargs):
            raise AssertionError("unsupported full-file request must be skipped")

    engine._client = ClientMustNotBeCalled()
    spans = await engine.transcribe_long(
        np.zeros(16_000, dtype=np.float32), 16_000, "en"
    )

    assert spans is None
