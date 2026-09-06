"""Demucs cache, chunk progress and cooperative cancellation without model weights."""

import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import uvt.separate as separate


class FakeModel:
    samplerate = 10
    segment = 2.0
    sources = ['drums', 'bass', 'other', 'vocals']

    def __init__(self):
        self.moved = []

    def eval(self):
        return self

    def to(self, device):
        self.moved.append(device)
        return self


@pytest.fixture
def fake_runtime(monkeypatch):
    torch = pytest.importorskip('torch')
    model = FakeModel()
    chunks = []
    syncs = []

    def apply_model(model, mix, *, callback=None, **kwargs):
        assert kwargs['split'] is True
        assert kwargs['num_workers'] == 0
        for offset in range(0, mix.shape[-1], 15):
            callback({'state': 'start', 'segment_offset': offset})
            chunks.append(offset)
            callback({'state': 'end', 'segment_offset': offset})
        return torch.stack([mix * .1, mix * .2, mix * .3, mix * .4], dim=1)

    monkeypatch.setitem(sys.modules, 'demucs.apply', SimpleNamespace(apply_model=apply_model))
    monkeypatch.setattr(separate, '_load_separation_model', lambda *_args, **_kwargs: model)
    monkeypatch.setattr(torch.mps, 'synchronize', lambda: syncs.append('sync'))
    monkeypatch.setattr(torch.mps, 'empty_cache', lambda: syncs.append('clear'))
    return SimpleNamespace(model=model, chunks=chunks, syncs=syncs, torch=torch, apply_model=apply_model)


def source():
    return np.linspace(-1, 1, 60, dtype=np.float32)


def test_native_chunks_report_progress_and_preserve_source_sum(fake_runtime):
    updates = []
    result = separate.separate_speech(source(), 10, device='mps', progress=lambda done, total: updates.append((done, total)))
    assert updates[0] == (0.0, 6.0)
    assert updates[-1] == (6.0, 6.0)
    assert len(updates) == len(fake_runtime.chunks) + 2
    assert all(left[0] <= right[0] for left, right in zip(updates, updates[1:]))
    assert all(round(100 * done / total) < 100 for done, total in updates[1:-1])
    np.testing.assert_allclose(result.speech + result.background, separate._as_stereo(source()), atol=1e-6)
    assert fake_runtime.model.moved[-1] == 'cpu'
    assert fake_runtime.syncs == ['sync'] * len(fake_runtime.chunks) + ['clear']


def test_cancellation_stops_before_next_native_chunk_and_releases_weights(fake_runtime):
    cancelled = threading.Event()
    updates = []

    def progress(done, total):
        updates.append((done, total))
        if done > 0:
            cancelled.set()

    with pytest.raises(separate.SeparationCancelled):
        separate.separate_speech(source(), 10, device='mps', progress=progress, cancel_event=cancelled)
    assert issubclass(separate.SeparationCancelled, asyncio.CancelledError)
    assert not issubclass(separate.SeparationCancelled, Exception)
    assert len(fake_runtime.chunks) == 1
    assert updates[-1][0] < updates[-1][1]
    assert fake_runtime.model.moved[-1] == 'cpu'
    assert fake_runtime.syncs[-1] == 'clear'


def test_already_cancelled_does_not_load_any_model(monkeypatch):
    cancelled = threading.Event()
    cancelled.set()
    monkeypatch.setattr(separate, '_load_separation_model', lambda *_a, **_k: pytest.fail('must not load'))
    with pytest.raises(separate.SeparationCancelled):
        separate.separate_speech(source(), 10, cancel_event=cancelled)


def test_old_demucs_callback_api_has_actionable_fallback_error(fake_runtime, monkeypatch):
    def old_apply(model, mix, **kwargs):
        pytest.fail('must not enter uninterruptible inference')

    monkeypatch.setitem(sys.modules, 'demucs.apply', SimpleNamespace(apply_model=old_apply))
    with pytest.raises(RuntimeError, match='обновите Demucs'):
        separate.separate_speech(source(), 10, cancel_event=threading.Event())


def test_multimodel_shift_progress_is_monotonic(fake_runtime, monkeypatch):
    fake_runtime.model.models = [FakeModel(), FakeModel()]

    def bag_apply(model, mix, *, callback=None, **kwargs):
        for model_index in range(2):
            for shift in range(2):
                for offset in (0, 15, 30, 45):
                    callback({'state': 'end', 'model_idx_in_bag': model_index, 'shift_idx': shift, 'segment_offset': offset})
        return fake_runtime.torch.stack([mix * .1, mix * .2, mix * .3, mix * .4], dim=1)

    monkeypatch.setitem(sys.modules, 'demucs.apply', SimpleNamespace(apply_model=bag_apply))
    updates = []
    separate.separate_speech(source(), 10, device='cpu', shifts=2, progress=lambda done, total: updates.append(done / total))
    assert updates == sorted(updates)
    assert updates[-1] == 1
    assert updates[4] == pytest.approx(.25)
    assert updates[8] == pytest.approx(.5)
    assert updates[12] == pytest.approx(.75)


def test_empty_audio_never_loads_models(monkeypatch):
    monkeypatch.setattr(separate, '_load_separation_model', lambda *_a, **_k: pytest.fail('must not load'))
    result = separate.separate_speech(np.array([], dtype=np.float32), 44100)
    assert result.speech.shape == (0,)
    assert result.background.shape == (0, 2)


def test_cached_hf_model_resolves_only_local_files_before_loading(monkeypatch, tmp_path):
    calls, loaded = [], []
    bag = tmp_path / 'htdemucs.yaml'
    bag.write_text('models: [first, second]\n')
    for name in ('first', 'second'):
        (tmp_path / f'{name}.safetensors').touch()

    def cached(repo, filename, **kwargs):
        assert kwargs == {'local_files_only': True}
        assert not loaded, 'All paths must resolve before loading any weights'
        calls.append((repo, filename))
        return str(tmp_path / filename)

    def load_weights(filename):
        loaded.append(Path(filename).name)
        return FakeModel()

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=cached))
    monkeypatch.setitem(sys.modules, 'demucs.hf', SimpleNamespace(DEFAULT_NAMESPACE='adefossez', hf_repo_name=lambda name: 'HTDemucs', load_safetensors_model=load_weights))
    monkeypatch.setitem(sys.modules, 'demucs.apply', SimpleNamespace(BagOfModels=lambda models, *_args: models))
    result = separate._load_cached_hf_model('htdemucs')
    assert len(result) == 2
    assert calls == [('adefossez/HTDemucs', 'htdemucs.yaml'), ('adefossez/HTDemucs', 'first.safetensors'), ('adefossez/HTDemucs', 'second.safetensors')]
    assert loaded == ['first.safetensors', 'second.safetensors']


def test_incomplete_hf_cache_does_not_load_partial_weights(monkeypatch, tmp_path):
    bag = tmp_path / 'htdemucs.yaml'
    bag.write_text('models: [first, missing]\n')

    def cached(repo, filename, **kwargs):
        assert kwargs['local_files_only'] is True
        if filename.startswith('missing'):
            raise FileNotFoundError(filename)
        return str(bag if filename.endswith('.yaml') else tmp_path / filename)

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=cached))
    monkeypatch.setitem(sys.modules, 'demucs.hf', SimpleNamespace(DEFAULT_NAMESPACE='adefossez', hf_repo_name=lambda name: 'HTDemucs', load_safetensors_model=lambda *_a: pytest.fail('partial weights loaded')))
    monkeypatch.setitem(sys.modules, 'demucs.apply', SimpleNamespace(BagOfModels=lambda *_args: pytest.fail('incomplete bag')))
    with pytest.raises(FileNotFoundError):
        separate._load_cached_hf_model('htdemucs')


def test_hf_cache_miss_preserves_legacy_checkpoint_fallback(monkeypatch):
    expected = object()
    monkeypatch.setattr(separate, '_load_cached_hf_model', lambda *_a: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(separate, '_load_cached_legacy_model', lambda name: expected if name == 'htdemucs' else None)
    monkeypatch.setitem(sys.modules, 'demucs.pretrained', SimpleNamespace(get_model=lambda *_a: pytest.fail('network-capable loader used')))
    assert separate._load_separation_model('htdemucs', allow_download=False) is expected


def test_missing_local_models_never_fall_back_to_network(monkeypatch):
    def missing(*_args):
        raise FileNotFoundError()

    monkeypatch.setattr(separate, '_load_cached_hf_model', missing)
    monkeypatch.setattr(separate, '_load_cached_legacy_model', missing)
    monkeypatch.setitem(sys.modules, 'demucs.pretrained', SimpleNamespace(get_model=lambda *_a: pytest.fail('network-capable loader used')))
    with pytest.raises(RuntimeError, match='Автоматическое скачивание отключено'):
        separate._load_separation_model('htdemucs', allow_download=False)


def test_download_requires_explicit_opt_in(monkeypatch):
    expected = object()
    monkeypatch.setitem(sys.modules, 'demucs.pretrained', SimpleNamespace(get_model=lambda name: expected))
    assert separate._load_separation_model('htdemucs', allow_download=True) is expected


def test_legacy_loader_uses_original_torch_checkpoint_cache(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path))
    seen = []
    expected = object()
    monkeypatch.setitem(sys.modules, 'demucs.pretrained', SimpleNamespace(REMOTE_ROOT=tmp_path / 'bags'))
    monkeypatch.setitem(sys.modules, 'demucs.repo', SimpleNamespace(
        LocalRepo=lambda root: seen.append(root) or 'local',
        BagOnlyRepo=lambda root, models: seen.append((root, models)) or 'bags',
        AnyModelRepo=lambda models, bags: SimpleNamespace(get_model=lambda name: expected),
    ))
    assert separate._load_cached_legacy_model('htdemucs') is expected
    assert seen == [checkpoints, (tmp_path / 'bags', 'local')]
