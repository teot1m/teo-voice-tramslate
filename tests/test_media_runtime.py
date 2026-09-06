from __future__ import annotations

import os
import subprocess

import pytest

from uvt import media_runtime as runtime


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    monkeypatch.setattr(runtime, "_checked", False)
    monkeypatch.setattr(runtime, "_repaired", False)
    monkeypatch.setattr(runtime, "_error_message", None)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: f"/mock/bin/{name}")
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)


def result(command, code=0, stderr=""):
    return subprocess.CompletedProcess(command, code, stderr=stderr)


def test_other_platforms_do_not_probe(monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: pytest.fail("must not probe"))
    assert runtime.ensure_media_runtime() is False


def test_healthy_pair_is_probed_only_once(monkeypatch):
    calls = []
    def probe(command, **kwargs):
        calls.append((command, kwargs))
        return result(command)
    monkeypatch.setattr(runtime.subprocess, "run", probe)
    assert runtime.ensure_media_runtime() is False
    assert runtime.ensure_media_runtime() is False
    assert len(calls) == 2
    assert {call[0][0] for call in calls} == {"/mock/bin/ffmpeg", "/mock/bin/ffprobe"}
    assert all(call[0][1:] == ["-version"] and call[1]["timeout"] <= 3 for call in calls)
    assert "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ


def test_existing_exact_abi_repairs_both_tools_and_preserves_previous_env(monkeypatch, tmp_path):
    cellar = tmp_path / "Cellar" / "x265"
    compatible = cellar / "3.6" / "lib"
    newer = cellar / "4.1" / "lib"
    compatible.mkdir(parents=True)
    newer.mkdir(parents=True)
    (compatible / "libx265.199.dylib").write_bytes(b"fixture")
    (newer / "libx265.212.dylib").write_bytes(b"fixture")
    monkeypatch.setattr(runtime, "_X265_ROOTS", (cellar,))
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/existing/fallback")
    calls = []
    def probe(command, **kwargs):
        fallback = kwargs["env"].get("DYLD_FALLBACK_LIBRARY_PATH", "")
        calls.append((command[0], fallback))
        if fallback.startswith(str(compatible) + ":"):
            return result(command)
        return result(command, -6, "dyld: Library not loaded: /opt/homebrew/opt/x265/lib/libx265.199.dylib\n")
    monkeypatch.setattr(runtime.subprocess, "run", probe)
    assert runtime.ensure_media_runtime() is True
    assert os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == f"{compatible}:/existing/fallback"
    assert len(calls) == 4
    assert all(str(newer) not in fallback for _, fallback in calls)
    assert runtime.ensure_media_runtime() is True
    assert len(calls) == 4


def test_ffprobe_failure_prevents_committing_fallback(monkeypatch):
    monkeypatch.setattr(runtime, "_candidate_library_dirs", lambda names: ["/mock/compatible"])
    calls = []
    def probe(command, **kwargs):
        calls.append(command)
        candidate = "DYLD_FALLBACK_LIBRARY_PATH" in kwargs["env"]
        if candidate and command[0].endswith("ffmpeg"):
            return result(command)
        return result(command, -6, "Library not loaded: /lib/libx265.199.dylib\n")
    monkeypatch.setattr(runtime.subprocess, "run", probe)
    with pytest.raises(runtime.MediaRuntimeError, match="совместимая установленная версия"):
        runtime.ensure_media_runtime()
    assert "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ
    with pytest.raises(runtime.MediaRuntimeError):
        runtime.ensure_media_runtime()
    assert len(calls) == 4


def test_unrelated_failure_is_not_hidden_by_library_repair(monkeypatch):
    monkeypatch.setattr(runtime.subprocess, "run", lambda command, **kwargs: result(command, 1, "unknown decoder problem"))
    monkeypatch.setattr(runtime, "_candidate_library_dirs", lambda names: pytest.fail("no x265 repair"))
    with pytest.raises(runtime.MediaRuntimeError, match="не известная ошибка"):
        runtime.ensure_media_runtime()


def test_missing_tools_and_probe_timeout_are_explicit(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    with pytest.raises(runtime.MediaRuntimeError, match="нужны ffmpeg и ffprobe"):
        runtime.ensure_media_runtime()
    monkeypatch.setattr(runtime, "_checked", False)
    monkeypatch.setattr(runtime, "_error_message", None)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: f"/mock/bin/{name}")
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 3)
    monkeypatch.setattr(runtime.subprocess, "run", timeout)
    with pytest.raises(runtime.MediaRuntimeError, match="3 секунды"):
        runtime.ensure_media_runtime()


@pytest.mark.parametrize("command", ["serve", "serve-personal", "gui", "dub"])
def test_media_cli_commands_report_runtime_error_before_starting(monkeypatch, capsys, command):
    from uvt import cli
    def broken():
        raise runtime.MediaRuntimeError("mock unavailable")
    monkeypatch.setattr(runtime, "ensure_media_runtime", broken)
    monkeypatch.setattr(cli, "_load_env_file", lambda: None)
    argv = [command, "fixture.mp4"] if command == "dub" else [command]
    assert cli.main(argv) == 1
    assert "Ошибка аудио: mock unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["version", "profiles"])
def test_non_media_cli_commands_do_not_probe(monkeypatch, command):
    from uvt import cli
    monkeypatch.setattr(runtime, "ensure_media_runtime", lambda: pytest.fail("no media probe"))
    monkeypatch.setattr(cli, "_load_env_file", lambda: None)
    assert cli.main([command]) == 0
