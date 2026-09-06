"""Startup errors must keep the existing server and avoid model warmup."""
from __future__ import annotations

import asyncio
import errno
import socket

import pytest

pytest.importorskip("aiohttp")
from aiohttp import web

from uvt import cli, server
from uvt.config import AppConfig
from uvt.server_settings import ServerSettingsStore


def _cfg():
    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.stt.engine = "dummy"
    cfg.translation.engine = "dummy"
    cfg.tts.engine = "dummy"
    return cfg


@pytest.fixture
def occupied_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        yield listener


async def test_busy_loopback_port_has_clear_error_without_warmup_or_harming_listener(
    monkeypatch, tmp_path, occupied_port,
):
    host, port = occupied_port.getsockname()
    prepared = []
    closed = []
    bound = asyncio.Event()
    monkeypatch.setenv("UVT_CACHE", str(tmp_path/"cache"))
    monkeypatch.setenv("UVT_API_TOKEN", "test-startup-token")
    monkeypatch.setattr(server.DubServer, "schedule_local_model_prepare", lambda self: prepared.append(self))

    async def close(self):
        closed.append(self)

    monkeypatch.setattr(server.DubServer, "close_prepared_models", close)
    with pytest.raises(server.ServerBindError) as raised:
        await server.run_server(
            _cfg(), host=host, port=port, stop_event=asyncio.Event(),
            bound_event=bound, settings_store=ServerSettingsStore.memory(),
        )
    error = raised.value
    assert isinstance(error, OSError)
    assert error.host == host and error.port == port
    assert f"http://{host}:{port}" in str(error)
    assert "занят" in str(error).lower()
    assert prepared == []
    assert len(closed) == 1
    assert not bound.is_set()
    # The old listener still accepts connections; the failed invocation must
    # not stop or replace whatever already owns this port.
    with socket.create_connection((host, port), timeout=1.0):
        connection, _ = occupied_port.accept()
        connection.close()
    assert occupied_port.getsockname() == (host, port)


async def test_other_bind_errors_are_not_disguised_as_port_conflict(monkeypatch,tmp_path):
    monkeypatch.setenv("UVT_CACHE",str(tmp_path/"cache"))
    original = PermissionError(errno.EACCES,"permission denied")
    prepared = []
    monkeypatch.setattr(server.DubServer,"schedule_local_model_prepare",lambda self: prepared.append(self))

    async def denied(self):
        raise original

    monkeypatch.setattr(web.TCPSite,"start",denied)
    with pytest.raises(PermissionError) as raised:
        await server.run_server(_cfg(),stop_event=asyncio.Event(),settings_store=ServerSettingsStore.memory())
    assert raised.value is original
    assert prepared == []


@pytest.mark.parametrize("command",["serve","serve-personal"])
def test_cli_reports_busy_port_once_and_returns_failure_without_traceback(monkeypatch,capsys,command):
    host,port="127.0.0.1",18765
    calls=[]
    prepared=[]
    monkeypatch.setattr(cli,"_load_env_file",lambda: None)
    monkeypatch.setattr(cli,"_strip_malloc_env",lambda: None)
    monkeypatch.setattr(cli,"setup_logging",lambda *_args: None)
    monkeypatch.setattr(cli,"load_config",lambda *_a,**_kw: _cfg())
    monkeypatch.setattr("uvt.media_runtime.prepare_media_runtime_for_cli",lambda argv: prepared.append(argv))
    monkeypatch.setattr(ServerSettingsStore,"default",classmethod(lambda cls: cls.memory()))

    async def rejected(*args,**kwargs):
        calls.append(kwargs)
        raise server.ServerBindError(host,port)

    monkeypatch.setattr(server,"run_server" if command=="serve" else "run_personal_servers",rejected)
    option="--port" if command=="serve" else "--free-port"
    arguments=[command,"--host",host,option,str(port)]
    assert cli.main(arguments) == 1
    output=capsys.readouterr()
    assert len(calls)==1 and prepared==[arguments]
    assert f"http://{host}:{port}" in output.err
    assert "занят" in output.err.lower()
    assert "Traceback" not in output.err+output.out
    assert output.err.count(f"http://{host}:{port}") == 1


@pytest.mark.parametrize("command",["serve","serve-personal"])
def test_cli_normal_stop_is_still_success(monkeypatch,command,capsys):
    monkeypatch.setattr(cli,"_load_env_file",lambda:None)
    monkeypatch.setattr(cli,"_strip_malloc_env",lambda:None)
    monkeypatch.setattr(cli,"setup_logging",lambda *_a:None)
    monkeypatch.setattr(cli,"load_config",lambda *_a,**_kw:_cfg())
    monkeypatch.setattr("uvt.media_runtime.prepare_media_runtime_for_cli",lambda argv:None)
    monkeypatch.setattr(ServerSettingsStore,"default",classmethod(lambda cls:cls.memory()))

    async def stopped(*args,**kwargs):
        return None

    monkeypatch.setattr(server,"run_server" if command=="serve" else "run_personal_servers",stopped)
    assert cli.main([command]) == 0
    assert "Traceback" not in capsys.readouterr().err
