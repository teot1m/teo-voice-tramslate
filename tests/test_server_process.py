"""Subprocess progress and cancellation guards; no network or media required."""

import asyncio
import os
import subprocess
import sys
import time

import pytest

import uvt.server as server


async def test_cr_progress_is_emitted_before_eof_and_across_chunks():
    stream = asyncio.StreamReader()
    lines = []
    reader = asyncio.create_task(server._read_process_lines(stream, lines.append))
    stream.feed_data(b'first\rUVT_PROG')
    await asyncio.sleep(0)
    assert lines == ['first']
    stream.feed_data(b'RESS: 42%\r\nlast')
    await asyncio.sleep(0)
    assert lines == ['first', 'UVT_PROGRESS: 42%']
    assert not reader.done()
    stream.feed_eof()
    await reader
    assert lines[-1] == 'last'


async def test_oversized_diagnostic_is_bounded_and_next_progress_survives():
    stream = asyncio.StreamReader(limit=65536)
    stream.feed_data(b'x' * 200000 + b'\rUVT_PROGRESS: 64.5%\n')
    stream.feed_eof()
    lines = []
    await server._read_process_lines(stream, lines.append)
    assert len(lines[0]) <= 16384
    assert lines[1:] == ['UVT_PROGRESS: 64.5%']


async def test_progress_callback_failure_does_not_stop_pipe_draining():
    stream = asyncio.StreamReader()
    stream.feed_data(b'bad\ngood\r')
    stream.feed_eof()
    lines = []

    def on_line(line):
        if line == 'bad':
            raise ValueError('presentation error')
        lines.append(line)

    await server._read_process_lines(stream, on_line)
    assert lines == ['good']


async def test_real_process_drains_both_pipes_and_long_diagnostics():
    lines = []
    code = "import sys; sys.stdout.buffer.write(b'x'*200000+b'\\rUVT_PROGRESS: 60%\\n'); sys.stdout.flush(); print('stderr-marker',file=sys.stderr,flush=True)"
    await server._run_process([sys.executable, '-c', code], 5, 'fixture', on_line=lines.append)
    assert 'UVT_PROGRESS: 60%' in lines
    assert 'stderr-marker' in lines
    assert max(map(len, lines)) <= 16384


async def test_real_process_failure_retains_exit_code():
    with pytest.raises(RuntimeError, match=r'код 7'):
        await server._run_process([sys.executable, '-c', 'raise SystemExit(7)'], 5, 'fixture', on_line=lambda _line: None)


class FakeProcess:
    pid = None
    stderr = None

    def __init__(self, *, exited=False, stdout=None):
        self.returncode = 0 if exited else None
        self.stdout = stdout
        self.stopped = asyncio.Event()
        self.terminated = 0
        self.killed = 0
        if exited:
            self.stopped.set()

    async def wait(self):
        await self.stopped.wait()
        return self.returncode

    def terminate(self):
        self.terminated += 1
        self.returncode = -15
        self.stopped.set()

    def kill(self):
        self.killed += 1
        self.returncode = -9
        self.stopped.set()


def patch_process(monkeypatch, process):
    async def create(*_args, **kwargs):
        if os.name == 'posix':
            assert kwargs['start_new_session'] is True
        return process
    monkeypatch.setattr(server.asyncio, 'create_subprocess_exec', create)


async def test_finished_process_with_open_pipe_cannot_outlive_deadline(monkeypatch):
    process = FakeProcess(exited=True, stdout=asyncio.StreamReader())
    patch_process(monkeypatch, process)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='не уложился'):
        await asyncio.wait_for(server._run_process(['fixture'], .03, 'fixture', on_line=lambda _line: None), 1)
    assert time.monotonic() - start < .5


async def test_reader_failure_terminates_live_child_immediately(monkeypatch):
    class BrokenStream:
        async def read(self, _size):
            raise OSError('pipe failed')

    process = FakeProcess(stdout=BrokenStream())
    patch_process(monkeypatch, process)
    with pytest.raises(OSError, match='pipe failed'):
        await server._run_process(['fixture'], 5, 'fixture', on_line=lambda _line: None)
    assert process.terminated == 1


async def test_repeated_cancellation_cannot_interrupt_cleanup(monkeypatch):
    process = FakeProcess(stdout=asyncio.StreamReader())
    patch_process(monkeypatch, process)
    cleanup_started, release = asyncio.Event(), asyncio.Event()
    original = server._kill_process

    async def cleanup(proc):
        cleanup_started.set()
        await release.wait()
        await original(proc)

    monkeypatch.setattr(server, '_kill_process', cleanup)
    task = asyncio.create_task(server._run_process(['fixture'], 5, 'fixture', on_line=lambda _line: None))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.wait_for(cleanup_started.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated == 1


async def test_unowned_process_never_signals_callers_process_group(monkeypatch):
    process = FakeProcess()
    process.pid = os.getpid()
    monkeypatch.setattr(server.os, 'killpg', lambda *_a: pytest.fail('must not signal inherited group'), raising=False)
    await server._kill_process(process)
    assert process.terminated == 1


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process groups')
async def test_timeout_stops_descendant_even_when_leader_already_exited():
    lines = []
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('child-ready',flush=True); time.sleep(30)"
    parent = f"import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',{child!r}]); print('CHILD_PID='+str(p.pid),flush=True)"
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='не уложился'):
        await asyncio.wait_for(server._run_process([sys.executable, '-c', parent], .3, 'fixture', on_line=lines.append), 4)
    assert time.monotonic() - start < 3
    child_pid = next(int(line.split('=', 1)[1]) for line in lines if line.startswith('CHILD_PID='))
    assert 'child-ready' in lines
    # A terminated child can briefly remain as an init-owned zombie. It must
    # not be alive and sleeping after the job has returned its timeout error.
    for _ in range(100):
        status = subprocess.run(['ps', '-o', 'stat=', '-p', str(child_pid)], capture_output=True, text=True).stdout.strip()
        if not status or status.startswith('Z'):
            break
        await asyncio.sleep(.01)
    assert not status or status.startswith('Z'), f'owned descendant still running: {status}'
