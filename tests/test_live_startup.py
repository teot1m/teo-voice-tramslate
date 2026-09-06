"""Live capture starts only after model readiness, with safe startup cleanup."""
from __future__ import annotations

import asyncio

import pytest

from uvt.app import Pipeline
from uvt.bus import Bus
from uvt.config import AppConfig
from uvt.metrics import Metrics
from uvt.services.base import Service


class ProbeService(Service):
    def __init__(self, name, events, *, source=False, gate=None, failure=None):
        super().__init__(Bus(), AppConfig(), Metrics())
        self.name = name
        self.consumes = None if source else "input"
        self.events = events
        self.gate = gate
        self.failure = failure
        self.setup_entered = asyncio.Event()
        self.setup_finished = False
        self.captured = asyncio.Event()
        self.torn_down = False

    async def setup(self):
        self.events.append(f"setup:{self.name}")
        self.setup_entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.failure is not None:
            raise self.failure
        self.setup_finished = True
        self.events.append(f"ready:{self.name}")

    async def teardown(self):
        # Mirrors a native worker: cleanup may only happen after it finishes.
        if self.gate is not None:
            assert self.gate.is_set()
        self.events.append(f"teardown:{self.name}")
        self.torn_down = True

    async def run_source(self):
        self.events.append(f"capture:{self.name}")
        self.captured.set()
        await asyncio.Event().wait()


def pipeline_of(*services):
    pipeline = object.__new__(Pipeline)
    pipeline.services = list(services)
    return pipeline


async def wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(poll(), 1)


async def test_capture_waits_for_all_downstream_warmups():
    events = []
    gate = asyncio.Event()
    capture = ProbeService("capture", events, source=True)
    stt = ProbeService("stt", events)
    translation = ProbeService("translation", events, gate=gate)
    pipeline = pipeline_of(capture, stt, translation)
    startup = asyncio.create_task(pipeline.start())
    await translation.setup_entered.wait()
    try:
        assert events == ["setup:translation"]
        assert capture.task is None
        assert not startup.done()
        gate.set()
        assert await asyncio.wait_for(startup, 1) is True
        await asyncio.wait_for(capture.captured.wait(), 1)
        assert events[:5] == [
            "setup:translation", "ready:translation", "setup:stt", "ready:stt", "setup:capture",
        ]
    finally:
        gate.set()
        await pipeline.stop()


async def test_setup_failure_stops_started_services_without_capture():
    events = []
    capture = ProbeService("capture", events, source=True)
    broken = ProbeService("translation", events, failure=ValueError("model unavailable"))
    output = ProbeService("output", events)
    pipeline = pipeline_of(capture, broken, output)

    with pytest.raises(RuntimeError, match="translation.*model unavailable") as error:
        await pipeline.start()

    assert isinstance(error.value.__cause__, ValueError)
    assert capture.task is None
    assert not capture.captured.is_set()
    assert broken.torn_down and output.torn_down
    assert all(service.task is None for service in pipeline.services)


@pytest.mark.parametrize("stop_by_event", [False, True])
async def test_stop_during_warmup_drains_only_active_setup(stop_by_event):
    events = []
    gate = asyncio.Event()
    stop = asyncio.Event()
    capture = ProbeService("capture", events, source=True)
    warming = ProbeService("translation", events, gate=gate)
    output = ProbeService("output", events)
    pipeline = pipeline_of(capture, warming, output)
    startup = asyncio.create_task(pipeline.start(stop_event=stop))
    await warming.setup_entered.wait()

    if stop_by_event:
        stop.set()
    else:
        startup.cancel()
    await wait_until(lambda: warming.task is not None and warming.task.cancelling())
    assert not startup.done()
    assert not warming.torn_down
    assert capture.task is None

    # Native setup must finish before teardown; remaining services never start.
    gate.set()
    if stop_by_event:
        assert await asyncio.wait_for(startup, 1) is False
    else:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(startup, 1)
    assert warming.setup_finished
    assert warming.torn_down and output.torn_down
    assert not capture.captured.is_set()
    assert all(service.task is None for service in pipeline.services)


async def test_service_start_is_nonblocking_and_restart_resets_readiness():
    events = []
    service = ProbeService("translation", events)
    await service.start()
    await service.wait_ready()
    await service.stop()

    gate = service.gate = asyncio.Event()
    service.setup_entered = asyncio.Event()
    service.setup_finished = False
    await service.start()  # Must return while setup still waits for its gate.
    waiting = asyncio.create_task(service.wait_ready())
    await service.setup_entered.wait()
    assert not waiting.done()
    assert not service.setup_finished
    gate.set()
    await asyncio.wait_for(waiting, 1)
    assert service.setup_finished
    await service.stop()


async def test_ready_waiter_wakes_when_task_is_cancelled_before_running():
    service = ProbeService("translation", [])
    await service.start()
    service.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(service.wait_ready(), 1)
    await service.stop()


async def test_preexisting_stop_does_not_start_any_service():
    capture = ProbeService("capture", [], source=True)
    stop = asyncio.Event()
    stop.set()
    assert await pipeline_of(capture).start(stop_event=stop) is False
    assert capture.task is None


async def test_stopping_initialized_service_does_not_erase_ready_result():
    service = ProbeService("translation", [])
    await service.start()
    await service.wait_ready()
    waiting = asyncio.create_task(service.wait_ready())
    await asyncio.sleep(0)
    await service.stop()
    await waiting
