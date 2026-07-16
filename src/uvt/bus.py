"""Шина сообщений: топики с несколькими подписчиками поверх asyncio.Queue.

Публикация не блокирует: при переполнении очереди подписчика самое старое
сообщение вытесняется (конвейер реального времени должен отставать, а не
накапливать бесконечный буфер).
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("uvt.bus")


class Topic:
    def __init__(self, name: str, maxsize: int = 512) -> None:
        self.name = name
        self.maxsize = maxsize
        self.dropped = 0
        self._subs: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(self.maxsize)
        self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subs:
            self._subs.remove(q)

    def publish(self, item) -> None:
        for q in self._subs:
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                self.dropped += 1
                if self.dropped % 100 == 1:
                    log.warning(
                        "топик '%s': подписчик не успевает, вытеснено %d сообщений",
                        self.name, self.dropped,
                    )
                try:
                    q.get_nowait()
                    q.put_nowait(item)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass


class Bus:
    def __init__(self) -> None:
        self._topics: dict[str, Topic] = {}

    def topic(self, name: str) -> Topic:
        if name not in self._topics:
            self._topics[name] = Topic(name)
        return self._topics[name]
