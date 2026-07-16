"""
An in-process, pub/sub event system for pipeline events (like ModuleStarted or ArtifactCreated).

1- No over-engineering: It is absolutely not a distributed message broker. There is no Kafka, RabbitMQ, or Redis dependency here.

2- Keep it local: Right now, the engine runs one pipeline in a single process at a time. A straightforward list of synchronous 
   callbacks gives us the exact same event stream without the network lag or the headaches of distributed failures.

3- Future-proof: If we ever need to move to a distributed setup down the road, we only have to rewrite this one module. The rest
   of the codebase just talks to the EventBus interface and won't change at all.

"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List

from ml_framework.orchestration.v2.contracts import Event, EventType

logger = logging.getLogger("ml_framework.orchestration.v2.event_bus")

Subscriber = Callable[[Event], None]


class EventBus:
    """
    Synchronous in-process publish/subscribe bus.

    subscribe(event_type, callback) : register a callback for one event type
    subscribe_all(callback)         : register a callback for every event type
    publish(event)                  : invoke all matching subscribers, in order

    If a subscriber crashes, we log the error and keep going. 
    A broken listener (like a logging sink) should never be able 
    to take down the entire pipeline.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Subscriber]] = {}
        self._global_subscribers: List[Subscriber] = []

    def subscribe(self, event_type: EventType, callback: Subscriber) -> None:
        self._subscribers.setdefault(event_type, []).append(callback)

    def subscribe_all(self, callback: Subscriber) -> None:
        self._global_subscribers.append(callback)

    def publish(self, event: Event) -> None:
        for callback in self._subscribers.get(event.type, []):
            self._safe_call(callback, event)
        for callback in self._global_subscribers:
            self._safe_call(callback, event)

    @staticmethod
    def _safe_call(callback: Subscriber, event: Event) -> None:
        try:
            callback(event)
        except Exception:
            logger.exception(
                "EventBus subscriber raised while handling %s — ignored.", event.type
            )
