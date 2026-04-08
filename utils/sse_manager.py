import asyncio
import json
from collections import defaultdict
from typing import AsyncGenerator


class SSEManager:
    """
    Event store + broadcaster for SSE streams.

    Uses a list buffer so late subscribers can replay all events
    from the beginning (avoids queue-drain race conditions).
    """

    def __init__(self):
        self._events: dict[str, list[dict]] = defaultdict(list)

    async def publish(self, run_id: str, event: dict):
        self._events[run_id].append(event)

    async def subscribe(
        self, run_id: str
    ) -> AsyncGenerator[str, None]:
        cursor = 0
        deadline = 300  # 5-minute max stream duration
        elapsed = 0.0

        while elapsed < deadline:
            events = self._events[run_id]

            # Yield all buffered events since last cursor position
            while cursor < len(events):
                event = events[cursor]
                cursor += 1
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "pipeline_complete":
                    self.cleanup(run_id)
                    return

            # Poll every 100ms for new events
            await asyncio.sleep(0.1)
            elapsed += 0.1

    def cleanup(self, run_id: str):
        self._events.pop(run_id, None)


sse_manager = SSEManager()
