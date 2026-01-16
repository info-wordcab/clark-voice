"""Call lifecycle operator - emits call_ended events to webhook sink.

This operator listens for call lifecycle events and emits appropriate
outputs to notify the Django backend via webhook.
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable

from loguru import logger

from operators.base import Operator
from pipeline.events import (
    CallContext,
    CallEvent,
    CallLifecycleEvent,
    OperatorOutput,
)


class CallLifecycleOperator(Operator):
    """Tracks call lifecycle and emits call_ended events.

    This operator:
    1. Tracks call start time
    2. On call end, emits call_ended with duration and clean transcript

    The transcript comes from LLMContext (via the ended event payload),
    which has properly aggregated user/assistant messages without garbling.

    The call_ended output is picked up by:
    - WebhookSink: sends to Django to trigger GPT-4o analysis
    - DatabaseSink: writes transcript to database

    Usage:
        operators = [
            CallLifecycleOperator(),
            # ... other operators
        ]
    """

    name = "call_lifecycle"

    def __init__(self) -> None:
        self._call_start_time: float = 0

    async def handle_event(
        self,
        event: CallEvent,
        emit: Callable[[OperatorOutput], Awaitable[None]],
        ctx: CallContext,
    ) -> None:
        """Handle call events and emit lifecycle outputs."""

        if isinstance(event, CallLifecycleEvent):
            if event.kind == "connected":
                # Call started - record start time
                self._call_start_time = ctx.call_start_time or time.time()
                logger.info(f"Call started: {ctx.call_sid}")

            elif event.kind == "ended":
                # Call ended - emit call_ended with duration and transcript
                # Transcript comes from LLMContext (clean, no garbling)
                duration = int(time.time() - self._call_start_time)
                transcript_messages = event.payload.get("transcript_messages", [])

                await emit(
                    OperatorOutput(
                        kind="call_ended",
                        payload={
                            "duration": duration,
                            "transcript_messages": transcript_messages,
                        },
                    )
                )
                logger.info(
                    f"Call ended: {ctx.call_sid} (duration: {duration}s, "
                    f"{len(transcript_messages)} messages)"
                )
