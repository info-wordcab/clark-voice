"""Transcript operator - emits transcript updates to database sink.

This operator listens for transcription events and assistant text events,
then emits transcript_update outputs for the DatabaseSink to persist.
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable, Optional

from loguru import logger

from operators.base import Operator
from pipeline.events import (
    AssistantTextEvent,
    CallContext,
    CallEvent,
    CallLifecycleEvent,
    OperatorOutput,
    UserTranscriptEvent,
)


class TranscriptOperator(Operator):
    """Emits transcript updates for database persistence.

    This operator:
    1. Tracks call start time for timestamp calculation
    2. Listens for UserTranscriptEvent (final transcriptions only)
    3. Listens for AssistantTextEvent
    4. Emits transcript_update outputs with role, text, and timestamp_ms

    The DatabaseSink will receive these outputs and append them
    to both the plain transcript field and the structured transcript_messages array.

    Usage:
        operators = [
            TranscriptOperator(),
            # ... other operators
        ]
    """

    name = "transcript"

    def __init__(self, include_interim: bool = False) -> None:
        """Initialize the transcript operator.

        Args:
            include_interim: If True, also emit interim transcriptions.
                            Default is False (only final transcriptions).
        """
        self._include_interim = include_interim
        self._call_start_time: Optional[float] = None

    def _get_timestamp_ms(self) -> int:
        """Get current timestamp in milliseconds relative to call start."""
        if self._call_start_time is None:
            return 0
        return int((time.time() - self._call_start_time) * 1000)

    async def handle_event(
        self,
        event: CallEvent,
        emit: Callable[[OperatorOutput], Awaitable[None]],
        ctx: CallContext,
    ) -> None:
        """Handle transcription and assistant text events."""

        # Track call start time for timestamp calculation
        if isinstance(event, CallLifecycleEvent) and event.kind == "connected":
            self._call_start_time = time.time()
            logger.debug("TranscriptOperator: call connected, tracking timestamps")

        if isinstance(event, UserTranscriptEvent):
            # Only emit final transcriptions unless configured otherwise
            if not event.is_final and not self._include_interim:
                return

            if not event.text.strip():
                return

            await emit(
                OperatorOutput(
                    kind="transcript_update",
                    payload={
                        "role": "user",
                        "text": event.text.strip(),
                        "is_final": event.is_final,
                        "timestamp_ms": self._get_timestamp_ms(),
                    },
                )
            )
            logger.debug(f"Transcript [user]: {event.text[:50]}...")

        elif isinstance(event, AssistantTextEvent):
            if not event.text.strip():
                return

            await emit(
                OperatorOutput(
                    kind="transcript_update",
                    payload={
                        "role": "assistant",
                        "text": event.text.strip(),
                        "timestamp_ms": self._get_timestamp_ms(),
                    },
                )
            )
            logger.debug(f"Transcript [assistant]: {event.text[:50]}...")
