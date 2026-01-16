"""Example operator demonstrating keyword detection.

This operator watches for specific keywords in user speech and triggers
actions like webhook notifications or injected assistant speech.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from operators.base import Operator
from pipeline.events import (
    CallContext,
    CallEvent,
    CallLifecycleEvent,
    OperatorOutput,
    UserTranscriptEvent,
)


class KeywordOperator(Operator):
    """Detects keywords in user speech and triggers actions.
    
    This is an example operator that demonstrates:
    - Reacting to specific user speech patterns
    - Sending webhook notifications
    - Injecting assistant speech mid-conversation
    
    Customize the keywords and actions for your use case.
    """
    
    name = "keyword_detector"

    # Keywords to detect and their associated webhook event names
    KEYWORDS = {
        "address": "address_mentioned",
        "appointment": "appointment_requested",
        "manager": "manager_requested",
        "complaint": "complaint_detected",
        "emergency": "emergency_detected",
        "callback": "callback_requested",
    }

    async def handle_event(
        self,
        event: CallEvent,
        emit: Callable[[OperatorOutput], Awaitable[None]],
        ctx: CallContext,
    ) -> None:
        # Log call lifecycle events
        if isinstance(event, CallLifecycleEvent):
            if event.kind == "started":
                await emit(
                    OperatorOutput(
                        kind="log",
                        payload={
                            "msg": "Call started",
                            "call_sid": ctx.call_sid,
                            "from": ctx.from_number,
                            "to": ctx.to_number,
                        },
                    )
                )
            elif event.kind == "ended":
                await emit(
                    OperatorOutput(
                        kind="log",
                        payload={"msg": "Call ended", "call_sid": ctx.call_sid},
                    )
                )
            return

        # React to final user transcripts
        if isinstance(event, UserTranscriptEvent) and event.is_final:
            text_lower = event.text.lower()
            
            for keyword, event_name in self.KEYWORDS.items():
                if keyword in text_lower:
                    # Send webhook notification
                    await emit(
                        OperatorOutput(
                            kind="webhook_event",
                            payload={
                                "event": event_name,
                                "keyword": keyword,
                                "transcript": event.text,
                                "call_sid": ctx.call_sid,
                            },
                        )
                    )
                    
                    # Log the detection
                    await emit(
                        OperatorOutput(
                            kind="log",
                            payload={
                                "msg": f"Keyword detected: {keyword}",
                                "event": event_name,
                            },
                        )
                    )
                    
                    # Optionally inject speech for certain keywords
                    if keyword == "emergency":
                        await emit(
                            OperatorOutput(
                                kind="speak",
                                payload={
                                    "text": "I understand this is urgent. Let me connect you with someone who can help right away."
                                },
                            )
                        )
                    elif keyword == "manager":
                        await emit(
                            OperatorOutput(
                                kind="speak",
                                payload={
                                    "text": "I'll make a note that you'd like to speak with a manager. Let me see how I can help in the meantime."
                                },
                            )
                        )
