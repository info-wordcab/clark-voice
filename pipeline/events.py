"""Event types for the receptionist pipeline.

These events flow through operators and can trigger actions like
webhook notifications or injected speech.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class CallContext:
    """Context information about the current call.

    This is populated as information becomes available during the call.
    """

    call_sid: Optional[str] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None

    # Business context (populated by BusinessContextOperator)
    business_id: Optional[str] = None
    business_name: Optional[str] = None

    # Internal tracking
    call_start_time: Optional[float] = None


@dataclass(frozen=True)
class CallLifecycleEvent:
    """Lifecycle events for the call.

    Kinds:
        - "connected": WebSocket connection established
        - "started": Call audio streaming began
        - "ended": Call ended
        - "user_speaking_start": User started speaking
        - "user_speaking_stop": User stopped speaking
        - "bot_speaking_start": Bot started speaking
        - "bot_speaking_stop": Bot stopped speaking
    """

    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UserTranscriptEvent:
    """User speech transcription event.

    Attributes:
        text: The transcribed text
        is_final: True if this is a final transcription, False if interim
    """

    text: str
    is_final: bool = True


@dataclass(frozen=True)
class AssistantTextEvent:
    """Assistant text output event.

    Emitted when the assistant generates text that will be spoken.
    """

    text: str


# Union type for all events that operators can receive
CallEvent = Union[CallLifecycleEvent, UserTranscriptEvent, AssistantTextEvent]


@dataclass(frozen=True)
class OperatorOutput:
    """Output from an operator that triggers actions.

    Kinds:
        - "log": Log message (sent to StdoutSink)
        - "webhook_event": Send to webhook URL (sent to WebhookSink)
        - "speak": Make the assistant speak (injects TTSSpeakFrame)

    Attributes:
        kind: The type of output
        payload: Data associated with the output
    """

    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
