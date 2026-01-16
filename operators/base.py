"""Base class for operators.

Operators react to call events and can emit outputs that trigger
actions like webhooks or injected speech.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from pipeline.events import CallContext, CallEvent, OperatorOutput


class Operator:
    """Base class for event-handling operators.
    
    Operators receive all call events and can emit outputs to trigger actions.
    
    To create a custom operator:
    
        class MyOperator(Operator):
            name = "my_operator"
            
            async def handle_event(self, event, emit, ctx):
                if isinstance(event, UserTranscriptEvent):
                    # React to user speech
                    if "help" in event.text.lower():
                        await emit(OperatorOutput(
                            kind="webhook_event",
                            payload={"event": "help_requested"}
                        ))
    
    Attributes:
        name: Identifier for this operator (used in logs)
    """
    
    name: str = "operator"

    async def handle_event(
        self,
        event: CallEvent,
        emit: Callable[[OperatorOutput], Awaitable[None]],
        ctx: CallContext,
    ) -> None:
        """Handle an incoming event.
        
        Override this method to implement custom event handling.
        
        Args:
            event: The event to handle (CallLifecycleEvent, UserTranscriptEvent, etc.)
            emit: Callback to emit outputs (webhook_event, speak, log)
            ctx: Current call context with metadata
        """
        pass
