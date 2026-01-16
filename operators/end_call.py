"""End call operator - detects closing messages and signals call termination.

This operator listens for closing messages from the assistant and triggers
an automatic hangup after the closing is spoken.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from loguru import logger

from operators.base import Operator
from pipeline.events import (
    AssistantTextEvent,
    CallContext,
    CallEvent,
    CallLifecycleEvent,
    OperatorOutput,
)


class EndCallOperator(Operator):
    """Detects closing messages and signals call termination.
    
    This operator:
    1. Captures the business's closing template from config (if set)
    2. Listens for assistant text that matches closing patterns
    3. After the closing message is fully spoken, emits an "end_call" output
    
    The closing is detected by matching against:
    - The business's custom closing template (from call flow settings)
    - Common closing phrases used by the LLM
    
    Usage:
        operators = [
            BusinessContextOperator(),
            EndCallOperator(),
            # ... other operators
        ]
    """
    
    name = "end_call"
    
    # Common closing phrases that indicate the call should end
    # These are patterns the LLM is instructed to use in the system prompt
    # NOTE: "thank you for calling" and "thanks for calling" are NOT included
    # because they're commonly used in greetings, not just closings
    CLOSING_PATTERNS = [
        "goodbye",
        "have a great day",
        "have a good day",
        "have a wonderful day",
        "we'll be in touch",
        "we will be in touch",
        "take care",
        "talk to you soon",
        "bye for now",
    ]
    
    def __init__(self) -> None:
        self._should_end_after_speech = False
        self._closing_template: Optional[str] = None
        self._pending_end = False
        self._accumulated_text = ""  # Accumulate text during bot speaking
        self._is_speaking = False

    async def handle_event(
        self,
        event: CallEvent,
        emit: Callable[[OperatorOutput], Awaitable[None]],
        ctx: CallContext,
    ) -> None:
        """Handle call events and detect closing messages."""

        # Capture closing template from system_prompt output
        # This is emitted by BusinessContextOperator when config loads
        if isinstance(event, CallLifecycleEvent) and event.kind == "config_loaded":
            self._closing_template = event.payload.get("closing_template", "")
            if self._closing_template:
                logger.debug(f"EndCallOperator: captured closing template: {self._closing_template[:50]}...")

        # Track bot speaking state
        if isinstance(event, CallLifecycleEvent) and event.kind == "bot_speaking_start":
            self._is_speaking = True
            self._accumulated_text = ""  # Reset for new utterance

        # Accumulate assistant text during speaking
        # Text events come word-by-word, so we need to build up the full message
        if isinstance(event, AssistantTextEvent):
            self._accumulated_text += event.text + " "
            # Check accumulated text for closing patterns
            if self._is_closing_message(self._accumulated_text):
                if not self._should_end_after_speech:
                    logger.info(f"EndCallOperator: MATCH! detected closing in: '{self._accumulated_text.strip()}'")
                    self._should_end_after_speech = True

        # After bot stops speaking, check if we should end the call
        if isinstance(event, CallLifecycleEvent) and event.kind == "bot_speaking_stop":
            self._is_speaking = False
            logger.debug(f"EndCallOperator: bot stopped, accumulated: '{self._accumulated_text.strip()[:80]}...'")
            logger.info(f"EndCallOperator: bot_speaking_stop, should_end={self._should_end_after_speech}, pending={self._pending_end}")
            if self._should_end_after_speech and not self._pending_end:
                logger.info("EndCallOperator: closing spoken, signaling call end")
                self._pending_end = True
                await emit(OperatorOutput(kind="end_call", payload={}))
                self._should_end_after_speech = False
            self._accumulated_text = ""  # Reset after speaking ends
    
    def _is_closing_message(self, text: str) -> bool:
        """Check if text contains a closing message.
        
        Matches against:
        1. The business's custom closing template (highest priority)
        2. Common closing patterns
        
        Args:
            text: The assistant's message text
            
        Returns:
            True if this appears to be a closing message
        """
        text_lower = text.lower()
        
        # Check custom closing template first
        if self._closing_template:
            # Match if the closing template is substantially contained in the text
            template_lower = self._closing_template.lower()
            # Check for significant overlap (at least 60% of template words)
            template_words = set(template_lower.split())
            text_words = set(text_lower.split())
            if template_words:
                overlap = len(template_words & text_words) / len(template_words)
                if overlap >= 0.6:
                    return True
        
        # Check common closing patterns
        for pattern in self.CLOSING_PATTERNS:
            if pattern in text_lower:
                return True
        
        return False
