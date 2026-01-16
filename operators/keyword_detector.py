"""Keyword detector operator - detects keywords/intents in transcript.

This operator listens for transcript events and emits keyword_detected outputs
when specific patterns are matched, with timestamps for audio marker positioning.
"""

from __future__ import annotations

import re
import time
from typing import Awaitable, Callable, Dict, List, Optional, Set

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


# Default keyword patterns for detection
DEFAULT_INTENT_PATTERNS = {
    "booking": [
        r"\b(book|schedule|appointment|reserve)\b",
        r"\b(available|availability)\b",
        r"\b(when can|what time)\b",
    ],
    "inquiry": [
        r"\b(how much|price|cost|rate)\b",
        r"\b(do you|can you|are you)\b",
        r"\b(information|info|details)\b",
    ],
    "urgent": [
        r"\b(urgent|emergency|asap|immediately)\b",
        r"\b(right now|right away)\b",
        r"\b(critical|serious)\b",
    ],
    "complaint": [
        r"\b(problem|issue|complaint|wrong)\b",
        r"\b(not working|broken|failed)\b",
        r"\b(unhappy|disappointed|frustrated)\b",
    ],
    "callback": [
        r"\b(call me back|call back|return my call)\b",
        r"\b(get back to me)\b",
    ],
}

DEFAULT_SENTIMENT_PATTERNS = {
    "positive": [
        r"\b(thank you|thanks|appreciate|great|wonderful|excellent)\b",
        r"\b(happy|pleased|satisfied)\b",
    ],
    "negative": [
        r"\b(angry|upset|frustrated|annoyed)\b",
        r"\b(terrible|awful|horrible|worst)\b",
        r"\b(never|won't|can't believe)\b",
    ],
}


class KeywordDetectorOperator(Operator):
    """Detects keywords and intents in transcript with timestamps.

    This operator:
    1. Tracks call start time for timestamp calculation
    2. Listens for UserTranscriptEvent (caller speech)
    3. Matches text against configurable keyword patterns
    4. Emits keyword_detected outputs with type, value, and timestamp_ms

    The DatabaseSink will receive these outputs and append them
    to the call's tags JSON array.

    Usage:
        operators = [
            KeywordDetectorOperator(),
            # ... other operators
        ]
    """

    name = "keyword_detector"

    def __init__(
        self,
        intent_patterns: Optional[Dict[str, List[str]]] = None,
        sentiment_patterns: Optional[Dict[str, List[str]]] = None,
        custom_keywords: Optional[List[str]] = None,
    ) -> None:
        """Initialize the keyword detector.

        Args:
            intent_patterns: Dict mapping intent names to regex patterns.
            sentiment_patterns: Dict mapping sentiment names to regex patterns.
            custom_keywords: List of custom keywords to detect (simple substring match).
        """
        self._intent_patterns = intent_patterns or DEFAULT_INTENT_PATTERNS
        self._sentiment_patterns = sentiment_patterns or DEFAULT_SENTIMENT_PATTERNS
        self._custom_keywords = custom_keywords or []
        self._call_start_time: Optional[float] = None
        self._detected_intents: Set[str] = set()  # Avoid duplicate detections

    def _get_timestamp_ms(self) -> int:
        """Get current timestamp in milliseconds relative to call start."""
        if self._call_start_time is None:
            return 0
        return int((time.time() - self._call_start_time) * 1000)

    def _compile_patterns(self, patterns: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for efficient matching."""
        compiled = {}
        for name, pattern_list in patterns.items():
            compiled[name] = [re.compile(p, re.IGNORECASE) for p in pattern_list]
        return compiled

    async def handle_event(
        self,
        event: CallEvent,
        emit: Callable[[OperatorOutput], Awaitable[None]],
        ctx: CallContext,
    ) -> None:
        """Handle transcript events and detect keywords."""

        # Track call start time
        if isinstance(event, CallLifecycleEvent) and event.kind == "started":
            self._call_start_time = time.time()
            self._detected_intents.clear()
            logger.debug("KeywordDetectorOperator: call started")

        # Only analyze user speech (caller)
        if isinstance(event, UserTranscriptEvent):
            if not event.is_final or not event.text.strip():
                return

            text = event.text.strip().lower()
            timestamp_ms = self._get_timestamp_ms()

            # Detect intents
            for intent_name, patterns in self._intent_patterns.items():
                if intent_name in self._detected_intents:
                    continue  # Already detected this intent

                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        self._detected_intents.add(intent_name)
                        await emit(
                            OperatorOutput(
                                kind="keyword_detected",
                                payload={
                                    "type": "intent",
                                    "value": intent_name,
                                    "confidence": 0.9,
                                    "detected_at_ms": timestamp_ms,
                                },
                            )
                        )
                        logger.info(f"Detected intent: {intent_name} at {timestamp_ms}ms")
                        break

            # Detect sentiment (can have multiple)
            for sentiment_name, patterns in self._sentiment_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        await emit(
                            OperatorOutput(
                                kind="keyword_detected",
                                payload={
                                    "type": "sentiment",
                                    "value": sentiment_name,
                                    "confidence": 0.8,
                                    "detected_at_ms": timestamp_ms,
                                },
                            )
                        )
                        logger.debug(f"Detected sentiment: {sentiment_name}")
                        break

            # Detect custom keywords
            for keyword in self._custom_keywords:
                if keyword.lower() in text:
                    await emit(
                        OperatorOutput(
                            kind="keyword_detected",
                            payload={
                                "type": "keyword",
                                "value": keyword,
                                "confidence": 1.0,
                                "detected_at_ms": timestamp_ms,
                            },
                        )
                    )
                    logger.debug(f"Detected custom keyword: {keyword}")
