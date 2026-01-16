"""Pipecat observer that bridges to the operator/sink system.

This observer watches pipecat frames and translates them to our event
types, then routes them through operators and sinks.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Iterable, List, Optional

from loguru import logger

from pipecat.services.cartesia.tts import GenerationConfig

# Style to emotion mapping for Cartesia TTS
# Each style maps to a primary emotion that sets the overall tone
STYLE_TO_EMOTION = {
    "friendly": "content",  # Warm, welcoming tone
    "direct": "determined",  # Straightforward, no-nonsense
    "professional": "neutral",  # Formal, business-like
    "empathetic": "sympathetic",  # Caring, understanding
}

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from pipeline.events import (
    AssistantTextEvent,
    CallContext,
    CallLifecycleEvent,
    UserTranscriptEvent,
)
from pipeline.runtime import PipelineRuntime

if TYPE_CHECKING:
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.pipeline.task import PipelineTask
    from pipecat.services.tts_service import TTSService

    from operators.base import Operator
    from sinks.base import Sink


class ReceptionistObserver(BaseObserver):
    """Observer that bridges pipecat frames to the operator/sink system.

    This observer:
    1. Watches pipecat frames (transcriptions, TTS text, speaking events)
    2. Translates them to our event types
    3. Routes events through operators
    4. Handles "speak" outputs by injecting TTSSpeakFrame
    5. Handles "system_prompt" outputs by updating LLM context

    Usage:
        observer = ReceptionistObserver(
            operators=[KeywordOperator()],
            sinks=[StdoutSink(), WebhookSink(url)],
        )

        # Set task and TTS after creation (needed for speak injection)
        observer.set_task(task)
        observer.set_tts(tts)
        observer.set_llm_context(messages)  # For dynamic system prompts

        # Add to pipeline task
        task = PipelineTask(pipeline, observers=[observer])
    """

    def __init__(
        self,
        operators: Iterable["Operator"],
        sinks: Iterable["Sink"],
    ):
        super().__init__()
        self._operators: List["Operator"] = list(operators)
        self._sinks: List["Sink"] = list(sinks)

        # These are set after construction
        self._task: Optional["PipelineTask"] = None
        self._tts: Optional["TTSService"] = None
        self._llm_messages: Optional[List[dict]] = None

        # Runtime is created per-call
        self._runtime: Optional[PipelineRuntime] = None
        self._speak_task: Optional[asyncio.Task] = None
        self._system_prompt_task: Optional[asyncio.Task] = None
        self._end_call_task: Optional[asyncio.Task] = None
        self._started = False

        # Event signaling that business config has been loaded
        # Used to delay greeting until voice/prompt are set
        self._config_ready_event: Optional[asyncio.Event] = None

        # Track speaking state to deduplicate events
        # Pipecat may send multiple frames from different processors
        self._bot_is_speaking = False
        self._user_is_speaking = False

        # Buffer for TTS text (used for closing detection only)
        # Transcript comes from LLMContext at call end
        self._tts_text_buffer = ""

    def set_task(self, task: "PipelineTask") -> None:
        """Set the pipeline task (needed for frame injection)."""
        self._task = task

    def set_tts(self, tts: "TTSService") -> None:
        """Set the TTS service (needed for speak injection)."""
        self._tts = tts

    def set_llm_context(self, messages: List[dict]) -> None:
        """Set the LLM messages list (for dynamic system prompt updates)."""
        self._llm_messages = messages

    @property
    def context(self) -> Optional[CallContext]:
        """Get the current call context."""
        return self._runtime.context if self._runtime else None

    async def start(self, call_metadata: Optional[dict] = None) -> None:
        """Start the observer and runtime.

        Args:
            call_metadata: Optional dict with call info (call_sid, from_number, to_number)
        """
        if self._started:
            return

        # Create config ready event for synchronization
        self._config_ready_event = asyncio.Event()

        # Create runtime with fresh context
        ctx = CallContext(call_start_time=time.time())

        # Populate context with call metadata if provided
        if call_metadata:
            ctx.call_sid = call_metadata.get("call_sid")
            ctx.from_number = call_metadata.get("from_number")
            ctx.to_number = call_metadata.get("to_number")

        self._runtime = PipelineRuntime(
            ctx,
            operators=self._operators,
            sinks=self._sinks,
        )

        await self._runtime.start()
        self._started = True

        # Start speak injection task
        self._speak_task = asyncio.create_task(self._process_speak_queue())

        # Start system prompt processing task
        self._system_prompt_task = asyncio.create_task(self._process_system_prompt_queue())

        # Start end call processing task
        self._end_call_task = asyncio.create_task(self._process_end_call_queue())

        # Publish connected event with call metadata
        await self._runtime.publish(
            CallLifecycleEvent(
                kind="connected",
                payload=call_metadata or {},
            )
        )
        logger.info("ReceptionistObserver started")

    async def wait_for_config(self, timeout: float = 3.0) -> bool:
        """Wait for business config to be loaded.

        This should be called before triggering the greeting to ensure
        the voice and system prompt are set.

        Args:
            timeout: Maximum seconds to wait for config

        Returns:
            True if config loaded, False if timeout
        """
        if self._config_ready_event is None:
            return False
        try:
            await asyncio.wait_for(self._config_ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Config load timeout after {timeout}s, using defaults")
            return False

    async def stop(self) -> None:
        """Stop the observer and runtime."""
        if not self._started:
            return

        # Publish ended event with clean transcript from LLMContext
        if self._runtime:
            # Extract conversation transcript from LLM context (excludes system prompt)
            transcript_messages = []
            if self._llm_messages:
                transcript_messages = [
                    {"role": msg.get("role"), "content": msg.get("content", "")}
                    for msg in self._llm_messages
                    if msg.get("role") in ("user", "assistant")
                ]

            await self._runtime.publish(
                CallLifecycleEvent(
                    kind="ended",
                    payload={"transcript_messages": transcript_messages},
                )
            )

            # Give operators time to process the ended event before cancelling
            # This ensures CallLifecycleOperator can emit call_ended to webhook
            await asyncio.sleep(0.3)

            await self._runtime.stop()

        # Cancel speak task
        if self._speak_task:
            self._speak_task.cancel()
            try:
                await self._speak_task
            except asyncio.CancelledError:
                pass

        # Cancel system prompt task
        if self._system_prompt_task:
            self._system_prompt_task.cancel()
            try:
                await self._system_prompt_task
            except asyncio.CancelledError:
                pass

        # Cancel end call task
        if self._end_call_task:
            self._end_call_task.cancel()
            try:
                await self._end_call_task
            except asyncio.CancelledError:
                pass

        self._started = False
        logger.info("ReceptionistObserver stopped")

    async def on_push_frame(self, data: FramePushed) -> None:
        """Handle frame push events from pipecat."""
        if not self._runtime:
            return

        frame = data.frame

        # Transcription events (user speech)
        # Note: We still emit these for operators that need real-time user transcript
        # The final clean transcript comes from LLMContext at call end
        if isinstance(frame, TranscriptionFrame):
            is_interim = isinstance(frame, InterimTranscriptionFrame)
            await self._runtime.publish(
                UserTranscriptEvent(
                    text=frame.text,
                    is_final=not is_interim,
                )
            )

        # Speaking events - use state tracking to deduplicate
        # Pipecat may send multiple frames from different processors
        elif isinstance(frame, UserStartedSpeakingFrame):
            if not self._user_is_speaking:
                self._user_is_speaking = True
                await self._runtime.publish(CallLifecycleEvent(kind="user_speaking_start"))
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._user_is_speaking:
                self._user_is_speaking = False
                await self._runtime.publish(CallLifecycleEvent(kind="user_speaking_stop"))
        elif isinstance(frame, BotStartedSpeakingFrame):
            if not self._bot_is_speaking:
                self._bot_is_speaking = True
                self._tts_text_buffer = ""  # Reset buffer for new utterance
                await self._runtime.publish(CallLifecycleEvent(kind="bot_speaking_start"))
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._bot_is_speaking:
                self._bot_is_speaking = False
                await self._runtime.publish(CallLifecycleEvent(kind="bot_speaking_stop"))

        # TTS text frames - emit for closing detection
        # EndCallOperator uses this to detect goodbye phrases and auto-hangup
        elif isinstance(frame, TTSTextFrame):
            if frame.text:
                self._tts_text_buffer += frame.text + " "
                await self._runtime.publish(AssistantTextEvent(text=frame.text))

    async def _process_speak_queue(self) -> None:
        """Process speak outputs by injecting TTSSpeakFrame."""
        from pipecat.frames.frames import TTSSpeakFrame

        while True:
            if not self._runtime:
                await asyncio.sleep(0.1)
                continue

            try:
                text = await self._runtime.next_speak_text()

                if self._tts and text:
                    logger.debug(f"Injecting speak: {text[:50]}...")
                    await self._tts.queue_frame(TTSSpeakFrame(text=text))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing speak queue: {e}")

    async def _process_system_prompt_queue(self) -> None:
        """Process system_prompt outputs by updating LLM context and TTS voice."""
        while True:
            if not self._runtime:
                await asyncio.sleep(0.1)
                continue

            try:
                payload = await self._runtime.next_system_prompt()

                business_name = payload.get("business_name", "Unknown")

                # Update TTS voice FIRST (before greeting)
                voice_id = payload.get("voice_id")
                if voice_id and self._tts:
                    logger.info(f"Setting TTS voice to: {voice_id}")
                    self._tts.set_voice(voice_id)

                # Update TTS emotion based on style
                style = payload.get("style", "friendly")
                emotion = STYLE_TO_EMOTION.get(style, "neutral")
                if self._tts:
                    logger.info(f"Setting TTS emotion to: {emotion} (style={style})")
                    self._tts.update_setting("generation_config", GenerationConfig(emotion=emotion))

                # Update system prompt
                system_prompt = payload.get("system_prompt", "")
                if system_prompt and self._llm_messages is not None:
                    # Update the system message (first message in the list)
                    if self._llm_messages and self._llm_messages[0].get("role") == "system":
                        old_prompt = self._llm_messages[0].get("content", "")[:50]
                        self._llm_messages[0]["content"] = system_prompt
                        logger.info(
                            f"Updated system prompt for {business_name} (was: {old_prompt}...)"
                        )
                    else:
                        # Insert a new system message at the beginning
                        self._llm_messages.insert(
                            0,
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                        )
                        logger.info(f"Inserted system prompt for {business_name}")

                    # Update the context if available (for tracking)
                    if self._runtime and self._runtime.context:
                        # Store business info in context for later use
                        self._runtime.context.business_id = payload.get("business_id")
                        self._runtime.context.business_name = business_name

                # Publish config_loaded event so other operators can react
                # (e.g., EndCallOperator needs the closing_template)
                await self._runtime.publish(
                    CallLifecycleEvent(
                        kind="config_loaded",
                        payload={
                            "business_id": payload.get("business_id"),
                            "business_name": business_name,
                            "closing_template": payload.get("closing_template", ""),
                        },
                    )
                )

                # Signal that config is ready (voice and prompt are set)
                if self._config_ready_event:
                    self._config_ready_event.set()
                    logger.info(f"Config ready for {business_name}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing system prompt queue: {e}")

    async def _process_end_call_queue(self) -> None:
        """Process end_call outputs by pushing EndFrame to terminate the call."""
        from pipecat.frames.frames import EndFrame

        while True:
            if not self._runtime:
                await asyncio.sleep(0.1)
                continue

            try:
                # Wait for end call signal
                await self._runtime.next_end_call()

                # Push EndFrame to terminate the call
                if self._task:
                    logger.info("Pushing EndFrame to terminate call")
                    await self._task.queue_frame(EndFrame())
                else:
                    logger.warning("Cannot end call: task not set")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing end call queue: {e}")
