"""Runtime for operators and sinks.

This provides the execution environment for operators to react to events
and emit outputs that are routed to sinks.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Iterable, List, Optional

from pipeline.events import CallContext, CallEvent, OperatorOutput

if TYPE_CHECKING:
    from operators.base import Operator
    from sinks.base import Sink


@dataclass
class _OperatorRunner:
    """Internal wrapper for running an operator with its event queue."""

    operator: "Operator"
    queue: asyncio.Queue[CallEvent]
    task: Optional[asyncio.Task[None]] = None


class PipelineRuntime:
    """Runtime that distributes events to operators and routes outputs to sinks.

    The runtime manages:
    - Event distribution to all registered operators
    - Output routing from operators to appropriate sinks
    - A special "speak" output type that triggers TTS injection

    Usage:
        runtime = PipelineRuntime(
            ctx=call_context,
            operators=[KeywordOperator()],
            sinks=[StdoutSink(), WebhookSink(url)],
        )
        await runtime.start()

        # Publish events from the pipecat observer
        await runtime.publish(UserTranscriptEvent(text="hello"))

        # When done
        await runtime.stop()
    """

    def __init__(
        self,
        ctx: CallContext,
        *,
        operators: Iterable["Operator"],
        sinks: Iterable["Sink"],
    ):
        self._ctx = ctx
        self._sinks: List["Sink"] = list(sinks)
        self._runners: List[_OperatorRunner] = [
            _OperatorRunner(operator=op, queue=asyncio.Queue(maxsize=200)) for op in operators
        ]

        # Queue for "speak" outputs - consumed by the observer to inject TTS
        self._speak_queue: asyncio.Queue[str] = asyncio.Queue()
        # Queue for "system_prompt" outputs - consumed by the observer to update LLM context
        self._system_prompt_queue: asyncio.Queue[dict] = asyncio.Queue()
        # Queue for "end_call" outputs - consumed by the observer to trigger hangup
        self._end_call_queue: asyncio.Queue[bool] = asyncio.Queue()
        self._started = False

    @property
    def context(self) -> CallContext:
        """Get the current call context."""
        return self._ctx

    async def start(self) -> None:
        """Start all operator tasks."""
        if self._started:
            return
        self._started = True
        for runner in self._runners:
            runner.task = asyncio.create_task(self._run_operator(runner))

    async def stop(self) -> None:
        """Stop all operator tasks."""
        for runner in self._runners:
            if runner.task:
                runner.task.cancel()
        for runner in self._runners:
            if runner.task:
                try:
                    await runner.task
                except asyncio.CancelledError:
                    pass
        self._started = False

    async def publish(self, event: CallEvent) -> None:
        """Publish an event to all operators.

        Events are queued to each operator. If an operator's queue is full,
        the event is dropped for that operator (non-blocking).
        """
        for runner in self._runners:
            try:
                runner.queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop event if operator can't keep up
                continue

    async def emit(self, output: OperatorOutput) -> None:
        """Handle output from an operator.

        Routes outputs to appropriate handlers:
        - "speak": Queues text for TTS injection
        - "system_prompt": Queues for LLM context update
        - "end_call": Queues signal to terminate the call
        - Other kinds: Sent to all sinks
        """
        if output.kind == "speak":
            text = output.payload.get("text", "")
            if text:
                await self._speak_queue.put(text)
            return

        if output.kind == "system_prompt":
            # Queue the system prompt for observer to update LLM context
            await self._system_prompt_queue.put(output.payload)
            # Also route to sinks for logging
        
        if output.kind == "end_call":
            # Queue the end call signal for observer to trigger hangup
            await self._end_call_queue.put(True)
            # Also route to sinks for logging

        # Route to sinks
        for sink in self._sinks:
            try:
                await sink.handle(output, self._ctx)
            except Exception:
                # Best-effort: sinks should not crash the pipeline
                pass

    async def next_speak_text(self) -> str:
        """Get the next text to speak (blocking).

        Called by the observer to get text for TTS injection.
        """
        return await self._speak_queue.get()

    def has_pending_speak(self) -> bool:
        """Check if there's pending text to speak."""
        return not self._speak_queue.empty()

    async def next_system_prompt(self) -> dict:
        """Get the next system prompt payload (blocking).

        Called by the observer to update LLM context.
        """
        return await self._system_prompt_queue.get()

    def has_pending_system_prompt(self) -> bool:
        """Check if there's a pending system prompt update."""
        return not self._system_prompt_queue.empty()

    async def next_end_call(self) -> bool:
        """Get the next end call signal (blocking).
        
        Called by the observer to trigger call termination.
        """
        return await self._end_call_queue.get()
    
    def has_pending_end_call(self) -> bool:
        """Check if there's a pending end call signal."""
        return not self._end_call_queue.empty()

    async def _run_operator(self, runner: _OperatorRunner) -> None:
        """Run loop for a single operator."""

        async def emit_callback(output: OperatorOutput) -> None:
            await self.emit(output)

        while True:
            event = await runner.queue.get()
            try:
                await runner.operator.handle_event(event, emit_callback, self._ctx)
            except Exception as exc:
                # Log operator errors but don't crash
                await self.emit(
                    OperatorOutput(
                        kind="log",
                        payload={
                            "level": "error",
                            "msg": "operator_error",
                            "operator": runner.operator.name,
                            "error": str(exc),
                        },
                    )
                )
