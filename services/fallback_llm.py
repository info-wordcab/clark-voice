"""Fallback LLM service that wraps multiple providers with automatic failover.

This service wraps multiple LLM services and automatically fails over to the
next provider when a rate limit (429) error occurs. The failover is silent -
no user-facing indication that a switch occurred.

Usage:
    services = [primary_service, secondary_service, fallback_service]
    provider_names = ["sambanova", "google", "openai"]
    llm = FallbackLLMService(services, provider_names)
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import CancelFrame, ErrorFrame, Frame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, StartFrame, StopFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.llm_service import LLMService


class FallbackLLMService(LLMService):
    """LLM service that delegates to a primary service with automatic fallback.

    When the current service encounters a rate limit error (429), it switches
    to the next service in the chain silently.

    This works by wrapping the services and intercepting their output. When
    an error frame indicates a rate limit, the request is retried with the
    next service.
    """

    def __init__(
        self,
        services: List[LLMService],
        provider_names: List[str],
    ):
        """Initialize the fallback LLM service.

        Args:
            services: List of LLM services in priority order (primary first)
            provider_names: Names of providers for logging (same order as services)
        """
        super().__init__()

        self._services = services
        self._provider_names = provider_names
        self._current_index = 0
        self._last_error_provider: Optional[str] = None
        self._started = False

        logger.info(
            f"FallbackLLMService initialized with {len(services)} providers: "
            f"{', '.join(provider_names)}"
        )

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and propagate to all wrapped services.

        This ensures wrapped services receive the TaskManager and other
        required components from the pipeline.
        """
        await super().setup(setup)
        for service in self._services:
            await service.setup(setup)

    def link(self, processor):
        """Link to next processor and propagate to wrapped services.

        Wrapped services need to be linked so their output frames flow to
        the next processor in the pipeline.
        """
        super().link(processor)
        for service in self._services:
            service.link(processor)
    
    @property
    def current_service(self) -> LLMService:
        """Get the current active LLM service."""
        return self._services[self._current_index]
    
    @property
    def current_provider(self) -> str:
        """Get the name of the current provider."""
        return self._provider_names[self._current_index]
    
    def _switch_to_next(self) -> bool:
        """Switch to the next provider in the chain.
        
        Returns:
            True if switched successfully, False if no more providers
        """
        if self._current_index < len(self._services) - 1:
            old_provider = self.current_provider
            self._current_index += 1
            logger.warning(
                f"Switching LLM from {old_provider} to {self.current_provider} "
                f"due to rate limit"
            )
            return True
        else:
            logger.error("All LLM providers exhausted, no fallback available")
            return False
    
    def _is_rate_limit_error(self, frame: Frame) -> bool:
        """Check if a frame represents a rate limit error."""
        if isinstance(frame, ErrorFrame):
            error_str = str(frame.error).lower()
            # Check for common rate limit indicators
            if "429" in error_str or "rate" in error_str or "quota" in error_str:
                return True
        return False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frame by delegating to the current service.

        If the service returns a rate limit error, switches to the next
        service and retries.
        """
        # Handle StartFrame specially - propagate to all wrapped services
        if isinstance(frame, StartFrame):
            await super().process_frame(frame, direction)
            for service in self._services:
                await service.process_frame(frame, direction)
            self._started = True
            return

        # Handle CancelFrame and StopFrame - propagate to ALL services to stop any ongoing work
        if isinstance(frame, (CancelFrame, StopFrame)):
            await super().process_frame(frame, direction)
            for service in self._services:
                await service.process_frame(frame, direction)
            return

        # For other frames, delegate to current service
        # Store original frame for potential retry
        original_frame = frame

        # Try current service
        try:
            await self.current_service.process_frame(frame, direction)
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str or "quota" in error_str:
                # Rate limit - try to switch
                if self._switch_to_next():
                    # Retry with new service
                    logger.info(f"Retrying with {self.current_provider}")
                    await self.current_service.process_frame(original_frame, direction)
                else:
                    # No more fallbacks, re-raise
                    raise
            else:
                # Non-rate-limit error, re-raise
                raise

    async def start(self, frame: Frame):
        """Start all services."""
        await super().start(frame)
        for service in self._services:
            await service.start(frame)
        self._started = True

    async def stop(self, frame: Frame):
        """Stop all services."""
        for service in self._services:
            await service.stop(frame)
        await super().stop(frame)
    
    async def cleanup(self):
        """Cleanup all services."""
        for service in self._services:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
        await super().cleanup()
