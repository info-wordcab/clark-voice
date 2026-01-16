"""Webhook sink for sending operator outputs to Django backend."""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Optional, Set

import httpx
from loguru import logger

from pipeline.events import CallContext, OperatorOutput
from sinks.base import Sink

# Global reference to webhook URL for recording uploads
WEBHOOK_BASE_URL = os.getenv("DJANGO_INTERNAL_URL", "http://localhost:8000")


async def upload_recording_to_django(
    call_sid: str,
    audio_data: bytes,
    sample_rate: int,
    num_channels: int,
) -> None:
    """Upload call recording to Django for storage.

    This is called by the AudioBufferProcessor's on_audio_data handler
    when the call ends. The audio is uploaded to Django which stores it
    in Tigris (production) or local filesystem (development).

    Args:
        call_sid: The call ID (Twilio CallSid)
        audio_data: Raw WAV audio bytes
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels
    """
    if not call_sid:
        logger.warning("Cannot upload recording: no call_sid provided")
        return

    if not audio_data or len(audio_data) == 0:
        logger.warning(f"Cannot upload recording for {call_sid}: no audio data")
        return

    # Convert audio to WAV format
    from utils.audio import create_wav_bytes

    wav_data = create_wav_bytes(audio_data, sample_rate, num_channels)

    # Base64 encode for JSON transport
    audio_b64 = base64.b64encode(wav_data).decode("utf-8")

    # Calculate duration for logging
    from utils.audio import get_audio_duration

    duration = get_audio_duration(audio_data, sample_rate, num_channels)

    logger.info(
        f"Uploading recording for call {call_sid}: "
        f"{len(wav_data) / 1024:.1f}KB, {duration:.1f}s, "
        f"{sample_rate}Hz, {num_channels}ch"
    )

    url = f"{WEBHOOK_BASE_URL}/api/telephony/pipecat/recording"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                json={
                    "call_id": call_sid,
                    "audio_data": audio_b64,
                    "sample_rate": sample_rate,
                    "num_channels": num_channels,
                    "duration": duration,
                    "format": "wav",
                },
            )

            if response.status_code == 200:
                logger.info(f"Recording uploaded for call {call_sid}")
            else:
                logger.error(
                    f"Recording upload failed for {call_sid}: "
                    f"status {response.status_code}: {response.text}"
                )
    except httpx.TimeoutException:
        logger.error(f"Recording upload timeout for {call_sid}")
    except Exception as e:
        logger.error(f"Recording upload error for {call_sid}: {e}")


class WebhookSink(Sink):
    """Sends operator outputs to Django's /api/telephony/pipecat/event endpoint.

    This sink formats payloads to match what Django expects:
    - event_type: The type of event (call_ended, call_summary, etc.)
    - call_id: The Twilio CallSid
    - Additional fields depending on event type

    Webhooks are sent as fire-and-forget to ensure they complete even if
    the pipeline is shutting down. This is critical for call_ended events
    which trigger GPT-4o analysis in Django.

    Supported event types from operators:
    - call_ended: Final call data (transcript, duration)
    - call_summary: AI-generated summary and analysis

    Args:
        url: The webhook URL (e.g., http://clark-web.internal:8000/api/telephony/pipecat/event)
        timeout: Request timeout in seconds (default: 30.0)
    """

    # Event types that should be sent to Django
    DJANGO_EVENT_TYPES = {"call_ended", "call_summary"}

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self._url = url
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout
        self._pending_tasks: Set[asyncio.Task] = set()

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialization of HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def handle(self, output: OperatorOutput, ctx: CallContext) -> None:
        """Handle operator output and send to Django if appropriate.

        Webhooks are sent as fire-and-forget background tasks to ensure
        they complete even if the pipeline shuts down.
        """
        # Skip if no call_sid (can't update a call without knowing which one)
        if not ctx.call_sid:
            logger.debug(f"Skipping webhook for {output.kind}: no call_sid in context")
            return

        # Only send specific event types to Django
        if output.kind not in self.DJANGO_EVENT_TYPES:
            logger.debug(f"Skipping webhook for {output.kind}: not a Django event type")
            return

        # Build payload in Django's expected format
        payload = self._build_django_payload(output, ctx)

        # Fire and forget - create background task so it completes
        # even if the calling operator task is cancelled
        task = asyncio.create_task(self._send_webhook(payload, output.kind, ctx.call_sid))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _send_webhook(self, payload: dict, event_type: str, call_sid: str) -> None:
        """Send the webhook request (runs in background)."""
        try:
            client = await self._get_client()
            response = await client.post(self._url, json=payload)

            if response.status_code == 200:
                logger.info(f"Webhook sent: {event_type} for call {call_sid}")
            else:
                logger.warning(
                    f"Webhook failed: {event_type} for call {call_sid} "
                    f"- status {response.status_code}: {response.text}"
                )
        except httpx.TimeoutException:
            logger.error(f"Webhook timeout: {event_type} for call {call_sid}")
        except Exception as e:
            logger.error(f"Webhook error: {event_type} for call {call_sid} - {e}")

    def _build_django_payload(self, output: OperatorOutput, ctx: CallContext) -> dict:
        """Build payload in the format Django's /api/telephony/pipecat/event expects."""

        # Base payload with event_type and call_id
        payload = {
            "event_type": output.kind,
            "call_id": ctx.call_sid,
        }

        # Add event-specific fields from output.payload
        if output.kind == "call_ended":
            # Include transcript_messages for Django to save and analyze
            payload.update(
                {
                    "duration": output.payload.get("duration", 0),
                    "transcript_messages": output.payload.get("transcript_messages", []),
                }
            )

        elif output.kind == "call_summary":
            payload.update(
                {
                    "summary": output.payload.get("summary", ""),
                    "sentiment": output.payload.get("sentiment", ""),
                    "intent": output.payload.get("intent", ""),
                }
            )

        return payload

    async def close(self) -> None:
        """Close the HTTP client, waiting for pending webhooks to complete."""
        # Wait for any pending webhook tasks to complete (with timeout)
        if self._pending_tasks:
            logger.info(f"Waiting for {len(self._pending_tasks)} pending webhooks...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._pending_tasks, return_exceptions=True),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for pending webhooks")

        if self._client:
            await self._client.aclose()
            self._client = None
