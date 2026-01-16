"""Database sink for direct Postgres writes via asyncpg.

This sink writes call events directly to the shared Postgres database,
bypassing Django for low-latency writes during active calls.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from loguru import logger

from pipeline.events import CallContext, OperatorOutput
from sinks.base import Sink

# Try to import asyncpg, but make it optional
try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logger.warning("asyncpg not installed, DatabaseSink will be disabled")


class DatabaseSink(Sink):
    """Writes call events directly to Postgres.

    This sink handles:
    - call_ended: Saves clean transcript and final call data

    The transcript comes from LLMContext at call end, which provides
    properly aggregated messages without the garbling issues of
    real-time TTSTextFrame buffering.

    Configuration:
        Set DATABASE_URL environment variable to the Postgres connection string.
        In production on Fly.io, this is provided automatically.

    Usage:
        sinks = [
            DatabaseSink(),
            WebhookSink(url),
        ]
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 5,
    ) -> None:
        if not ASYNCPG_AVAILABLE:
            self._pool = None
            self._enabled = False
            return

        self._database_url = database_url or os.getenv("DATABASE_URL", "")
        self._min_connections = min_connections
        self._max_connections = max_connections
        self._pool: Optional[asyncpg.Pool] = None
        self._enabled = bool(self._database_url)

        if not self._enabled:
            logger.warning("DATABASE_URL not set, DatabaseSink disabled")

    async def _get_pool(self) -> Optional[asyncpg.Pool]:
        """Lazy initialization of connection pool."""
        if not self._enabled:
            return None

        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    self._database_url,
                    min_size=self._min_connections,
                    max_size=self._max_connections,
                )
                logger.info("Database connection pool created")
            except Exception as e:
                logger.error(f"Failed to create database pool: {e}")
                self._enabled = False
                return None

        return self._pool

    async def handle(self, output: OperatorOutput, ctx: CallContext) -> None:
        """Handle operator outputs and write to database."""
        if not self._enabled:
            return

        pool = await self._get_pool()
        if not pool:
            return

        try:
            if output.kind == "call_ended":
                await self._handle_call_ended(pool, output, ctx)
        except Exception as e:
            logger.error(f"Database error in {output.kind}: {e}")

    async def _handle_call_ended(
        self,
        pool: asyncpg.Pool,
        output: OperatorOutput,
        ctx: CallContext,
    ) -> None:
        """Update call with final status and clean transcript.

        The transcript_messages come from LLMContext, which has properly
        aggregated user/assistant messages without garbling.
        """
        if not ctx.call_sid:
            return

        duration = output.payload.get("duration", 0)
        transcript_messages = output.payload.get("transcript_messages", [])

        # Build plain text transcript for backwards compatibility
        plain_transcript = "\n".join(
            f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')}"
            for msg in transcript_messages
        )

        async with pool.acquire() as conn:
            # Update call record with transcript and status
            await conn.execute(
                """
                UPDATE calls 
                SET status = 'completed',
                    duration = $1,
                    transcript = $2,
                    transcript_messages = $3::jsonb,
                    ended_at = $4,
                    updated_at = $4
                WHERE id = $5
                """,
                duration,
                plain_transcript,
                json.dumps(transcript_messages),
                datetime.utcnow(),
                ctx.call_sid,
            )

        logger.info(
            f"Saved call {ctx.call_sid}: {len(transcript_messages)} messages, {duration}s duration"
        )

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")
