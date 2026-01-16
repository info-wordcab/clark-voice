"""Stdout sink for logging operator outputs to console."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from pipeline.events import CallContext, OperatorOutput
from sinks.base import Sink


class StdoutSink(Sink):
    """Logs operator outputs to stdout.
    
    Formats outputs with timestamps and call context for easy debugging.
    """
    
    async def handle(self, output: OperatorOutput, ctx: CallContext) -> None:
        now = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        
        # Build payload with context
        payload = dict(output.payload)
        if ctx.call_sid:
            payload["call_sid"] = ctx.call_sid
        
        payload_str = json.dumps(payload, default=str)
        print(f"[{now}] {output.kind}: {payload_str}")
