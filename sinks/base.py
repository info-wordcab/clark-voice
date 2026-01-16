"""Base class for sinks.

Sinks receive operator outputs and perform actions like logging or
sending webhooks.
"""

from __future__ import annotations

from pipeline.events import CallContext, OperatorOutput


class Sink:
    """Base class for output sinks.
    
    Sinks receive OperatorOutput and perform side effects like:
    - Logging to console
    - Sending webhooks
    - Writing to databases
    - etc.
    
    To create a custom sink:
    
        class MySink(Sink):
            async def handle(self, output, ctx):
                if output.kind == "my_event":
                    # Do something with the output
                    pass
    """
    
    async def handle(self, output: OperatorOutput, ctx: CallContext) -> None:
        """Handle an operator output.
        
        Override this method to implement custom output handling.
        
        Args:
            output: The output from an operator
            ctx: Current call context with metadata
        """
        pass
