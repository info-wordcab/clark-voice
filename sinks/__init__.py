from sinks.base import Sink
from sinks.database import DatabaseSink
from sinks.stdout import StdoutSink
from sinks.webhook import WebhookSink

__all__ = ["Sink", "DatabaseSink", "StdoutSink", "WebhookSink"]
