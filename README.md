# AI Receptionist

A pipecat-based AI receptionist that handles phone calls with:
- **AssemblyAI** for Speech-to-Text
- **SambaNova** (Llama-4) for LLM
- **Inworld AI** for Text-to-Speech

## Features

- Real-time voice conversation over Twilio or WebRTC
- Keyword detection with webhook notifications
- Mid-call speech injection (operators can make the bot speak)
- Extensible operator/sink system for custom event handling

## Quick Start

### 1. Install Dependencies

```bash
cd receptionist
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Locally (WebRTC)

```bash
python bot.py -t webrtc
```

Open http://localhost:7860/client in your browser.

### 4. Run with Twilio

```bash
# Start ngrok tunnel
ngrok http 7860

# Run bot with Twilio transport
python bot.py -t twilio -x your-subdomain.ngrok.io
```

Configure your Twilio phone number webhook to `https://your-subdomain.ngrok.io/`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Pipecat Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  Transport Input → AssemblyAI STT → LLM Context Aggregator  │
│        → SambaNova LLM → Inworld TTS → Transport Output     │
├─────────────────────────────────────────────────────────────┤
│                   ReceptionistObserver                       │
│    Watches frames and translates to operator events          │
│                          ↓                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Operators  │ → │   Runtime   │ → │    Sinks    │      │
│  │ (react to   │    │ (routes     │    │ (webhook,   │      │
│  │  events)    │    │  outputs)   │    │  logging)   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                                                    │
│         └── "speak" outputs → TTSSpeakFrame injection       │
└─────────────────────────────────────────────────────────────┘
```

## Customization

### Adding Operators

Operators react to call events (transcriptions, lifecycle, etc.):

```python
# operators/my_operator.py
from operators.base import Operator
from pipeline.events import UserTranscriptEvent, OperatorOutput

class MyOperator(Operator):
    name = "my_operator"
    
    async def handle_event(self, event, emit, ctx):
        if isinstance(event, UserTranscriptEvent) and event.is_final:
            if "schedule" in event.text.lower():
                # Send webhook
                await emit(OperatorOutput(
                    kind="webhook_event",
                    payload={"event": "scheduling_requested"}
                ))
                # Make the bot speak
                await emit(OperatorOutput(
                    kind="speak",
                    payload={"text": "I can help you schedule an appointment."}
                ))
```

Register in `bot.py`:

```python
def build_operators():
    return [
        KeywordOperator(),
        MyOperator(),
    ]
```

### Adding Sinks

Sinks receive operator outputs and perform actions:

```python
# sinks/database.py
from sinks.base import Sink

class DatabaseSink(Sink):
    async def handle(self, output, ctx):
        if output.kind == "webhook_event":
            # Save to database
            await self.db.insert({
                "event": output.payload,
                "call_sid": ctx.call_sid,
            })
```

### Customizing the System Prompt

Edit `config.py` to change `SYSTEM_PROMPT`.

## Event Types

### CallLifecycleEvent
- `connected` - WebSocket connection established
- `started` - Call audio streaming began
- `ended` - Call ended
- `user_speaking_start/stop` - User speech detection
- `bot_speaking_start/stop` - Bot speech detection

### UserTranscriptEvent
- `text` - Transcribed text
- `is_final` - True if final transcription

### AssistantTextEvent
- `text` - Text the assistant is speaking

## Operator Output Kinds

- `log` - Sent to StdoutSink (console logging)
- `webhook_event` - Sent to WebhookSink (POST to WEBHOOK_URL)
- `speak` - Injects TTSSpeakFrame (makes the bot speak)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ASSEMBLYAI_API_KEY` | Yes | AssemblyAI API key for STT |
| `SAMBANOVA_API_KEY` | Yes | SambaNova API key for LLM |
| `INWORLD_API_KEY` | Yes | Inworld API key for TTS |
| `WEBHOOK_URL` | No | URL to POST call events to |
| `INWORLD_VOICE_ID` | No | Voice ID (default: Ashley) |
| `INWORLD_MODEL` | No | TTS model (default: inworld-tts-1) |
| `SAMBANOVA_MODEL` | No | LLM model (default: Llama-4-Maverick-17B-128E-Instruct) |
