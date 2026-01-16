"""Configuration and constants for the receptionist bot."""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

# API Keys
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
INWORLD_API_KEY = os.getenv("INWORLD_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")

# LLM API Keys - per provider
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# STT Provider: "deepgram", "assemblyai", or "cartesia"
STT_PROVIDER = os.getenv("STT_PROVIDER", "assemblyai")

# Turn Detection Model:
# - "pipecat" (default): LocalSmartTurnAnalyzerV3 (ML-based smart turn detection)
# - "assemblyai": AssemblyAI's built-in turn detection (only when STT_PROVIDER=assemblyai)
# - "none": VAD only (no smart turn detection, just silence-based)
TURN_DETECTION_MODEL = os.getenv("TURN_DETECTION_MODEL", "pipecat")

# LLM Configuration - hierarchical fallback
# Values can be: "sambanova", "google", "openai"
# PRIMARY_LLM is required, SECONDARY_LLM and FALLBACK_LLM are optional
PRIMARY_LLM = os.getenv("PRIMARY_LLM")
SECONDARY_LLM = os.getenv("SECONDARY_LLM")
FALLBACK_LLM = os.getenv("FALLBACK_LLM")

# TTS Provider: "inworld" or "cartesia"
# Cartesia has lower latency with streaming synthesis
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "cartesia")

# Optional webhook for call events
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# Django API URL (internal via Fly.io 6PN)
DJANGO_INTERNAL_URL = os.getenv("DJANGO_INTERNAL_URL", "http://localhost:8000")

# Database URL for direct Postgres writes (shared with Django)
DATABASE_URL = os.getenv("DATABASE_URL", "")

# TTS Configuration
INWORLD_VOICE_ID = os.getenv("INWORLD_VOICE_ID", "Ashley")
INWORLD_MODEL = os.getenv("INWORLD_MODEL", "inworld-tts-1")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", os.getenv("CARTESIA_DEFAULT_VOICE_ID", ""))

# LLM Configuration
# GPT-4o-mini is faster with good quality for voice; GPT-4o for complex tasks
SAMBANOVA_MODEL = os.getenv("SAMBANOVA_MODEL", "Llama-4-Maverick-17B-128E-Instruct")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview")

# VAD Configuration
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.3"))

# Default system prompt (used when Django API is unavailable)
# In production, the system prompt is fetched from Django API
SYSTEM_PROMPT = """You are a friendly and professional AI receptionist. Your role is to:

- Greet callers warmly and professionally
- Answer questions about the business
- Take messages when the appropriate person is unavailable
- Help schedule appointments or callbacks
- Direct callers to the right department or person
- Handle common inquiries with helpful information

Guidelines for your responses:
- Keep responses brief and conversational - this is a phone call
- Be warm but professional
- Ask clarifying questions when needed
- If you cannot help with something, offer to take a message or transfer the call
- Never use emojis, bullet points, or special characters that cannot be spoken naturally
- Speak in complete, natural sentences

If asked about specific business details you do not know, politely explain that you can take a message or help connect them with someone who can assist."""
