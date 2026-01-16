"""Benchmark configuration."""

import os
from dataclasses import dataclass, field
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv(override=True)

# Test parameters
NUM_RUNS = 5
PAUSE_BETWEEN_RUNS = 3.0  # seconds

# Test phrase for TTS
TTS_TEST_PHRASE = "Hello, thank you for calling. How may I assist you today?"

# Test prompt for LLM
LLM_TEST_PROMPT = "You are a receptionist. A caller just said 'Hi, I'd like to speak with a manager.' Respond briefly."

# API Keys
API_KEYS = {
    "assemblyai": os.getenv("ASSEMBLYAI_API_KEY", ""),
    "deepgram": os.getenv("DEEPGRAM_API_KEY", ""),
    "sambanova": os.getenv("SAMBANOVA_API_KEY", ""),
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "cartesia": os.getenv("CARTESIA_API_KEY", ""),
    "rime": os.getenv("RIME_API_KEY", ""),
    "hume": os.getenv("HUME_API_KEY", ""),
    "inworld": os.getenv("INWORLD_API_KEY", ""),
}

# Voice IDs
VOICE_IDS = {
    "cartesia": os.getenv("CARTESIA_DEFAULT_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091"),
    "rime": "marsh",
    "hume": os.getenv("HUME_DEFAULT_VOICE_ID", ""),
    "inworld": "Ashley",
    "deepgram": "aura-2-helena-en",
}

# LLM Models to test
LLM_MODELS = {
    "sambanova": {
        "provider": "sambanova",
        "model": "Llama-4-Maverick-17B-128E-Instruct",
    },
    "gpt-5-nano": {
        "provider": "openai",
        "model": "gpt-5-nano",
    },
    "gpt-5-mini": {
        "provider": "openai",
        "model": "gpt-5-mini",
    },
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
    },
}

# TTS Providers to test
TTS_PROVIDERS = ["cartesia", "rime", "deepgram", "hume", "inworld"]

# STT Providers to test
STT_PROVIDERS = ["assemblyai", "deepgram"]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    provider: str
    model: str = ""
    ttfb: float = 0.0
    total_time: float = 0.0
    error: str = ""
    success: bool = True


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results for a provider."""
    provider: str
    model: str = ""
    ttfb_avg: float = 0.0
    ttfb_min: float = 0.0
    ttfb_max: float = 0.0
    ttfb_std: float = 0.0
    total_time_avg: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    runs: List[BenchmarkResult] = field(default_factory=list)


def check_api_keys() -> Dict[str, bool]:
    """Check which API keys are available."""
    return {k: bool(v) for k, v in API_KEYS.items()}
