"""AI Receptionist Bot using Pipecat.

This bot uses configurable providers:
- STT: AssemblyAI (default), Deepgram, or Cartesia (ink-whisper)
- LLM: OpenAI GPT-4o (default) or SambaNova Llama-4
- TTS: Cartesia (default) or Inworld

It includes an operator/sink system for reacting to call events
and triggering actions like webhooks or injected speech.

Transport: Twilio WebSocket Media Streams
"""

import asyncio
import os
import sys

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments, WebSocketRunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

from config import (
    ASSEMBLYAI_API_KEY,
    CARTESIA_API_KEY,
    CARTESIA_VOICE_ID,
    DEEPGRAM_API_KEY,
    DJANGO_INTERNAL_URL,
    FALLBACK_LLM,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    INWORLD_API_KEY,
    INWORLD_MODEL,
    INWORLD_VOICE_ID,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PRIMARY_LLM,
    SAMBANOVA_API_KEY,
    SAMBANOVA_MODEL,
    SECONDARY_LLM,
    STT_PROVIDER,
    SYSTEM_PROMPT,
    TTS_PROVIDER,
    TURN_DETECTION_MODEL,
    VAD_STOP_SECS,
    WEBHOOK_URL,
)
from operators.business_context import BusinessContextOperator
from operators.call_lifecycle import CallLifecycleOperator
from operators.end_call import EndCallOperator
from pipeline.observer import ReceptionistObserver
from sinks.stdout import StdoutSink
from sinks.webhook import WebhookSink, upload_recording_to_django

load_dotenv(override=True)

# Global to store call data extracted from Twilio WebSocket
# This is set before transport creation and used in on_client_connected
_current_call_data: Optional[dict] = None


def get_vad_params() -> VADParams:
    """Get VAD parameters tuned for low-latency phone calls.

    These parameters balance responsiveness with avoiding false triggers
    that cut off AI responses mid-sentence:

    - stop_secs: 0.3s - respond quickly after user stops (configurable)
    - start_secs: 0.2s - slightly slower to let AI finish speaking
    - min_volume: 0.5 - filter background noise on phone lines
    - confidence: 0.75 - higher confidence to reduce false positives
    """
    return VADParams(
        stop_secs=VAD_STOP_SECS,  # Configurable silence threshold
        start_secs=0.2,  # Slower trigger to avoid cutting off AI
        min_volume=0.5,  # Higher threshold to filter phone noise
        confidence=0.75,  # Higher confidence to reduce false triggers
    )


def get_turn_analyzer():
    """Get turn analyzer based on TURN_DETECTION_MODEL config.

    Returns:
        LocalSmartTurnAnalyzerV3 if TURN_DETECTION_MODEL="pipecat", None otherwise.
        - "pipecat": ML-based smart turn detection (requires pipecat-ai[local-smart-turn-v3])
        - "assemblyai": None (AssemblyAI handles turn detection internally)
        - "none": None (VAD-only, silence-based turn detection)
    """
    if TURN_DETECTION_MODEL == "pipecat":
        # Import only when needed - requires pipecat-ai[local-smart-turn-v3]
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        logger.debug("Creating LocalSmartTurnAnalyzerV3 for turn detection")
        return LocalSmartTurnAnalyzerV3()
    # For "assemblyai" or "none", return None (VAD-only or AssemblyAI handles it)
    return None


# Transport parameters for Twilio
# turn_analyzer is set based on TURN_DETECTION_MODEL config
transport_params = {
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=get_vad_params()),
        turn_analyzer=get_turn_analyzer(),
    ),
}


def build_operators():
    """Build the list of operators for event handling.

    Operators handle call events and emit outputs to sinks.
    Transcript is now captured from LLMContext at call end (not real-time).
    """
    operators = [
        # Fetch business config from Django API on call start
        BusinessContextOperator(),
        # Track call lifecycle and emit call_ended events
        CallLifecycleOperator(),
        # Auto-hangup after closing message
        EndCallOperator(),
    ]

    logger.info(f"Loaded {len(operators)} operators")
    return operators


def build_sinks():
    """Build the list of sinks for output handling.

    Sinks receive operator outputs and perform actions.
    All call data is sent to Django via webhook (which handles
    transcript storage and GPT-4o analysis).
    """
    sinks = [StdoutSink()]

    if WEBHOOK_URL:
        sinks.append(WebhookSink(WEBHOOK_URL))
        logger.info(f"Webhook sink enabled: {WEBHOOK_URL}")
    else:
        logger.warning("WEBHOOK_URL not set - call data will not be saved!")

    return sinks


def create_stt_service():
    """Create STT service based on configured provider."""
    if STT_PROVIDER == "deepgram":
        if not DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY is required when STT_PROVIDER=deepgram")
        from pipecat.services.deepgram.stt import DeepgramSTTService

        logger.info("Using Deepgram STT")
        return DeepgramSTTService(api_key=DEEPGRAM_API_KEY)
    elif STT_PROVIDER == "assemblyai":
        if not ASSEMBLYAI_API_KEY:
            raise ValueError("ASSEMBLYAI_API_KEY is required when STT_PROVIDER=assemblyai")
        from pipecat.services.assemblyai.stt import AssemblyAISTTService
        from pipecat.services.assemblyai.models import AssemblyAIConnectionParams

        # vad_force_turn_endpoint=True uses pipecat's Silero VAD for turn detection
        # vad_force_turn_endpoint=False uses AssemblyAI's built-in turn detection
        use_assemblyai_turn = TURN_DETECTION_MODEL == "assemblyai"

        if use_assemblyai_turn:
            logger.info("Using AssemblyAI STT with AssemblyAI turn detection")
            # Configure AssemblyAI's built-in turn detection
            connection_params = AssemblyAIConnectionParams(
                end_of_turn_confidence_threshold=0.5,
                min_end_of_turn_silence_when_confident=300,  # ms
                max_turn_silence=1200,  # ms
            )
            return AssemblyAISTTService(
                api_key=ASSEMBLYAI_API_KEY,
                connection_params=connection_params,
                vad_force_turn_endpoint=False,
            )
        else:
            logger.info("Using AssemblyAI STT with pipecat turn detection")
            return AssemblyAISTTService(
                api_key=ASSEMBLYAI_API_KEY,
                vad_force_turn_endpoint=True,
            )
    elif STT_PROVIDER == "cartesia":
        if not CARTESIA_API_KEY:
            raise ValueError("CARTESIA_API_KEY is required when STT_PROVIDER=cartesia")
        from pipecat.services.cartesia.stt import CartesiaSTTService

        logger.info("Using Cartesia STT (ink-whisper)")
        return CartesiaSTTService(api_key=CARTESIA_API_KEY)
    else:
        raise ValueError(f"Unknown STT_PROVIDER: {STT_PROVIDER}")


def _create_single_llm(provider: str):
    """Create a single LLM service by provider name.

    Args:
        provider: One of "sambanova", "google", "openai"

    Returns:
        LLM service instance or None if API key is missing or provider unavailable
    """
    if provider == "sambanova":
        if not SAMBANOVA_API_KEY:
            logger.warning("SAMBANOVA_API_KEY not set, skipping SambaNova")
            return None
        from pipecat.services.sambanova.llm import SambaNovaLLMService

        return SambaNovaLLMService(api_key=SAMBANOVA_API_KEY, model=SAMBANOVA_MODEL)

    elif provider == "google":
        if not GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not set, skipping Google")
            return None
        try:
            from pipecat.services.google.llm import GoogleLLMService

            return GoogleLLMService(api_key=GOOGLE_API_KEY, model=GOOGLE_MODEL)
        except Exception as e:
            logger.warning(f"Google LLM not available: {e}")
            return None

    elif provider == "openai":
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set, skipping OpenAI")
            return None
        from pipecat.services.openai.llm import OpenAILLMService

        return OpenAILLMService(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)

    else:
        logger.warning(f"Unknown LLM provider: {provider}")
        return None


def create_llm_service():
    """Create LLM service with fallback chain.

    Uses PRIMARY_LLM as the main service. If SECONDARY_LLM or FALLBACK_LLM
    are configured, wraps in a FallbackLLMService for automatic failover
    on rate limit errors.
    """
    from services.fallback_llm import FallbackLLMService

    if not PRIMARY_LLM:
        raise ValueError("PRIMARY_LLM environment variable is required")

    # Build list of configured LLM services
    services = []
    provider_names = []

    for provider in [PRIMARY_LLM, SECONDARY_LLM, FALLBACK_LLM]:
        if provider:
            service = _create_single_llm(provider)
            if service:
                services.append(service)
                provider_names.append(provider)

    if not services:
        raise ValueError(
            f"No LLM services could be created. PRIMARY_LLM={PRIMARY_LLM}, check API keys are set."
        )

    # Log the LLM chain
    logger.info(f"LLM chain configured: {' -> '.join(provider_names)}")

    # If only one service, return it directly (no fallback needed)
    if len(services) == 1:
        logger.info(f"Using single LLM: {provider_names[0]}")
        return services[0]

    # Wrap multiple services in fallback handler
    logger.info(f"Using FallbackLLM with {len(services)} providers")
    return FallbackLLMService(services, provider_names)


async def create_tts_service(session: aiohttp.ClientSession):
    """Create TTS service based on configured provider."""
    if TTS_PROVIDER == "cartesia":
        if not CARTESIA_API_KEY:
            raise ValueError("CARTESIA_API_KEY is required when TTS_PROVIDER=cartesia")
        if not CARTESIA_VOICE_ID:
            raise ValueError("CARTESIA_VOICE_ID is required when TTS_PROVIDER=cartesia")
        from pipecat.services.cartesia.tts import CartesiaTTSService

        logger.info(f"Using Cartesia TTS: voice={CARTESIA_VOICE_ID}")
        return CartesiaTTSService(api_key=CARTESIA_API_KEY, voice_id=CARTESIA_VOICE_ID)
    elif TTS_PROVIDER == "inworld":
        if not INWORLD_API_KEY:
            raise ValueError("INWORLD_API_KEY is required when TTS_PROVIDER=inworld")
        from pipecat.services.inworld.tts import InworldTTSService

        logger.info(f"Using Inworld TTS: voice={INWORLD_VOICE_ID}, model={INWORLD_MODEL}")
        return InworldTTSService(
            api_key=INWORLD_API_KEY,
            aiohttp_session=session,
            voice_id=INWORLD_VOICE_ID,
            model=INWORLD_MODEL,
            streaming=True,
        )
    else:
        raise ValueError(f"Unknown TTS_PROVIDER: {TTS_PROVIDER}")


async def run_bot(
    transport: BaseTransport, runner_args: RunnerArguments, call_data: Optional[dict] = None
):
    """Run the receptionist bot.

    Args:
        transport: The Pipecat transport (Daily, Twilio WebSocket, etc.)
        runner_args: Arguments from the Pipecat runner
        call_data: Optional call metadata extracted from Twilio WebSocket.
                   For Twilio, contains: stream_id, call_id, body (customParameters)
    """
    logger.info("Starting AI Receptionist")
    llm_config = f"{PRIMARY_LLM}"
    if SECONDARY_LLM:
        llm_config += f"->{SECONDARY_LLM}"
    if FALLBACK_LLM:
        llm_config += f"->{FALLBACK_LLM}"
    turn_info = TURN_DETECTION_MODEL if STT_PROVIDER == "assemblyai" else "pipecat"
    logger.info(
        f"Configuration: STT={STT_PROVIDER}, Turn={turn_info}, LLM={llm_config}, TTS={TTS_PROVIDER}, VAD_STOP={VAD_STOP_SECS}s"
    )

    async with aiohttp.ClientSession() as session:
        # Initialize services based on configuration
        stt = create_stt_service()
        llm = create_llm_service()
        tts = await create_tts_service(session)

        # Build context with system prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        context = LLMContext(messages)

        context_aggregator = LLMContextAggregatorPair(context)

        # Build observer with operators and sinks
        observer = ReceptionistObserver(
            operators=build_operators(),
            sinks=build_sinks(),
        )
        observer.set_tts(tts)
        observer.set_llm_context(messages)  # Enable dynamic system prompts

        # Create audio recorder for call recording
        # Stereo: user audio on left channel, bot audio on right channel
        # buffer_size=0 means collect entire call and trigger on EndFrame
        # Note: Don't set sample_rate - let it match the pipeline's internal rate
        audio_recorder = AudioBufferProcessor(
            num_channels=2,  # Stereo (user + bot)
            buffer_size=0,  # Collect entire call
        )

        # Variable to capture call_sid for the recording handler
        recording_call_sid = None

        @audio_recorder.event_handler("on_audio_data")
        async def on_recording_complete(
            buffer, audio_data: bytes, sample_rate: int, num_channels: int
        ):
            """Upload recording to Django when call ends.

            Args:
                buffer: The AudioBufferProcessor instance
                audio_data: Raw PCM audio bytes
                sample_rate: Sample rate in Hz
                num_channels: Number of audio channels
            """
            if recording_call_sid and audio_data:
                # Fire and forget - don't block pipeline shutdown
                asyncio.create_task(
                    upload_recording_to_django(
                        recording_call_sid, audio_data, sample_rate, num_channels
                    )
                )

        # Build pipeline with audio recorder
        # IMPORTANT: AudioBufferProcessor must be AFTER transport.output() to capture
        # both user input and bot output audio. Placing it earlier only captures input.
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                audio_recorder,  # After output to capture both sides of conversation
                context_aggregator.assistant(),
            ]
        )

        # Create task with observer
        # Note: Don't set audio sample rates - let pipecat handle resampling
        # (Twilio uses 8kHz, but AssemblyAI needs 16kHz internally)
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[observer],
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )

        # Set task reference for observer (needed for frame injection)
        observer.set_task(task)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")

            # Extract call metadata from call_data (for Twilio) or runner_args
            call_metadata = {}

            if call_data:
                # Twilio: call_data comes from parse_telephony_websocket()
                # call_data["body"] contains customParameters from TwiML <Parameter> tags
                custom_params = call_data.get("body", {})
                call_metadata = {
                    "call_sid": call_data.get("call_id"),  # Twilio CallSid
                    "stream_sid": call_data.get("stream_id"),  # Twilio StreamSid
                    "from_number": custom_params.get("caller_phone"),
                    "to_number": custom_params.get("called_phone"),
                    "business_id": custom_params.get("business_id"),
                    # Demo call parameters (from landing page)
                    "is_demo": custom_params.get("is_demo"),
                    "industry": custom_params.get("industry"),
                    # Recording control (for test calls)
                    "save_recording": custom_params.get("save_recording", "true").lower() == "true",
                }
                if call_metadata.get("is_demo"):
                    logger.info(
                        f"Demo call: {call_metadata.get('call_sid')} "
                        f"industry={call_metadata.get('industry')}"
                    )
                else:
                    logger.info(
                        f"Twilio call: {call_metadata.get('call_sid')} "
                        f"from {call_metadata.get('from_number')} "
                        f"to {call_metadata.get('to_number')} "
                        f"(business: {call_metadata.get('business_id')})"
                    )
            elif hasattr(runner_args, "body") and runner_args.body:
                # Other transports may pass body directly
                call_metadata = {
                    "call_sid": runner_args.body.get("call_sid"),
                    "from_number": runner_args.body.get("from_number"),
                    "to_number": runner_args.body.get("to_number"),
                }

            # Start the observer with call metadata
            await observer.start(call_metadata=call_metadata)

            # Start recording (only for real calls, not demo calls or unsaved test calls)
            nonlocal recording_call_sid
            is_demo = call_metadata.get("is_demo")
            save_recording = call_metadata.get("save_recording", True)
            call_sid = call_metadata.get("call_sid")

            if call_sid and not is_demo and save_recording:
                recording_call_sid = call_sid
                await audio_recorder.start_recording()
                logger.info(f"Started recording for call {recording_call_sid}")
            else:
                recording_call_sid = None
                reason = (
                    "demo call"
                    if is_demo
                    else "save_recording=false"
                    if not save_recording
                    else "no call_sid"
                )
                logger.info(f"Skipping recording: {reason}")

            # Wait for business config to load (voice + system prompt)
            # This ensures the correct voice is used for the greeting
            config_loaded = await observer.wait_for_config(timeout=3.0)
            if config_loaded:
                logger.info("Business config loaded, starting greeting with correct voice")
            else:
                logger.warning("Config timeout, starting greeting with default voice")

            # Kick off the conversation with a greeting
            messages.append(
                {
                    "role": "system",
                    "content": "A caller has just connected. Use the EXACT greeting from the <greeting> tag in your system prompt - say it word for word, do not paraphrase or shorten it.",
                }
            )
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")

            # Stop the observer
            await observer.stop()

            # Cancel the task
            await task.cancel()

        # Run the pipeline
        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
        await runner.run(task)

        # Ensure observer is stopped after pipeline completes
        # This is a fallback in case on_client_disconnected didn't fire
        # (e.g., when EndFrame triggers hangup before disconnect event)
        await observer.stop()


async def create_transport_with_call_data(runner_args: RunnerArguments):
    """Create Twilio transport and extract call_data.

    Parses the Twilio WebSocket to extract customParameters from TwiML <Parameter> tags.

    Returns:
        tuple: (transport, call_data)
    """
    if not isinstance(runner_args, WebSocketRunnerArguments):
        raise ValueError("Only WebSocket (Twilio) transport is supported")

    # Parse the WebSocket to capture call_data before creating the transport
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)

    logger.info(
        f"Parsed telephony WebSocket: type={transport_type}, call_id={call_data.get('call_id')}"
    )
    logger.debug(f"Call data body (customParameters): {call_data.get('body', {})}")

    if transport_type != "twilio":
        raise ValueError(f"Unsupported telephony provider: {transport_type}")

    # Create transport params
    params = transport_params["twilio"]()
    params.add_wav_header = False

    from pipecat.serializers.twilio import TwilioFrameSerializer

    params.serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport

    transport = FastAPIWebsocketTransport(websocket=runner_args.websocket, params=params)

    return transport, call_data


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud and local runner.

    For telephony transports (Twilio), this extracts call metadata from the
    WebSocket connection before creating the transport. The metadata includes
    customParameters from the TwiML <Parameter> tags (call_sid, business_id, etc.)
    """
    transport, call_data = await create_transport_with_call_data(runner_args)
    await run_bot(transport, runner_args, call_data)


if __name__ == "__main__":
    # Use standard pipecat runner for Twilio WebSocket
    from pipecat.runner.run import main

    main()
