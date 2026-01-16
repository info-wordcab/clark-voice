"""End-to-End Pipeline Latency Benchmark.

Measures the time from user turn completion (VAD end-of-speech) to first TTS audio byte.
Uses a simulated audio transport to inject pre-recorded audio and capture output timing.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from benchmark.config import API_KEYS, VOICE_IDS, BenchmarkResult


# Default VAD stop seconds (can be overridden)
DEFAULT_VAD_STOP_SECS = 0.2

# Provider configurations for testing
PROVIDER_CONFIGS = {
    "fastest": {
        "name": "Deepgram + GPT-4o + Cartesia",
        "stt": "deepgram",
        "llm": {"provider": "openai", "model": "gpt-4o"},
        "tts": "cartesia",
    },
    "dg-gpt4o-inworld": {
        "name": "Deepgram + GPT-4o + Inworld",
        "stt": "deepgram",
        "llm": {"provider": "openai", "model": "gpt-4o"},
        "tts": "inworld",
    },
    "dg-sambanova-cartesia": {
        "name": "Deepgram + SambaNova + Cartesia",
        "stt": "deepgram",
        "llm": {"provider": "sambanova", "model": "Llama-4-Maverick-17B-128E-Instruct"},
        "tts": "cartesia",
    },
    "aai-gpt4o-cartesia": {
        "name": "AssemblyAI + GPT-4o + Cartesia",
        "stt": "assemblyai",
        "llm": {"provider": "openai", "model": "gpt-4o"},
        "tts": "cartesia",
    },
}


@dataclass
class E2EBenchmarkResult:
    """Result of an end-to-end benchmark run."""
    config_name: str
    turn_to_first_audio_ms: float
    llm_ttfb_ms: float = 0.0
    tts_ttfb_ms: float = 0.0
    total_response_ms: float = 0.0
    error: str = ""
    success: bool = True


class LatencyMeasurementProcessor(FrameProcessor):
    """Processor that measures latency between key pipeline events."""

    def __init__(self):
        super().__init__()
        self.user_stopped_time: Optional[float] = None
        self.first_tts_audio_time: Optional[float] = None
        self.tts_started_time: Optional[float] = None
        self.response_complete_time: Optional[float] = None
        self._got_first_audio = False
        self._measurement_complete = asyncio.Event()

    def reset(self):
        """Reset measurement state for a new run."""
        self.user_stopped_time = None
        self.first_tts_audio_time = None
        self.tts_started_time = None
        self.response_complete_time = None
        self._got_first_audio = False
        self._measurement_complete.clear()

    async def wait_for_measurement(self, timeout: float = 30.0) -> bool:
        """Wait for measurement to complete."""
        try:
            await asyncio.wait_for(self._measurement_complete.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def get_latency_ms(self) -> Optional[float]:
        """Get the turn-to-first-audio latency in milliseconds."""
        if self.user_stopped_time and self.first_tts_audio_time:
            return (self.first_tts_audio_time - self.user_stopped_time) * 1000
        return None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and record timing."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            self.user_stopped_time = time.perf_counter()
            logger.debug(f"User stopped speaking at {self.user_stopped_time}")

        elif isinstance(frame, TTSStartedFrame):
            self.tts_started_time = time.perf_counter()
            logger.debug(f"TTS started at {self.tts_started_time}")

        elif isinstance(frame, TTSAudioRawFrame) and not self._got_first_audio:
            self.first_tts_audio_time = time.perf_counter()
            self._got_first_audio = True
            latency = self.get_latency_ms()
            logger.info(f"First TTS audio received - Turn-to-Audio latency: {latency:.1f}ms")
            self._measurement_complete.set()

        await self.push_frame(frame, direction)


class SimulatedInputTransport(BaseInputTransport):
    """Simulated input transport that injects pre-recorded audio."""

    def __init__(self, params: TransportParams):
        super().__init__(params)
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self, frame: StartFrame):
        """Start the simulated input transport."""
        await super().start(frame)
        self._running = True
        await self.set_transport_ready(frame)

    async def cleanup(self):
        """Clean up the transport."""
        self._running = False
        await super().cleanup()

    async def inject_audio(self, audio_data: bytes, sample_rate: int = 16000):
        """Inject audio data into the pipeline."""
        # Break into chunks (~20ms each)
        chunk_size = sample_rate * 2 * 20 // 1000  # 20ms of 16-bit mono audio
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if chunk:
                frame = InputAudioRawFrame(
                    audio=chunk,
                    sample_rate=sample_rate,
                    num_channels=1,
                )
                await self.push_audio_frame(frame)
                await asyncio.sleep(0.02)  # 20ms delay between chunks


class SimulatedOutputTransport(BaseOutputTransport):
    """Simulated output transport that captures output audio."""

    def __init__(self, params: TransportParams):
        super().__init__(params)
        self.audio_chunks: List[bytes] = []

    async def start(self, frame: StartFrame):
        """Start the simulated output transport."""
        await super().start(frame)

    async def write_raw_audio_frames(self, frames: bytes):
        """Capture output audio frames."""
        self.audio_chunks.append(frames)


class SimulatedTransport(BaseTransport):
    """Combined simulated transport for benchmarking."""

    def __init__(self, params: TransportParams):
        super().__init__()
        self._params = params
        self._input = SimulatedInputTransport(params)
        self._output = SimulatedOutputTransport(params)

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output

    async def inject_audio(self, audio_data: bytes, sample_rate: int = 16000):
        """Inject audio into the input transport."""
        await self._input.inject_audio(audio_data, sample_rate)


def create_stt_service(provider: str):
    """Create STT service based on provider name."""
    if provider == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService
        return DeepgramSTTService(api_key=API_KEYS["deepgram"])
    elif provider == "assemblyai":
        from pipecat.services.assemblyai.stt import AssemblyAISTTService
        return AssemblyAISTTService(api_key=API_KEYS["assemblyai"])
    else:
        raise ValueError(f"Unknown STT provider: {provider}")


def create_llm_service(config: dict):
    """Create LLM service based on config."""
    provider = config["provider"]
    model = config["model"]

    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService
        return OpenAILLMService(api_key=API_KEYS["openai"], model=model)
    elif provider == "sambanova":
        from pipecat.services.sambanova.llm import SambaNovaLLMService
        return SambaNovaLLMService(api_key=API_KEYS["sambanova"], model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def create_tts_service(provider: str, session: aiohttp.ClientSession):
    """Create TTS service based on provider name."""
    if provider == "inworld":
        from pipecat.services.inworld.tts import InworldTTSService
        return InworldTTSService(
            api_key=API_KEYS["inworld"],
            aiohttp_session=session,
            voice_id=VOICE_IDS["inworld"],
            model="inworld-tts-1",
            streaming=True,
        )
    elif provider == "cartesia":
        from pipecat.services.cartesia.tts import CartesiaTTSService
        return CartesiaTTSService(
            api_key=API_KEYS["cartesia"],
            voice_id=VOICE_IDS["cartesia"],
        )
    elif provider == "deepgram":
        from pipecat.services.deepgram.tts import DeepgramTTSService
        return DeepgramTTSService(
            api_key=API_KEYS["deepgram"],
            voice=VOICE_IDS["deepgram"],
        )
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")


async def generate_test_audio() -> bytes:
    """Generate test audio using Cartesia TTS.
    
    Returns PCM audio data of a test phrase.
    """
    import websockets
    import json
    import base64

    api_key = API_KEYS["cartesia"]
    voice_id = VOICE_IDS["cartesia"]
    test_phrase = "Hello, I would like to speak with the manager please."

    url = f"wss://api.cartesia.ai/tts/websocket?api_key={api_key}&cartesia_version=2025-04-16"

    audio_chunks = []

    async with websockets.connect(url) as ws:
        msg = {
            "context_id": "test",
            "model_id": "sonic-3",
            "voice": {"mode": "id", "id": voice_id},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
            "transcript": test_phrase,
            "continue": False,
        }
        await ws.send(json.dumps(msg))

        async for response in ws:
            data = json.loads(response)
            if data.get("type") == "chunk" and data.get("data"):
                audio_chunks.append(base64.b64decode(data["data"]))
            elif data.get("type") == "done":
                break

    return b"".join(audio_chunks)


async def run_e2e_benchmark(
    config_name: str,
    test_audio: bytes,
    num_runs: int = 3,
) -> List[E2EBenchmarkResult]:
    """Run end-to-end benchmark for a specific configuration."""
    config = PROVIDER_CONFIGS[config_name]
    results = []

    logger.info(f"Benchmarking: {config['name']}")

    for run_idx in range(num_runs):
        logger.info(f"  Run {run_idx + 1}/{num_runs}")

        try:
            async with aiohttp.ClientSession() as session:
                # Create services
                stt = create_stt_service(config["stt"])
                llm = create_llm_service(config["llm"])
                tts = await create_tts_service(config["tts"], session)

                # Create transport with VAD
                transport_params = TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=DEFAULT_VAD_STOP_SECS)),
                )
                transport = SimulatedTransport(transport_params)

                # Create latency measurement processor
                latency_processor = LatencyMeasurementProcessor()

                # Build context
                messages = [
                    {"role": "system", "content": "You are a helpful receptionist. Respond briefly and professionally."},
                ]
                context = LLMContext(messages)
                context_aggregator = LLMContextAggregatorPair(context)

                # Build pipeline with latency processor after TTS
                pipeline = Pipeline(
                    [
                        transport.input(),
                        stt,
                        context_aggregator.user(),
                        llm,
                        tts,
                        latency_processor,  # Measures timing here
                        transport.output(),
                        context_aggregator.assistant(),
                    ]
                )

                # Create task
                task = PipelineTask(
                    pipeline,
                    params=PipelineParams(
                        enable_metrics=True,
                        enable_usage_metrics=True,
                    ),
                )

                # Start the pipeline
                runner = PipelineRunner(handle_sigint=False)

                # Run pipeline in background
                pipeline_task = asyncio.create_task(runner.run(task))

                # Give pipeline time to start
                await asyncio.sleep(1.0)

                # Inject test audio
                logger.debug("Injecting test audio...")
                await transport.inject_audio(test_audio, sample_rate=16000)

                # Add silence after speech to trigger VAD end-of-speech
                silence = bytes(16000 * 2)  # 1 second of silence
                await transport.inject_audio(silence, sample_rate=16000)

                # Wait for measurement
                success = await latency_processor.wait_for_measurement(timeout=30.0)

                if success:
                    latency_ms = latency_processor.get_latency_ms()
                    results.append(E2EBenchmarkResult(
                        config_name=config_name,
                        turn_to_first_audio_ms=latency_ms,
                        success=True,
                    ))
                    logger.info(f"    Latency: {latency_ms:.1f}ms")
                else:
                    results.append(E2EBenchmarkResult(
                        config_name=config_name,
                        turn_to_first_audio_ms=0,
                        error="Timeout waiting for response",
                        success=False,
                    ))
                    logger.warning("    Timeout waiting for response")

                # Cleanup
                await task.cancel()
                pipeline_task.cancel()
                try:
                    await pipeline_task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"    Error: {e}")
            results.append(E2EBenchmarkResult(
                config_name=config_name,
                turn_to_first_audio_ms=0,
                error=str(e),
                success=False,
            ))

        # Pause between runs
        if run_idx < num_runs - 1:
            await asyncio.sleep(2.0)

    return results


async def run_all_e2e_benchmarks(
    configs: Optional[List[str]] = None,
    num_runs: int = 3,
) -> dict:
    """Run end-to-end benchmarks for all configurations."""
    configs = configs or list(PROVIDER_CONFIGS.keys())

    logger.info("Generating test audio...")
    test_audio = await generate_test_audio()
    logger.info(f"Generated {len(test_audio)} bytes of test audio")

    all_results = {}

    for config_name in configs:
        results = await run_e2e_benchmark(config_name, test_audio, num_runs)
        all_results[config_name] = results

        # Calculate summary
        successful = [r for r in results if r.success]
        if successful:
            avg_latency = sum(r.turn_to_first_audio_ms for r in successful) / len(successful)
            min_latency = min(r.turn_to_first_audio_ms for r in successful)
            max_latency = max(r.turn_to_first_audio_ms for r in successful)
            logger.info(
                f"  Summary: avg={avg_latency:.1f}ms, min={min_latency:.1f}ms, "
                f"max={max_latency:.1f}ms, success={len(successful)}/{len(results)}"
            )

    return all_results


def print_e2e_summary(results: dict):
    """Print summary of end-to-end benchmark results."""
    print("\n" + "=" * 80)
    print("END-TO-END LATENCY BENCHMARK RESULTS")
    print("(Time from user turn end to first TTS audio byte)")
    print("=" * 80)

    # Sort by average latency
    summaries = []
    for config_name, runs in results.items():
        successful = [r for r in runs if r.success]
        if successful:
            avg = sum(r.turn_to_first_audio_ms for r in successful) / len(successful)
            min_val = min(r.turn_to_first_audio_ms for r in successful)
            max_val = max(r.turn_to_first_audio_ms for r in successful)
            summaries.append((config_name, avg, min_val, max_val, len(successful), len(runs)))

    summaries.sort(key=lambda x: x[1])

    print(f"\n{'Configuration':<50} | {'Avg':>8} | {'Min':>8} | {'Max':>8} | {'Success':>8}")
    print("-" * 90)

    for config_name, avg, min_val, max_val, success_count, total in summaries:
        config = PROVIDER_CONFIGS[config_name]
        print(
            f"{config['name'][:50]:<50} | {avg:>7.0f}ms | {min_val:>7.0f}ms | "
            f"{max_val:>7.0f}ms | {success_count}/{total}"
        )

    print("=" * 80)


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    configs = None
    num_runs = 3

    args = sys.argv[1:]
    
    if "--help" in args or "-h" in args:
        print("Usage: python -m benchmark.e2e_benchmark [config1,config2,...] [--runs N]")
        print(f"\nAvailable configs: {', '.join(PROVIDER_CONFIGS.keys())}")
        sys.exit(0)

    if "--runs" in args:
        idx = args.index("--runs")
        num_runs = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]  # Remove --runs and its value

    # Remaining args are config names
    if args:
        configs = args[0].split(",")

    async def main():
        results = await run_all_e2e_benchmarks(configs, num_runs)
        print_e2e_summary(results)

    asyncio.run(main())
