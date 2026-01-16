"""STT Provider Benchmarks.

Measures Time To First Byte (TTFB) for various STT providers.
First generates test audio using a TTS provider, then measures STT latency.
"""

import asyncio
import base64
import io
import struct
import time
from typing import List, Optional

import aiohttp
from loguru import logger

from benchmark.config import (
    API_KEYS,
    VOICE_IDS,
    TTS_TEST_PHRASE,
    NUM_RUNS,
    PAUSE_BETWEEN_RUNS,
    BenchmarkResult,
    BenchmarkSummary,
)


def generate_sine_wave_audio(
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> bytes:
    """Generate a simple sine wave audio for testing.
    
    Returns PCM 16-bit audio data.
    """
    import math
    
    num_samples = int(sample_rate * duration_sec)
    samples = []
    
    for i in range(num_samples):
        t = i / sample_rate
        # Simple sine wave
        value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack('<h', value))
    
    return b''.join(samples)


async def generate_test_audio_with_cartesia() -> Optional[bytes]:
    """Generate test audio using Cartesia TTS."""
    import websockets
    import json
    
    api_key = API_KEYS["cartesia"]
    voice_id = VOICE_IDS["cartesia"]
    
    if not api_key:
        logger.warning("No Cartesia API key, using synthetic audio")
        return None
    
    url = f"wss://api.cartesia.ai/tts/websocket?api_key={api_key}&cartesia_version=2025-04-16"
    
    try:
        audio_chunks = []
        
        async with websockets.connect(url) as ws:
            context_msg = {
                "context_id": "audio_gen",
                "model_id": "sonic-3",
                "voice": {"mode": "id", "id": voice_id},
                "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
                "transcript": TTS_TEST_PHRASE,
                "continue": False,
            }
            await ws.send(json.dumps(context_msg))
            
            async for msg in ws:
                data = json.loads(msg)
                if data.get("type") == "chunk" and data.get("data"):
                    audio_chunks.append(base64.b64decode(data["data"]))
                elif data.get("type") == "done":
                    break
        
        if audio_chunks:
            return b''.join(audio_chunks)
        return None
        
    except Exception as e:
        logger.warning(f"Failed to generate audio with Cartesia: {e}")
        return None


async def get_test_audio() -> bytes:
    """Get test audio for STT benchmarks."""
    # Try to generate with Cartesia
    audio = await generate_test_audio_with_cartesia()
    
    if audio:
        logger.info(f"Generated test audio with Cartesia ({len(audio)} bytes)")
        return audio
    
    # Fallback to synthetic sine wave
    logger.info("Using synthetic sine wave audio for STT benchmark")
    return generate_sine_wave_audio()


async def benchmark_assemblyai(session: aiohttp.ClientSession, audio: bytes) -> BenchmarkResult:
    """Benchmark AssemblyAI STT TTFB using real-time streaming."""
    import websockets
    import json
    
    api_key = API_KEYS["assemblyai"]
    
    if not api_key:
        return BenchmarkResult(provider="assemblyai", error="No API key", success=False)
    
    # AssemblyAI real-time WebSocket URL
    url = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"
    
    try:
        start_time = time.perf_counter()
        first_transcript_time = None
        
        headers = {"Authorization": api_key}
        
        async with websockets.connect(url, additional_headers=headers) as ws:
            # Wait for session start
            msg = await ws.recv()
            data = json.loads(msg)
            if data.get("message_type") != "SessionBegins":
                return BenchmarkResult(
                    provider="assemblyai",
                    error=f"Unexpected message: {data}",
                    success=False
                )
            
            # Send audio in chunks
            chunk_size = 3200  # 100ms at 16kHz, 16-bit
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                await ws.send(json.dumps({
                    "audio_data": base64.b64encode(chunk).decode()
                }))
                await asyncio.sleep(0.05)  # Simulate real-time
                
                # Check for any transcripts
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    data = json.loads(msg)
                    if data.get("message_type") == "PartialTranscript" and data.get("text"):
                        first_transcript_time = time.perf_counter()
                        break
                    elif data.get("message_type") == "FinalTranscript" and data.get("text"):
                        first_transcript_time = time.perf_counter()
                        break
                except asyncio.TimeoutError:
                    continue
            
            # If no transcript yet, wait a bit more
            if not first_transcript_time:
                try:
                    for _ in range(20):  # Wait up to 2 seconds
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                        data = json.loads(msg)
                        if data.get("text"):
                            first_transcript_time = time.perf_counter()
                            break
                except asyncio.TimeoutError:
                    pass
            
            # Close session
            await ws.send(json.dumps({"terminate_session": True}))
        
        if first_transcript_time:
            ttfb = first_transcript_time - start_time
            return BenchmarkResult(provider="assemblyai", ttfb=ttfb, total_time=ttfb)
        else:
            return BenchmarkResult(
                provider="assemblyai",
                error="No transcript received",
                success=False
            )
            
    except Exception as e:
        return BenchmarkResult(provider="assemblyai", error=str(e), success=False)


async def benchmark_deepgram_stt(session: aiohttp.ClientSession, audio: bytes) -> BenchmarkResult:
    """Benchmark Deepgram STT TTFB using real-time streaming."""
    import websockets
    import json
    
    api_key = API_KEYS["deepgram"]
    
    if not api_key:
        return BenchmarkResult(provider="deepgram", error="No API key", success=False)
    
    # Deepgram real-time WebSocket URL
    url = "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1&interim_results=true"
    
    try:
        start_time = time.perf_counter()
        first_transcript_time = None
        
        headers = {"Authorization": f"Token {api_key}"}
        
        async with websockets.connect(url, additional_headers=headers) as ws:
            # Send audio in chunks
            chunk_size = 3200  # 100ms at 16kHz, 16-bit
            
            async def send_audio():
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i+chunk_size]
                    await ws.send(chunk)
                    await asyncio.sleep(0.05)  # Simulate real-time
                # Send close message
                await ws.send(json.dumps({"type": "CloseStream"}))
            
            async def receive_transcripts():
                nonlocal first_transcript_time
                async for msg in ws:
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                        if transcript:
                            first_transcript_time = time.perf_counter()
                            return
                        if data.get("type") == "Results" and data.get("is_final"):
                            return
            
            # Run send and receive concurrently
            await asyncio.gather(
                send_audio(),
                asyncio.wait_for(receive_transcripts(), timeout=5.0),
            )
        
        if first_transcript_time:
            ttfb = first_transcript_time - start_time
            return BenchmarkResult(provider="deepgram", ttfb=ttfb, total_time=ttfb)
        else:
            return BenchmarkResult(
                provider="deepgram",
                error="No transcript received",
                success=False
            )
            
    except asyncio.TimeoutError:
        return BenchmarkResult(provider="deepgram", error="Timeout waiting for transcript", success=False)
    except Exception as e:
        return BenchmarkResult(provider="deepgram", error=str(e), success=False)


# Map provider names to benchmark functions
STT_BENCHMARKS = {
    "assemblyai": benchmark_assemblyai,
    "deepgram": benchmark_deepgram_stt,
}


async def run_stt_benchmark(provider: str, audio: bytes, num_runs: int = NUM_RUNS) -> BenchmarkSummary:
    """Run STT benchmark for a specific provider."""
    if provider not in STT_BENCHMARKS:
        return BenchmarkSummary(
            provider=provider,
            error_count=1,
            runs=[BenchmarkResult(provider=provider, error=f"Unknown provider: {provider}", success=False)]
        )
    
    benchmark_fn = STT_BENCHMARKS[provider]
    results: List[BenchmarkResult] = []
    
    logger.info(f"Benchmarking STT: {provider} ({num_runs} runs)")
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_runs):
            logger.debug(f"  Run {i+1}/{num_runs}")
            result = await benchmark_fn(session, audio)
            results.append(result)
            
            if result.success:
                logger.debug(f"    TTFB: {result.ttfb*1000:.1f}ms")
            else:
                logger.warning(f"    Error: {result.error}")
            
            if i < num_runs - 1:
                await asyncio.sleep(PAUSE_BETWEEN_RUNS)
    
    # Calculate summary statistics
    successful = [r for r in results if r.success]
    
    if successful:
        ttfbs = [r.ttfb for r in successful]
        import statistics
        
        summary = BenchmarkSummary(
            provider=provider,
            ttfb_avg=statistics.mean(ttfbs),
            ttfb_min=min(ttfbs),
            ttfb_max=max(ttfbs),
            ttfb_std=statistics.stdev(ttfbs) if len(ttfbs) > 1 else 0.0,
            total_time_avg=statistics.mean([r.total_time for r in successful]),
            success_rate=len(successful) / len(results),
            error_count=len(results) - len(successful),
            runs=results,
        )
    else:
        summary = BenchmarkSummary(
            provider=provider,
            success_rate=0.0,
            error_count=len(results),
            runs=results,
        )
    
    return summary


async def run_all_stt_benchmarks(providers: Optional[List[str]] = None) -> dict:
    """Run benchmarks for all STT providers."""
    from benchmark.config import STT_PROVIDERS
    
    providers = providers or STT_PROVIDERS
    
    # Generate test audio first
    logger.info("Generating test audio for STT benchmarks...")
    audio = await get_test_audio()
    
    results = {}
    
    for provider in providers:
        summary = await run_stt_benchmark(provider, audio)
        results[provider] = summary
        
        if summary.success_rate > 0:
            logger.info(
                f"STT {provider}: TTFB avg={summary.ttfb_avg*1000:.1f}ms, "
                f"min={summary.ttfb_min*1000:.1f}ms, max={summary.ttfb_max*1000:.1f}ms, "
                f"success={summary.success_rate*100:.0f}%"
            )
        else:
            logger.warning(f"STT {provider}: All runs failed")
    
    return results
