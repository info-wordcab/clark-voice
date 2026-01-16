"""TTS Provider Benchmarks.

Measures Time To First Byte (TTFB) for various TTS providers.
"""

import asyncio
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


async def benchmark_cartesia(session: aiohttp.ClientSession) -> BenchmarkResult:
    """Benchmark Cartesia TTS TTFB."""
    import websockets
    import json
    
    api_key = API_KEYS["cartesia"]
    voice_id = VOICE_IDS["cartesia"]
    
    if not api_key:
        return BenchmarkResult(provider="cartesia", error="No API key", success=False)
    
    url = f"wss://api.cartesia.ai/tts/websocket?api_key={api_key}&cartesia_version=2025-04-16"
    
    try:
        start_time = time.perf_counter()
        first_byte_time = None
        
        async with websockets.connect(url) as ws:
            # Send context message
            context_msg = {
                "context_id": "benchmark",
                "model_id": "sonic-3",
                "voice": {"mode": "id", "id": voice_id},
                "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
                "transcript": TTS_TEST_PHRASE,
                "continue": False,
            }
            await ws.send(json.dumps(context_msg))
            
            # Wait for first audio chunk
            async for msg in ws:
                data = json.loads(msg)
                if data.get("type") == "chunk" and data.get("data"):
                    first_byte_time = time.perf_counter()
                    break
                elif data.get("type") == "done":
                    break
                elif data.get("type") == "error":
                    return BenchmarkResult(
                        provider="cartesia",
                        error=data.get("message", "Unknown error"),
                        success=False
                    )
        
        if first_byte_time:
            ttfb = first_byte_time - start_time
            return BenchmarkResult(provider="cartesia", ttfb=ttfb, total_time=ttfb)
        else:
            return BenchmarkResult(provider="cartesia", error="No audio received", success=False)
            
    except Exception as e:
        return BenchmarkResult(provider="cartesia", error=str(e), success=False)


async def benchmark_rime(session: aiohttp.ClientSession) -> BenchmarkResult:
    """Benchmark Rime TTS TTFB."""
    import websockets
    
    api_key = API_KEYS["rime"]
    voice_id = VOICE_IDS["rime"]
    
    if not api_key:
        return BenchmarkResult(provider="rime", error="No API key", success=False)
    
    # Rime WebSocket URL with query params (per their docs)
    url = f"wss://users.rime.ai/ws?speaker={voice_id}&modelId=mistv2&audioFormat=pcm&samplingRate=16000"
    
    try:
        start_time = time.perf_counter()
        first_byte_time = None
        
        # Auth via headers
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async with websockets.connect(url, additional_headers=headers) as ws:
            # Send text directly (not JSON), then EOS to signal end
            await ws.send(TTS_TEST_PHRASE)
            await ws.send("<EOS>")
            
            # Wait for first audio chunk (raw bytes)
            async for msg in ws:
                if isinstance(msg, bytes) and len(msg) > 0:
                    first_byte_time = time.perf_counter()
                    break
        
        if first_byte_time:
            ttfb = first_byte_time - start_time
            return BenchmarkResult(provider="rime", ttfb=ttfb, total_time=ttfb)
        else:
            return BenchmarkResult(provider="rime", error="No audio received", success=False)
            
    except Exception as e:
        return BenchmarkResult(provider="rime", error=str(e), success=False)


async def benchmark_deepgram_tts(session: aiohttp.ClientSession) -> BenchmarkResult:
    """Benchmark Deepgram TTS TTFB."""
    import websockets
    import json
    
    api_key = API_KEYS["deepgram"]
    voice = VOICE_IDS["deepgram"]
    
    if not api_key:
        return BenchmarkResult(provider="deepgram", error="No API key", success=False)
    
    url = f"wss://api.deepgram.com/v1/speak?model={voice}&encoding=linear16&sample_rate=24000"
    
    try:
        start_time = time.perf_counter()
        first_byte_time = None
        
        headers = {"Authorization": f"Token {api_key}"}
        
        async with websockets.connect(url, additional_headers=headers) as ws:
            # Send text
            msg = {"type": "Speak", "text": TTS_TEST_PHRASE}
            await ws.send(json.dumps(msg))
            
            # Send flush to indicate end of text
            await ws.send(json.dumps({"type": "Flush"}))
            
            # Wait for first audio chunk
            async for msg in ws:
                if isinstance(msg, bytes) and len(msg) > 0:
                    first_byte_time = time.perf_counter()
                    break
                elif isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("type") == "Flushed":
                        break
        
        if first_byte_time:
            ttfb = first_byte_time - start_time
            return BenchmarkResult(provider="deepgram", ttfb=ttfb, total_time=ttfb)
        else:
            return BenchmarkResult(provider="deepgram", error="No audio received", success=False)
            
    except Exception as e:
        return BenchmarkResult(provider="deepgram", error=str(e), success=False)


async def benchmark_hume(session: aiohttp.ClientSession) -> BenchmarkResult:
    """Benchmark Hume TTS TTFB using HTTP streaming endpoint."""
    import json
    
    api_key = API_KEYS["hume"]
    voice_id = VOICE_IDS["hume"]
    
    if not api_key:
        return BenchmarkResult(provider="hume", error="No API key", success=False)
    if not voice_id:
        return BenchmarkResult(provider="hume", error="No voice ID configured", success=False)
    
    # Use HTTP streaming endpoint (per Hume docs: /v0/tts/stream/json)
    url = "https://api.hume.ai/v0/tts/stream/json"
    
    try:
        start_time = time.perf_counter()
        first_byte_time = None
        
        headers = {
            "X-Hume-Api-Key": api_key,
            "Content-Type": "application/json",
        }
        
        payload = {
            "utterances": [
                {
                    "text": TTS_TEST_PHRASE,
                    "voice": {"id": voice_id},
                }
            ],
            "format": {"type": "wav"},
            "instant_mode": True,
        }
        
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return BenchmarkResult(
                    provider="hume",
                    error=f"HTTP {resp.status}: {error_text[:100]}",
                    success=False
                )
            
            # Read streaming JSON lines for first audio snippet
            async for line in resp.content:
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    # Look for audio data in snippet
                    if data.get("type") == "audio" or data.get("audio"):
                        first_byte_time = time.perf_counter()
                        break
                    if data.get("snippet") and data["snippet"].get("audio"):
                        first_byte_time = time.perf_counter()
                        break
                except json.JSONDecodeError:
                    continue
        
        if first_byte_time:
            ttfb = first_byte_time - start_time
            return BenchmarkResult(provider="hume", ttfb=ttfb, total_time=ttfb)
        else:
            return BenchmarkResult(provider="hume", error="No audio received", success=False)
            
    except Exception as e:
        return BenchmarkResult(provider="hume", error=str(e), success=False)


async def benchmark_inworld(session: aiohttp.ClientSession) -> BenchmarkResult:
    """Benchmark Inworld TTS TTFB using HTTP streaming."""
    api_key = API_KEYS["inworld"]
    voice_id = VOICE_IDS["inworld"]
    
    if not api_key:
        return BenchmarkResult(provider="inworld", error="No API key", success=False)
    
    url = "https://api.inworld.ai/tts/v1/voice:stream"
    
    try:
        start_time = time.perf_counter()
        first_byte_time = None
        
        headers = {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "text": TTS_TEST_PHRASE,
            "voiceId": voice_id,
            "modelId": "inworld-tts-1",
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": 24000,
            },
        }
        
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return BenchmarkResult(
                    provider="inworld",
                    error=f"HTTP {resp.status}: {error_text[:100]}",
                    success=False
                )
            
            # Read first chunk
            async for chunk in resp.content.iter_chunked(1024):
                if chunk:
                    first_byte_time = time.perf_counter()
                    break
        
        if first_byte_time:
            ttfb = first_byte_time - start_time
            return BenchmarkResult(provider="inworld", ttfb=ttfb, total_time=ttfb)
        else:
            return BenchmarkResult(provider="inworld", error="No audio received", success=False)
            
    except Exception as e:
        return BenchmarkResult(provider="inworld", error=str(e), success=False)


# Map provider names to benchmark functions
TTS_BENCHMARKS = {
    "cartesia": benchmark_cartesia,
    "rime": benchmark_rime,
    "deepgram": benchmark_deepgram_tts,
    "hume": benchmark_hume,
    "inworld": benchmark_inworld,
}


async def run_tts_benchmark(provider: str, num_runs: int = NUM_RUNS) -> BenchmarkSummary:
    """Run TTS benchmark for a specific provider."""
    if provider not in TTS_BENCHMARKS:
        return BenchmarkSummary(
            provider=provider,
            error_count=1,
            runs=[BenchmarkResult(provider=provider, error=f"Unknown provider: {provider}", success=False)]
        )
    
    benchmark_fn = TTS_BENCHMARKS[provider]
    results: List[BenchmarkResult] = []
    
    logger.info(f"Benchmarking TTS: {provider} ({num_runs} runs)")
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_runs):
            logger.debug(f"  Run {i+1}/{num_runs}")
            result = await benchmark_fn(session)
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


async def run_all_tts_benchmarks(providers: Optional[List[str]] = None) -> dict:
    """Run benchmarks for all TTS providers."""
    from benchmark.config import TTS_PROVIDERS
    
    providers = providers or TTS_PROVIDERS
    results = {}
    
    for provider in providers:
        summary = await run_tts_benchmark(provider)
        results[provider] = summary
        
        if summary.success_rate > 0:
            logger.info(
                f"TTS {provider}: TTFB avg={summary.ttfb_avg*1000:.1f}ms, "
                f"min={summary.ttfb_min*1000:.1f}ms, max={summary.ttfb_max*1000:.1f}ms, "
                f"success={summary.success_rate*100:.0f}%"
            )
        else:
            logger.warning(f"TTS {provider}: All runs failed")
    
    return results
