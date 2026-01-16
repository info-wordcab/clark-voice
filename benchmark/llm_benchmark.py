"""LLM Provider Benchmarks.

Measures Time To First Byte (TTFB) for various LLM providers.
"""

import asyncio
import time
from typing import List, Optional

import aiohttp
from loguru import logger

from benchmark.config import (
    API_KEYS,
    LLM_MODELS,
    LLM_TEST_PROMPT,
    NUM_RUNS,
    PAUSE_BETWEEN_RUNS,
    BenchmarkResult,
    BenchmarkSummary,
)


async def benchmark_openai(session: aiohttp.ClientSession, model: str) -> BenchmarkResult:
    """Benchmark OpenAI LLM TTFB."""
    api_key = API_KEYS["openai"]
    
    if not api_key:
        return BenchmarkResult(provider="openai", model=model, error="No API key", success=False)
    
    url = "https://api.openai.com/v1/chat/completions"
    
    try:
        start_time = time.perf_counter()
        first_token_time = None
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful receptionist. Be brief."},
                {"role": "user", "content": LLM_TEST_PROMPT},
            ],
            "stream": True,
        }
        # GPT-5 models don't support max_tokens
        if not model.startswith("gpt-5"):
            payload["max_tokens"] = 100
        
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return BenchmarkResult(
                    provider="openai",
                    model=model,
                    error=f"HTTP {resp.status}: {error_text[:100]}",
                    success=False
                )
            
            # Read SSE stream for first content chunk
            async for line in resp.content:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    import json
                    try:
                        data = json.loads(line_str[6:])
                        if data.get("choices", [{}])[0].get("delta", {}).get("content"):
                            first_token_time = time.perf_counter()
                            break
                    except json.JSONDecodeError:
                        continue
        
        end_time = time.perf_counter()
        
        if first_token_time:
            ttfb = first_token_time - start_time
            return BenchmarkResult(
                provider="openai",
                model=model,
                ttfb=ttfb,
                total_time=end_time - start_time
            )
        else:
            return BenchmarkResult(
                provider="openai",
                model=model,
                error="No content received",
                success=False
            )
            
    except Exception as e:
        return BenchmarkResult(provider="openai", model=model, error=str(e), success=False)


async def benchmark_sambanova(session: aiohttp.ClientSession, model: str) -> BenchmarkResult:
    """Benchmark SambaNova LLM TTFB."""
    api_key = API_KEYS["sambanova"]
    
    if not api_key:
        return BenchmarkResult(provider="sambanova", model=model, error="No API key", success=False)
    
    # SambaNova uses OpenAI-compatible API
    url = "https://api.sambanova.ai/v1/chat/completions"
    
    try:
        start_time = time.perf_counter()
        first_token_time = None
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful receptionist. Be brief."},
                {"role": "user", "content": LLM_TEST_PROMPT},
            ],
            "stream": True,
            "max_tokens": 100,
        }
        
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return BenchmarkResult(
                    provider="sambanova",
                    model=model,
                    error=f"HTTP {resp.status}: {error_text[:100]}",
                    success=False
                )
            
            # Read SSE stream for first content chunk
            async for line in resp.content:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    import json
                    try:
                        data = json.loads(line_str[6:])
                        if data.get("choices", [{}])[0].get("delta", {}).get("content"):
                            first_token_time = time.perf_counter()
                            break
                    except json.JSONDecodeError:
                        continue
        
        end_time = time.perf_counter()
        
        if first_token_time:
            ttfb = first_token_time - start_time
            return BenchmarkResult(
                provider="sambanova",
                model=model,
                ttfb=ttfb,
                total_time=end_time - start_time
            )
        else:
            return BenchmarkResult(
                provider="sambanova",
                model=model,
                error="No content received",
                success=False
            )
            
    except Exception as e:
        return BenchmarkResult(provider="sambanova", model=model, error=str(e), success=False)


# Map provider names to benchmark functions
LLM_BENCHMARKS = {
    "openai": benchmark_openai,
    "sambanova": benchmark_sambanova,
}


async def run_llm_benchmark(
    name: str,
    provider: str,
    model: str,
    num_runs: int = NUM_RUNS
) -> BenchmarkSummary:
    """Run LLM benchmark for a specific provider/model."""
    if provider not in LLM_BENCHMARKS:
        return BenchmarkSummary(
            provider=name,
            model=model,
            error_count=1,
            runs=[BenchmarkResult(provider=name, model=model, error=f"Unknown provider: {provider}", success=False)]
        )
    
    benchmark_fn = LLM_BENCHMARKS[provider]
    results: List[BenchmarkResult] = []
    
    logger.info(f"Benchmarking LLM: {name} ({model}) ({num_runs} runs)")
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_runs):
            logger.debug(f"  Run {i+1}/{num_runs}")
            result = await benchmark_fn(session, model)
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
            provider=name,
            model=model,
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
            provider=name,
            model=model,
            success_rate=0.0,
            error_count=len(results),
            runs=results,
        )
    
    return summary


async def run_all_llm_benchmarks(models: Optional[dict] = None) -> dict:
    """Run benchmarks for all LLM models."""
    models = models or LLM_MODELS
    results = {}
    
    for name, config in models.items():
        summary = await run_llm_benchmark(
            name=name,
            provider=config["provider"],
            model=config["model"],
        )
        results[name] = summary
        
        if summary.success_rate > 0:
            logger.info(
                f"LLM {name}: TTFB avg={summary.ttfb_avg*1000:.1f}ms, "
                f"min={summary.ttfb_min*1000:.1f}ms, max={summary.ttfb_max*1000:.1f}ms, "
                f"success={summary.success_rate*100:.0f}%"
            )
        else:
            logger.warning(f"LLM {name}: All runs failed")
    
    return results
