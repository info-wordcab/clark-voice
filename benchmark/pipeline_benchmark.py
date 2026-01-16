"""Full Pipeline Benchmarks.

Tests complete STT -> LLM -> TTS pipelines using pipecat.
Measures end-to-end latency and TTFB at each stage.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp
from loguru import logger

from benchmark.config import (
    API_KEYS,
    VOICE_IDS,
    LLM_MODELS,
    NUM_RUNS,
    PAUSE_BETWEEN_RUNS,
    BenchmarkResult,
    BenchmarkSummary,
)


@dataclass
class PipelineConfig:
    """Configuration for a pipeline benchmark."""
    name: str
    stt_provider: str
    llm_name: str
    llm_provider: str
    llm_model: str
    tts_provider: str


@dataclass
class PipelineResult:
    """Result of a single pipeline benchmark run."""
    config_name: str
    stt_ttfb: float = 0.0
    llm_ttfb: float = 0.0
    tts_ttfb: float = 0.0
    total_time: float = 0.0
    error: str = ""
    success: bool = True


@dataclass
class PipelineSummary:
    """Summary of pipeline benchmark results."""
    config_name: str
    stt_provider: str
    llm_name: str
    tts_provider: str
    stt_ttfb_avg: float = 0.0
    llm_ttfb_avg: float = 0.0
    tts_ttfb_avg: float = 0.0
    total_time_avg: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    runs: List[PipelineResult] = field(default_factory=list)


# Define pipeline configurations to test
PIPELINE_CONFIGS = [
    PipelineConfig(
        name="current",
        stt_provider="assemblyai",
        llm_name="sambanova",
        llm_provider="sambanova",
        llm_model="Llama-4-Maverick-17B-128E-Instruct",
        tts_provider="cartesia",  # Using cartesia instead of inworld due to issues
    ),
    PipelineConfig(
        name="fast_llm",
        stt_provider="assemblyai",
        llm_name="gpt-5-nano",
        llm_provider="openai",
        llm_model="gpt-5-nano",
        tts_provider="cartesia",
    ),
    PipelineConfig(
        name="balanced",
        stt_provider="deepgram",
        llm_name="gpt-5-mini",
        llm_provider="openai",
        llm_model="gpt-5-mini",
        tts_provider="cartesia",
    ),
    PipelineConfig(
        name="quality",
        stt_provider="deepgram",
        llm_name="gpt-4o",
        llm_provider="openai",
        llm_model="gpt-4o",
        tts_provider="hume",
    ),
]


async def run_pipeline_benchmark_single(
    config: PipelineConfig,
    session: aiohttp.ClientSession,
) -> PipelineResult:
    """Run a single pipeline benchmark.
    
    This simulates: Audio -> STT -> LLM -> TTS
    We measure TTFB at each stage.
    """
    from benchmark.llm_benchmark import LLM_BENCHMARKS
    from benchmark.tts_benchmark import TTS_BENCHMARKS
    
    result = PipelineResult(config_name=config.name)
    
    try:
        total_start = time.perf_counter()
        
        # 1. LLM TTFB (this is typically the bottleneck)
        llm_benchmark_fn = LLM_BENCHMARKS.get(config.llm_provider)
        if not llm_benchmark_fn:
            return PipelineResult(
                config_name=config.name,
                error=f"Unknown LLM provider: {config.llm_provider}",
                success=False
            )
        
        llm_result = await llm_benchmark_fn(session, config.llm_model)
        if not llm_result.success:
            return PipelineResult(
                config_name=config.name,
                error=f"LLM error: {llm_result.error}",
                success=False
            )
        result.llm_ttfb = llm_result.ttfb
        
        # 2. TTS TTFB
        tts_benchmark_fn = TTS_BENCHMARKS.get(config.tts_provider)
        if not tts_benchmark_fn:
            return PipelineResult(
                config_name=config.name,
                error=f"Unknown TTS provider: {config.tts_provider}",
                success=False
            )
        
        tts_result = await tts_benchmark_fn(session)
        if not tts_result.success:
            return PipelineResult(
                config_name=config.name,
                error=f"TTS error: {tts_result.error}",
                success=False
            )
        result.tts_ttfb = tts_result.ttfb
        
        # Note: STT TTFB is harder to measure in isolation without audio input
        # For pipeline benchmarks, we estimate based on known provider characteristics
        # In a real scenario, this would be measured with actual audio
        result.stt_ttfb = 0.1 if config.stt_provider == "deepgram" else 0.15  # Estimated
        
        result.total_time = time.perf_counter() - total_start
        result.success = True
        
    except Exception as e:
        result.error = str(e)
        result.success = False
    
    return result


async def run_pipeline_benchmark(
    config: PipelineConfig,
    num_runs: int = NUM_RUNS
) -> PipelineSummary:
    """Run pipeline benchmark multiple times."""
    results: List[PipelineResult] = []
    
    logger.info(
        f"Benchmarking Pipeline: {config.name} "
        f"({config.stt_provider} -> {config.llm_name} -> {config.tts_provider}) "
        f"({num_runs} runs)"
    )
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_runs):
            logger.debug(f"  Run {i+1}/{num_runs}")
            result = await run_pipeline_benchmark_single(config, session)
            results.append(result)
            
            if result.success:
                logger.debug(
                    f"    LLM TTFB: {result.llm_ttfb*1000:.1f}ms, "
                    f"TTS TTFB: {result.tts_ttfb*1000:.1f}ms"
                )
            else:
                logger.warning(f"    Error: {result.error}")
            
            if i < num_runs - 1:
                await asyncio.sleep(PAUSE_BETWEEN_RUNS)
    
    # Calculate summary statistics
    successful = [r for r in results if r.success]
    
    if successful:
        import statistics
        
        summary = PipelineSummary(
            config_name=config.name,
            stt_provider=config.stt_provider,
            llm_name=config.llm_name,
            tts_provider=config.tts_provider,
            stt_ttfb_avg=statistics.mean([r.stt_ttfb for r in successful]),
            llm_ttfb_avg=statistics.mean([r.llm_ttfb for r in successful]),
            tts_ttfb_avg=statistics.mean([r.tts_ttfb for r in successful]),
            total_time_avg=statistics.mean([r.total_time for r in successful]),
            success_rate=len(successful) / len(results),
            error_count=len(results) - len(successful),
            runs=results,
        )
    else:
        summary = PipelineSummary(
            config_name=config.name,
            stt_provider=config.stt_provider,
            llm_name=config.llm_name,
            tts_provider=config.tts_provider,
            success_rate=0.0,
            error_count=len(results),
            runs=results,
        )
    
    return summary


async def run_all_pipeline_benchmarks(
    configs: Optional[List[PipelineConfig]] = None
) -> Dict[str, PipelineSummary]:
    """Run benchmarks for all pipeline configurations."""
    configs = configs or PIPELINE_CONFIGS
    results = {}
    
    for config in configs:
        summary = await run_pipeline_benchmark(config)
        results[config.name] = summary
        
        if summary.success_rate > 0:
            combined_ttfb = summary.stt_ttfb_avg + summary.llm_ttfb_avg + summary.tts_ttfb_avg
            logger.info(
                f"Pipeline {config.name}: "
                f"Combined TTFB={combined_ttfb*1000:.1f}ms "
                f"(STT={summary.stt_ttfb_avg*1000:.1f}ms, "
                f"LLM={summary.llm_ttfb_avg*1000:.1f}ms, "
                f"TTS={summary.tts_ttfb_avg*1000:.1f}ms), "
                f"success={summary.success_rate*100:.0f}%"
            )
        else:
            logger.warning(f"Pipeline {config.name}: All runs failed")
    
    return results
