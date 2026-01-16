#!/usr/bin/env python3
"""Main benchmark runner.

Runs all benchmarks and outputs results to JSON and console.

Usage:
    python -m benchmark.run_benchmarks [--tts] [--llm] [--stt] [--pipeline] [--all]
    
Examples:
    # Run all benchmarks
    python -m benchmark.run_benchmarks --all
    
    # Run only TTS and LLM benchmarks
    python -m benchmark.run_benchmarks --tts --llm
    
    # Run only pipeline benchmarks
    python -m benchmark.run_benchmarks --pipeline
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from benchmark.config import check_api_keys, BenchmarkSummary


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
    )


def summary_to_dict(summary) -> dict:
    """Convert a summary to a JSON-serializable dict."""
    if hasattr(summary, 'runs'):
        d = asdict(summary)
        # Remove detailed runs for cleaner output
        d.pop('runs', None)
        return d
    return asdict(summary)


async def run_tts_benchmarks() -> dict:
    """Run TTS benchmarks."""
    from benchmark.tts_benchmark import run_all_tts_benchmarks
    
    logger.info("=" * 60)
    logger.info("Running TTS Benchmarks")
    logger.info("=" * 60)
    
    results = await run_all_tts_benchmarks()
    return {k: summary_to_dict(v) for k, v in results.items()}


async def run_llm_benchmarks() -> dict:
    """Run LLM benchmarks."""
    from benchmark.llm_benchmark import run_all_llm_benchmarks
    
    logger.info("=" * 60)
    logger.info("Running LLM Benchmarks")
    logger.info("=" * 60)
    
    results = await run_all_llm_benchmarks()
    return {k: summary_to_dict(v) for k, v in results.items()}


async def run_stt_benchmarks() -> dict:
    """Run STT benchmarks."""
    from benchmark.stt_benchmark import run_all_stt_benchmarks
    
    logger.info("=" * 60)
    logger.info("Running STT Benchmarks")
    logger.info("=" * 60)
    
    results = await run_all_stt_benchmarks()
    return {k: summary_to_dict(v) for k, v in results.items()}


async def run_pipeline_benchmarks() -> dict:
    """Run pipeline benchmarks."""
    from benchmark.pipeline_benchmark import run_all_pipeline_benchmarks
    
    logger.info("=" * 60)
    logger.info("Running Pipeline Benchmarks")
    logger.info("=" * 60)
    
    results = await run_all_pipeline_benchmarks()
    return {k: summary_to_dict(v) for k, v in results.items()}


def print_results_table(results: dict):
    """Print results as a formatted table."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info("=" * 80)
    
    # TTS Results
    if "tts" in results:
        logger.info("")
        logger.info("TTS Providers (sorted by TTFB):")
        logger.info("-" * 60)
        tts_sorted = sorted(
            results["tts"].items(),
            key=lambda x: x[1].get("ttfb_avg", float("inf"))
        )
        for name, data in tts_sorted:
            if data.get("success_rate", 0) > 0:
                logger.info(
                    f"  {name:15} | TTFB: {data['ttfb_avg']*1000:6.1f}ms | "
                    f"min: {data['ttfb_min']*1000:6.1f}ms | max: {data['ttfb_max']*1000:6.1f}ms | "
                    f"success: {data['success_rate']*100:.0f}%"
                )
            else:
                logger.info(f"  {name:15} | FAILED (errors: {data.get('error_count', 'N/A')})")
    
    # LLM Results
    if "llm" in results:
        logger.info("")
        logger.info("LLM Models (sorted by TTFB):")
        logger.info("-" * 60)
        llm_sorted = sorted(
            results["llm"].items(),
            key=lambda x: x[1].get("ttfb_avg", float("inf"))
        )
        for name, data in llm_sorted:
            if data.get("success_rate", 0) > 0:
                logger.info(
                    f"  {name:15} | TTFB: {data['ttfb_avg']*1000:6.1f}ms | "
                    f"min: {data['ttfb_min']*1000:6.1f}ms | max: {data['ttfb_max']*1000:6.1f}ms | "
                    f"success: {data['success_rate']*100:.0f}%"
                )
            else:
                logger.info(f"  {name:15} | FAILED (errors: {data.get('error_count', 'N/A')})")
    
    # STT Results
    if "stt" in results:
        logger.info("")
        logger.info("STT Providers (sorted by TTFB):")
        logger.info("-" * 60)
        stt_sorted = sorted(
            results["stt"].items(),
            key=lambda x: x[1].get("ttfb_avg", float("inf"))
        )
        for name, data in stt_sorted:
            if data.get("success_rate", 0) > 0:
                logger.info(
                    f"  {name:15} | TTFB: {data['ttfb_avg']*1000:6.1f}ms | "
                    f"min: {data['ttfb_min']*1000:6.1f}ms | max: {data['ttfb_max']*1000:6.1f}ms | "
                    f"success: {data['success_rate']*100:.0f}%"
                )
            else:
                logger.info(f"  {name:15} | FAILED (errors: {data.get('error_count', 'N/A')})")
    
    # Pipeline Results
    if "pipeline" in results:
        logger.info("")
        logger.info("Pipelines (sorted by combined TTFB):")
        logger.info("-" * 60)
        
        def get_combined_ttfb(data):
            return (
                data.get("stt_ttfb_avg", 0) +
                data.get("llm_ttfb_avg", 0) +
                data.get("tts_ttfb_avg", 0)
            )
        
        pipeline_sorted = sorted(
            results["pipeline"].items(),
            key=lambda x: get_combined_ttfb(x[1]) if x[1].get("success_rate", 0) > 0 else float("inf")
        )
        for name, data in pipeline_sorted:
            if data.get("success_rate", 0) > 0:
                combined = get_combined_ttfb(data)
                logger.info(
                    f"  {name:15} | Combined: {combined*1000:6.1f}ms | "
                    f"STT: {data['stt_ttfb_avg']*1000:.0f}ms | "
                    f"LLM: {data['llm_ttfb_avg']*1000:.0f}ms | "
                    f"TTS: {data['tts_ttfb_avg']*1000:.0f}ms"
                )
            else:
                logger.info(f"  {name:15} | FAILED")
    
    logger.info("")
    logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(description="Run provider benchmarks")
    parser.add_argument("--tts", action="store_true", help="Run TTS benchmarks")
    parser.add_argument("--llm", action="store_true", help="Run LLM benchmarks")
    parser.add_argument("--stt", action="store_true", help="Run STT benchmarks")
    parser.add_argument("--pipeline", action="store_true", help="Run pipeline benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-o", "--output", type=str, help="Output JSON file path")
    args = parser.parse_args()
    
    # If no specific benchmarks selected, run all
    if not any([args.tts, args.llm, args.stt, args.pipeline, args.all]):
        args.all = True
    
    setup_logging(args.verbose)
    
    # Check API keys
    logger.info("Checking API keys...")
    api_keys = check_api_keys()
    for name, available in api_keys.items():
        status = "OK" if available else "MISSING"
        logger.info(f"  {name}: {status}")
    
    # Run benchmarks
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "api_keys_available": api_keys,
    }
    
    if args.all or args.tts:
        results["tts"] = await run_tts_benchmarks()
    
    if args.all or args.llm:
        results["llm"] = await run_llm_benchmarks()
    
    if args.all or args.stt:
        results["stt"] = await run_stt_benchmarks()
    
    if args.all or args.pipeline:
        results["pipeline"] = await run_pipeline_benchmarks()
    
    # Print summary table
    print_results_table(results)
    
    # Save to file
    output_path = args.output
    if not output_path:
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"benchmark_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
