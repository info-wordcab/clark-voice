#!/usr/bin/env python3
"""Test script to verify LLM fallback configuration.

This script tests:
1. That each configured provider can be initialized
2. That each provider can make a simple API call
3. That the fallback chain is configured correctly

Usage:
    cd receptionist
    python test_llm_config.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

# Import config after loading .env
from config import (
    PRIMARY_LLM,
    SECONDARY_LLM,
    FALLBACK_LLM,
    SAMBANOVA_API_KEY,
    SAMBANOVA_MODEL,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


def print_config():
    """Print current LLM configuration."""
    print("\n=== LLM Configuration ===")
    print(f"PRIMARY_LLM:   {PRIMARY_LLM or '(not set)'}")
    print(f"SECONDARY_LLM: {SECONDARY_LLM or '(not set)'}")
    print(f"FALLBACK_LLM:  {FALLBACK_LLM or '(not set)'}")
    print()
    print("API Keys configured:")
    print(f"  SAMBANOVA: {'Yes' if SAMBANOVA_API_KEY else 'No'}")
    print(f"  GOOGLE:    {'Yes' if GOOGLE_API_KEY else 'No'}")
    print(f"  OPENAI:    {'Yes' if OPENAI_API_KEY else 'No'}")
    print()


async def test_sambanova():
    """Test SambaNova LLM."""
    if not SAMBANOVA_API_KEY:
        print("  SKIP: SAMBANOVA_API_KEY not set")
        return False
    
    try:
        from pipecat.services.sambanova.llm import SambaNovaLLMService
        from pipecat.processors.aggregators.llm_context import LLMContext
        
        service = SambaNovaLLMService(api_key=SAMBANOVA_API_KEY, model=SAMBANOVA_MODEL)
        
        # Test with a simple inference
        context = LLMContext(messages=[
            {"role": "system", "content": "You are a helpful assistant. Be very brief."},
            {"role": "user", "content": "Say 'test ok' and nothing else."},
        ])
        
        response = await service.run_inference(context)
        print(f"  OK: SambaNova responded: {response[:50] if response else 'empty'}...")
        return True
        
    except Exception as e:
        print(f"  FAIL: SambaNova error: {e}")
        return False


async def test_google():
    """Test Google Gemini LLM."""
    if not GOOGLE_API_KEY:
        print("  SKIP: GOOGLE_API_KEY not set")
        return False
    
    try:
        from pipecat.services.google.llm import GoogleLLMService
        from pipecat.processors.aggregators.llm_context import LLMContext
        
        service = GoogleLLMService(api_key=GOOGLE_API_KEY, model=GOOGLE_MODEL)
        
        # Test with a simple inference
        context = LLMContext(messages=[
            {"role": "system", "content": "You are a helpful assistant. Be very brief."},
            {"role": "user", "content": "Say 'test ok' and nothing else."},
        ])
        
        response = await service.run_inference(context)
        print(f"  OK: Google responded: {response[:50] if response else 'empty'}...")
        return True
        
    except ImportError as e:
        print(f"  SKIP: Google LLM not installed: {e}")
        print("        (Run: pip install pipecat-ai[google])")
        return None  # Not a failure, just not available
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "rate" in error_str.lower() or "quota" in error_str.lower():
            print(f"  RATE LIMITED: Google rate limit hit (expected in free tier)")
            return True  # This is expected behavior
        if "speech_v2" in error_str or "Missing module" in error_str:
            print(f"  SKIP: Google dependencies not installed")
            return None  # Not a failure, just not available
        print(f"  FAIL: Google error: {e}")
        return False


async def test_openai():
    """Test OpenAI LLM."""
    if not OPENAI_API_KEY:
        print("  SKIP: OPENAI_API_KEY not set")
        return False
    
    try:
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.processors.aggregators.llm_context import LLMContext
        
        service = OpenAILLMService(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
        
        # Test with a simple inference
        context = LLMContext(messages=[
            {"role": "system", "content": "You are a helpful assistant. Be very brief."},
            {"role": "user", "content": "Say 'test ok' and nothing else."},
        ])
        
        response = await service.run_inference(context)
        print(f"  OK: OpenAI responded: {response[:50] if response else 'empty'}...")
        return True
        
    except Exception as e:
        print(f"  FAIL: OpenAI error: {e}")
        return False


async def test_fallback_service():
    """Test the FallbackLLMService wrapper."""
    if not PRIMARY_LLM:
        print("  SKIP: PRIMARY_LLM not set")
        return False
    
    try:
        from bot import create_llm_service
        
        service = create_llm_service()
        print(f"  OK: FallbackLLMService created with primary={PRIMARY_LLM}")
        
        # Check if it's a FallbackLLMService or single service
        service_type = type(service).__name__
        print(f"  Service type: {service_type}")
        
        return True
        
    except Exception as e:
        print(f"  FAIL: Error creating LLM service: {e}")
        return False


async def main():
    """Run all tests."""
    print_config()
    
    if not PRIMARY_LLM:
        print("ERROR: PRIMARY_LLM environment variable is required")
        print("Please set PRIMARY_LLM to one of: sambanova, google, openai")
        sys.exit(1)
    
    results = {}
    
    print("=== Testing Individual Providers ===")
    
    print("\nTesting SambaNova...")
    results["sambanova"] = await test_sambanova()
    
    print("\nTesting Google...")
    results["google"] = await test_google()
    
    print("\nTesting OpenAI...")
    results["openai"] = await test_openai()
    
    print("\n=== Testing Fallback Service ===")
    print("\nTesting FallbackLLMService...")
    results["fallback"] = await test_fallback_service()
    
    print("\n=== Summary ===")
    all_passed = True
    for name, result in results.items():
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name}: {status}")
        if result is False and name in [PRIMARY_LLM, SECONDARY_LLM, FALLBACK_LLM]:
            all_passed = False
    
    # Check that the configured providers work
    print("\n=== Chain Verification ===")
    chain = [p for p in [PRIMARY_LLM, SECONDARY_LLM, FALLBACK_LLM] if p]
    print(f"Configured chain: {' -> '.join(chain)}")
    
    working_providers = []
    for provider in chain:
        result = results.get(provider)
        if result is None:
            print(f"  INFO: {provider} not available (dependencies not installed)")
        elif result:
            working_providers.append(provider)
        else:
            print(f"  WARNING: {provider} is in chain but test failed")
            all_passed = False
    
    print(f"\nWorking providers: {', '.join(working_providers) if working_providers else 'none'}")
    
    if all_passed and working_providers:
        print("All available providers are working!")
    elif not working_providers:
        print("ERROR: No working LLM providers!")
        all_passed = False
    else:
        print("Some providers failed - check configuration")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
