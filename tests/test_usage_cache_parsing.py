"""Unit tests for usage cache-split parsing in OpenAI-compatible client.

Covers three usage shapes seen in production:
- DeepSeek official: prompt_cache_hit_tokens / prompt_cache_miss_tokens at top level
- Tencent TokenHub: prompt_tokens_details.cached_tokens (OpenAI style, included in prompt_tokens)
- Others (e.g. MiniMax): no cache fields at all → hit=miss=0
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_agent.llm.openai_client import OpenAIClient


def _client() -> OpenAIClient:
    return OpenAIClient(api_key="test-key")


def _response(usage) -> SimpleNamespace:
    message = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage,
    )


def test_deepseek_style_fields_preserved():
    usage = SimpleNamespace(
        prompt_tokens=90300,
        completion_tokens=716,
        total_tokens=91016,
        prompt_cache_hit_tokens=83200,
        prompt_cache_miss_tokens=7100,
    )
    parsed = _client()._parse_response(_response(usage)).usage
    assert parsed.prompt_cache_hit_tokens == 83200
    assert parsed.prompt_cache_miss_tokens == 7100


def test_openai_style_cached_tokens_fallback():
    # TokenHub shape: no DeepSeek fields, cache reported via prompt_tokens_details
    usage = SimpleNamespace(
        prompt_tokens=17196,
        completion_tokens=10,
        total_tokens=17206,
        prompt_tokens_details=SimpleNamespace(cached_tokens=17152),
    )
    parsed = _client()._parse_response(_response(usage)).usage
    assert parsed.prompt_cache_hit_tokens == 17152
    assert parsed.prompt_cache_miss_tokens == 44


def test_no_cache_fields_stay_zero():
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=50, total_tokens=1050)
    parsed = _client()._parse_response(_response(usage)).usage
    assert parsed.prompt_cache_hit_tokens == 0
    assert parsed.prompt_cache_miss_tokens == 0


def test_cached_tokens_clamped_to_prompt():
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=5,
        total_tokens=105,
        prompt_tokens_details=SimpleNamespace(cached_tokens=500),
    )
    parsed = _client()._parse_response(_response(usage)).usage
    assert parsed.prompt_cache_hit_tokens == 100
    assert parsed.prompt_cache_miss_tokens == 0


def test_details_without_cached_tokens():
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=5,
        total_tokens=105,
        prompt_tokens_details=SimpleNamespace(cached_tokens=None),
    )
    parsed = _client()._parse_response(_response(usage)).usage
    assert parsed.prompt_cache_hit_tokens == 0
    assert parsed.prompt_cache_miss_tokens == 0
