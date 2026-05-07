"""Tests for _repair_truncated_json in openai_client.py."""

import pytest
from mini_agent.llm.openai_client import _repair_truncated_json


class TestRepairTruncatedJson:
    def test_valid_json_unchanged(self):
        """Valid JSON should parse correctly."""
        result = _repair_truncated_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_array(self):
        """Valid JSON array should parse correctly."""
        result = _repair_truncated_json('[{"a": 1}, {"b": 2}]')
        assert result == [{"a": 1}, {"b": 2}]

    def test_unterminated_string_value(self):
        """String value missing closing quote should be repaired."""
        result = _repair_truncated_json('{"key": "value')
        assert result == {"key": "value"}

    def test_unterminated_string_with_unicode(self):
        """Unterminated string with unicode chars should be repaired."""
        result = _repair_truncated_json('{"name": "测试内容')
        assert result == {"name": "测试内容"}

    def test_missing_closing_brace(self):
        """Missing closing brace should be repaired."""
        result = _repair_truncated_json('{"key": "value"')
        assert result == {"key": "value"}

    def test_nested_missing_braces(self):
        """Nested object missing closing braces."""
        result = _repair_truncated_json('{"outer": {"inner": "value"')
        assert result == {"outer": {"inner": "value"}}

    def test_missing_closing_bracket(self):
        """Array missing closing bracket."""
        result = _repair_truncated_json('[{"a": 1}, {"b": 2')
        assert result == [{"a": 1}, {"b": 2}]

    def test_trailing_comma_in_object(self):
        """Object with trailing comma."""
        result = _repair_truncated_json('{"a": 1, "b": 2,')
        assert result == {"a": 1, "b": 2}

    def test_completely_broken_json(self):
        """Completely unrepairable JSON should return None."""
        result = _repair_truncated_json('not json at all {{{')
        assert result is None

    def test_empty_string(self):
        """Empty string should return None."""
        result = _repair_truncated_json('')
        assert result is None

    def test_number_value(self):
        """Valid JSON with number values."""
        result = _repair_truncated_json('{"count": 42, "price": 3.14}')
        assert result == {"count": 42, "price": 3.14}

    def test_boolean_and_null(self):
        """Valid JSON with boolean and null."""
        result = _repair_truncated_json('{"active": true, "desc": null}')
        assert result == {"active": True, "desc": None}
