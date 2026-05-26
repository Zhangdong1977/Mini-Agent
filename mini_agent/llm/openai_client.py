"""OpenAI LLM client implementation."""

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from ..retry import RetryConfig, async_retry
from ..schema import FunctionCall, LLMResponse, Message, TokenUsage, ToolCall
from .base import LLMClientBase

logger = logging.getLogger(__name__)


def _repair_truncated_json(json_str: str) -> Any | None:
    """Attempt to repair common JSON truncation errors from LLM output.

    Returns parsed object on success, None if unrepairable.
    """
    s = json_str.strip()

    if not s:
        return None

    # Strategy 1: Unterminated string — append closing quote
    if not s.endswith('"'):
        try:
            return json.loads(s + '"')
        except json.JSONDecodeError:
            pass

    # Strategy 2: Balance unclosed braces and brackets
    open_braces = s.count('{') - s.count('}')
    open_brackets = s.count('[') - s.count(']')

    if open_braces > 0 or open_brackets > 0:
        # Check if we're inside an unterminated string value
        stripped = s.rstrip()
        if stripped and stripped[-1] not in ('}', ']', '"', ':', ',', ' ', '\n', '\t'):
            # Value without closing quote — try adding quote + closing brackets
            try:
                repaired = s + '"' + '}' * open_braces + ']' * open_brackets
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Just close brackets/braces
        try:
            repaired = s + '}' * open_braces + ']' * open_brackets
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Trailing comma — trim and close
    if s.endswith(','):
        try:
            return json.loads(s.rstrip(',') + '}' * open_braces + ']' * open_brackets)
        except json.JSONDecodeError:
            pass

    return None


class OpenAIClient(LLMClientBase):
    """LLM client using OpenAI's protocol.

    This client uses the official OpenAI SDK and supports:
    - Reasoning content (via reasoning_split=True)
    - Tool calling
    - Retry logic
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/v1",
        model: str = "MiniMax-M2.5",
        retry_config: RetryConfig | None = None,
        timeout: float | None = None,
        reasoning_split: bool = True,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: API key for authentication
            api_base: Base URL for the API (default: MiniMax OpenAI endpoint)
            model: Model name to use (default: MiniMax-M2.5)
            retry_config: Optional retry configuration
            timeout: Optional timeout in seconds for API calls
            reasoning_split: Whether to enable reasoning_split for MiniMax (default: True)
        """
        super().__init__(api_key, api_base, model, retry_config)
        self.reasoning_split = reasoning_split

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
        )

    async def _make_api_request(
        self,
        api_messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Execute API request (core method that can be retried).

        Args:
            api_messages: List of messages in OpenAI format
            tools: Optional list of tools
            max_tokens: Optional maximum tokens for response output

        Returns:
            OpenAI ChatCompletion response (full response including usage)

        Raises:
            Exception: API call failed
        """
        params = {
            "model": self.model,
            "messages": api_messages,
        }

        # MiniMax-specific: enable reasoning_split to separate thinking content
        if self.reasoning_split:
            params["extra_body"] = {"reasoning_split": True}

        if tools:
            params["tools"] = self._convert_tools(tools)

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Use OpenAI SDK's chat.completions.create
        response = await self.client.chat.completions.create(**params)
        # Return full response to access usage info
        return response

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to OpenAI format.

        Args:
            tools: List of Tool objects or dicts

        Returns:
            List of tools in OpenAI dict format
        """
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                # If already a dict, check if it's in OpenAI format
                if "type" in tool and tool["type"] == "function":
                    result.append(tool)
                else:
                    # Assume it's in Anthropic format, convert to OpenAI
                    result.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "description": tool["description"],
                                "parameters": tool["input_schema"],
                            },
                        }
                    )
            elif hasattr(tool, "to_openai_schema"):
                # Tool object with to_openai_schema method
                result.append(tool.to_openai_schema())
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")
        return result

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to OpenAI format.

        Args:
            messages: List of internal Message objects

        Returns:
            Tuple of (system_message, api_messages)
            Note: OpenAI includes system message in the messages array
        """
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                # OpenAI includes system message in messages array
                api_messages.append({"role": "system", "content": msg.content})
                continue

            # For user messages
            if msg.role == "user":
                api_messages.append({"role": "user", "content": msg.content})

            # For assistant messages
            elif msg.role == "assistant":
                assistant_msg = {"role": "assistant"}

                # Add content if present
                if msg.content:
                    assistant_msg["content"] = msg.content

                # Add tool calls if present
                if msg.tool_calls:
                    tool_calls_list = []
                    for tool_call in msg.tool_calls:
                        tool_calls_list.append(
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": json.dumps(tool_call.function.arguments),
                                },
                            }
                        )
                    assistant_msg["tool_calls"] = tool_calls_list

                # IMPORTANT: Add reasoning_details if thinking is present
                # This is CRITICAL for Interleaved Thinking to work properly!
                # The complete response_message (including reasoning_details) must be
                # preserved in Message History and passed back to the model in the next turn.
                # This ensures the model's chain of thought is not interrupted.
                if msg.thinking:
                    assistant_msg["reasoning_details"] = [{"text": msg.thinking}]

                api_messages.append(assistant_msg)

            # For tool result messages
            elif msg.role == "tool":
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )

        return None, api_messages

    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare the request for OpenAI API.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            Dictionary containing request parameters
        """
        _, api_messages = self._convert_messages(messages)

        return {
            "api_messages": api_messages,
            "tools": tools,
        }

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI response into LLMResponse.

        Args:
            response: OpenAI ChatCompletion response (full response object)

        Returns:
            LLMResponse object
        """
        # Get message from response
        message = response.choices[0].message

        # Extract text content
        text_content = message.content or ""

        # Extract thinking content from reasoning_details
        thinking_content = ""
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            # reasoning_details is a list of reasoning blocks
            for detail in message.reasoning_details:
                if hasattr(detail, "text"):
                    thinking_content += detail.text

        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                # Parse arguments from JSON string with repair fallback
                raw_args = tool_call.function.arguments
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JSON decode error for tool_call {tool_call.function.name}: {e}. "
                        f"Attempting repair..."
                    )
                    repaired = _repair_truncated_json(raw_args)
                    if repaired is not None:
                        arguments = repaired
                        logger.info(
                            f"Successfully repaired JSON for tool_call {tool_call.function.name}"
                        )
                    else:
                        logger.error(
                            f"Failed to repair JSON for tool_call {tool_call.function.name}, "
                            f"skipping this tool call. Raw args (first 200 chars): "
                            f"{raw_args[:200]}"
                        )
                        continue

                tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        type="function",
                        function=FunctionCall(
                            name=tool_call.function.name,
                            arguments=arguments,
                        ),
                    )
                )

        # Extract token usage from response
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return LLMResponse(
            content=text_content,
            thinking=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason="stop",  # OpenAI doesn't provide finish_reason in the message
            usage=usage,
        )

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate response from OpenAI LLM.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools
            max_tokens: Optional maximum tokens for response output (prevents truncation)

        Returns:
            LLMResponse containing the generated content
        """
        # Prepare request
        request_params = self._prepare_request(messages, tools)

        # Make API request with retry logic
        if self.retry_config.enabled:
            # Apply retry logic
            retry_decorator = async_retry(config=self.retry_config, on_retry=self.retry_callback)
            api_call = retry_decorator(self._make_api_request)
            response = await api_call(
                request_params["api_messages"],
                request_params["tools"],
                max_tokens,
            )
        else:
            # Don't use retry
            response = await self._make_api_request(
                request_params["api_messages"],
                request_params["tools"],
                max_tokens,
            )

        # Parse and return response
        return self._parse_response(response)
