"""Tests for OpenCode Go DeepSeek image handling.

The opencode_go provider uses the OpenAI chat conversion path. For DeepSeek V4
models (which lack vision), image blocks are stripped and a hint is injected
before the converter runs, preventing OpenAIConversionError.
"""

from api.models.anthropic import (
    ContentBlockImage,
    ContentBlockText,
    Message,
    MessagesRequest,
)
from providers.exceptions import InvalidRequestError
from providers.opencode.request import (
    _strip_image_blocks_and_hint,
    build_request_body,
)


def test_strip_image_blocks_and_hint_strips_images():
    """Images are removed from messages and a hint block is appended."""
    request = MessagesRequest(
        model="deepseek-v4-pro",
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockText(type="text", text="look at this"),
                    ContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc",
                        },
                    ),
                ],
            ),
        ],
    )

    stripped = _strip_image_blocks_and_hint(request)

    assert stripped is True
    content = request.messages[0].content
    assert isinstance(content, list)
    block_types = [b.type for b in content]  # type: ignore[union-attr]
    assert "image" not in block_types
    assert "text" in block_types
    texts = [b.text for b in content if b.type == "text"]  # type: ignore[union-attr]
    combined = " ".join(texts)
    assert "look at this" in combined
    assert "understand_image" in combined


def test_strip_image_blocks_and_hint_no_images_unchanged():
    """Messages without images are unchanged and returns False."""
    request = MessagesRequest(
        model="deepseek-v4-pro",
        messages=[
            Message(
                role="user",
                content=[ContentBlockText(type="text", text="hello")],
            ),
        ],
    )

    stripped = _strip_image_blocks_and_hint(request)

    assert stripped is False
    content = request.messages[0].content
    assert isinstance(content, list)
    assert content[0].text == "hello"  # type: ignore[union-attr]


def test_strip_image_blocks_and_hint_string_content_unchanged():
    """Messages with string content (not list) are skipped."""
    request = MessagesRequest(
        model="deepseek-v4-pro",
        messages=[Message(role="user", content="plain text")],
    )

    stripped = _strip_image_blocks_and_hint(request)

    assert stripped is False
    assert request.messages[0].content == "plain text"


def test_strip_image_blocks_and_hint_empty_messages():
    """Empty messages list returns False."""
    request = MessagesRequest(model="deepseek-v4-pro", messages=[])

    stripped = _strip_image_blocks_and_hint(request)

    assert stripped is False


class TestBuildRequestBodyDeepSeekV4:
    """Tests for build_request_body with DeepSeek V4 models (pro and flash)."""

    def test_image_stripped_and_hint_injected(self):
        """Image blocks are stripped and hint is injected before conversion."""
        request = MessagesRequest(
            model="deepseek-v4-pro",
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentBlockText(type="text", text="describe this image"),
                        ContentBlockImage(
                            type="image",
                            source={
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "img123",
                            },
                        ),
                    ],
                ),
            ],
        )

        body = build_request_body(request, thinking_enabled=False)

        messages = body["messages"]
        # Should be one user message with only text (image stripped + hint appended)
        user_msg = messages[0]
        assert user_msg["role"] == "user"
        user_text = user_msg["content"]
        assert "describe this image" in user_text
        assert "understand_image" in user_text

    def test_image_only_message_gets_hint(self):
        """An image-only user message gets the hint as content."""
        request = MessagesRequest(
            model="deepseek-v4-pro",
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentBlockImage(
                            type="image",
                            source={
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "x",
                            },
                        ),
                    ],
                ),
            ],
        )

        body = build_request_body(request, thinking_enabled=False)

        user_msg = body["messages"][0]
        assert "understand_image" in user_msg["content"]

    def test_no_images_passes_through_normally(self):
        """Messages without images pass through unchanged for DeepSeek V4."""
        request = MessagesRequest(
            model="deepseek-v4-pro",
            messages=[Message(role="user", content="hello world")],
        )

        body = build_request_body(request, thinking_enabled=False)

        user_msg = body["messages"][0]
        assert user_msg["content"] == "hello world"

    def test_deepseek_v4_flash_also_strips_images(self):
        """deepseek-v4-flash also gets the image stripping treatment."""
        request = MessagesRequest(
            model="deepseek-v4-flash",
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentBlockText(type="text", text="analyze"),
                        ContentBlockImage(
                            type="image",
                            source={
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "y",
                            },
                        ),
                    ],
                ),
            ],
        )

        body = build_request_body(request, thinking_enabled=False)

        user_text = body["messages"][0]["content"]
        assert "analyze" in user_text
        assert "understand_image" in user_text

    def test_thinking_enabled_preserves_reasoning_effort(self):
        """When thinking is enabled for DeepSeek V4, reasoning_effort is still set."""
        request = MessagesRequest.model_validate(
            {
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "hello"}],
                "thinking": {"type": "enabled", "budget_tokens": 16000},
            }
        )

        body = build_request_body(request, thinking_enabled=True)

        assert "reasoning_effort" in body


class TestBuildRequestBodyNonDeepSeek:
    """Non-DeepSeek models should still error on images."""

    def test_non_deepseek_model_with_image_raises(self):
        """Non-DeepSeek models still raise InvalidRequestError for images."""
        request = MessagesRequest(
            model="qwen3.7-max",
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentBlockText(type="text", text="look"),
                        ContentBlockImage(
                            type="image",
                            source={
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "z",
                            },
                        ),
                    ],
                ),
            ],
        )

        import pytest

        with pytest.raises(InvalidRequestError, match="image"):
            build_request_body(request, thinking_enabled=False)

    def test_non_deepseek_model_no_image_passes(self):
        """Non-DeepSeek models without images pass through normally."""
        request = MessagesRequest(
            model="qwen3.7-max",
            messages=[Message(role="user", content="hi")],
        )

        body = build_request_body(request, thinking_enabled=False)

        assert body["messages"][0]["content"] == "hi"
