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
    _TOOL_IMAGE_STRIP_HINT,
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


class TestStripImagesInToolResults:
    """Images nested inside tool_result.content are also stripped (the screenshot bloat fix)."""

    def test_image_inside_tool_result_stripped(self):
        """Image block nested inside a tool_result is stripped; placeholder for empty result."""
        request = MessagesRequest.model_validate(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "screenshot_page",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": "x" * 5000,
                                        },
                                    },
                                ],
                            },
                        ],
                    },
                ],
            }
        )

        stripped = _strip_image_blocks_and_hint(request)

        assert stripped is True
        tool_result = request.messages[1].content[0]
        assert isinstance(tool_result.content, list)
        # The image should be gone; placeholder text added since it was image-only.
        assert len(tool_result.content) == 1
        assert tool_result.content[0].text == _TOOL_IMAGE_STRIP_HINT  # type: ignore[union-attr]

    def test_image_and_text_inside_tool_result_image_stripped(self):
        """When tool_result has text + image, image is stripped, text is kept."""
        request = MessagesRequest.model_validate(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t2",
                                "name": "Read",
                                "input": {"path": "x.png"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t2",
                                "content": [
                                    {"type": "text", "text": "file contents"},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": "y" * 1000,
                                        },
                                    },
                                ],
                            },
                        ],
                    },
                ],
            }
        )

        stripped = _strip_image_blocks_and_hint(request)

        assert stripped is True
        tool_result = request.messages[1].content[0]
        inner = tool_result.content
        assert isinstance(inner, list)
        block_types = [
            b["type"] if isinstance(b, dict) else b.type for b in inner  # type: ignore[union-attr]
        ]
        assert "image" not in block_types
        assert "text" in block_types
        texts = [
            b["text"] if isinstance(b, dict) else b.text  # type: ignore[union-attr]
            for b in inner
            if (b["type"] if isinstance(b, dict) else b.type) == "text"
        ]
        assert "file contents" in texts

    def test_image_in_tool_result_and_top_level_both_stripped(self):
        """Top-level image stripped + hint; tool_result image stripped + placeholder."""
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
                                "media_type": "image/jpeg",
                                "data": "top",
                            },
                        ),
                    ],
                ),
                Message(
                    role="assistant",
                    content=[
                        {
                            "type": "tool_use",  # type: ignore[dict-item]
                            "id": "t3",
                            "name": "screenshot_page",
                            "input": {},
                        }
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",  # type: ignore[dict-item]
                            "tool_use_id": "t3",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "z" * 500,
                                    },
                                },
                            ],
                        },
                    ],
                ),
            ],
        )

        stripped = _strip_image_blocks_and_hint(request)

        assert stripped is True

        # Top-level: image stripped + hint appended
        user_content = request.messages[0].content
        assert isinstance(user_content, list)
        user_texts = [b.text for b in user_content if b.type == "text"]  # type: ignore[union-attr]
        combined = " ".join(user_texts)
        assert "look at this" in combined
        assert "understand_image" in combined

        # Tool result: image stripped, placeholder added
        tool_result = request.messages[2].content[0]
        assert isinstance(tool_result.content, list)
        assert len(tool_result.content) == 1
        assert tool_result.content[0].text == _TOOL_IMAGE_STRIP_HINT  # type: ignore[union-attr]

    def test_tool_result_without_images_unchanged(self):
        """Tool results without images pass through unchanged."""
        request = MessagesRequest.model_validate(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t4",
                                "name": "Read",
                                "input": {"path": "x.txt"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t4",
                                "content": [
                                    {"type": "text", "text": "plain text result"},
                                ],
                            },
                        ],
                    },
                ],
            }
        )

        stripped = _strip_image_blocks_and_hint(request)

        assert stripped is False
        tool_result = request.messages[1].content[0]
        inner = tool_result.content
        assert isinstance(inner, list)
        assert inner[0]["text"] == "plain text result"
