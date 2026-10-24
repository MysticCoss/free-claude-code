"""Pydantic models for the Anthropic Messages protocol."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _AnthropicBlockBase(BaseModel):
    """Pass through protocol extensions such as ``cache_control``."""

    model_config = ConfigDict(extra="allow")


class ContentBlockText(_AnthropicBlockBase):
    type: Literal["text"]
    text: str


class ContentBlockImage(_AnthropicBlockBase):
    type: Literal["image"]
    source: dict[str, Any]


class ContentBlockDocument(_AnthropicBlockBase):
    """Anthropic document block (e.g. PDF files via the Files API)."""

    type: Literal["document"]
    source: dict[str, Any]


class ContentBlockToolUse(_AnthropicBlockBase):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockToolResult(_AnthropicBlockBase):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[Any] | dict[str, Any]


class ContentBlockThinking(_AnthropicBlockBase):
    type: Literal["thinking"]
    thinking: str
    signature: str | None = None


class ContentBlockRedactedThinking(_AnthropicBlockBase):
    type: Literal["redacted_thinking"]
    data: str


class ContentBlockServerToolUse(_AnthropicBlockBase):
    """Anthropic server-side tool invocation (e.g. ``web_search``, ``web_fetch``)."""

    type: Literal["server_tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockWebSearchToolResult(_AnthropicBlockBase):
    type: Literal["web_search_tool_result"]
    tool_use_id: str
    content: Any


class ContentBlockWebFetchToolResult(_AnthropicBlockBase):
    type: Literal["web_fetch_tool_result"]
    tool_use_id: str
    content: Any


class SystemContent(_AnthropicBlockBase):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: (
        str
        | list[
            ContentBlockText
            | ContentBlockImage
            | ContentBlockDocument
            | ContentBlockToolUse
            | ContentBlockToolResult
            | ContentBlockThinking
            | ContentBlockRedactedThinking
            | ContentBlockServerToolUse
            | ContentBlockWebSearchToolResult
            | ContentBlockWebFetchToolResult
        ]
    )
    reasoning_content: str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _flatten_tool_notices(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [_tool_notice_as_text(block) for block in value]


def _tool_notice_as_text(block: object) -> object:
    """Render Claude Code's deferred-tool notices as text.

    ``tool_addition`` and ``tool_removal`` blocks announce tools appearing or
    disappearing mid-conversation. Translated providers have no equivalent
    blocks; the tools themselves are still advertised through ``tools``, so
    only the notice needs to survive.
    """
    if not isinstance(block, dict):
        return block
    tool = block.get("tool")
    tool = tool if isinstance(tool, dict) else {}
    if block.get("type") == "tool_addition":
        text = _tool_addition_text(tool)
    elif block.get("type") == "tool_removal":
        text = _tool_removal_text(tool)
    else:
        return block
    notice = {"type": "text", "text": text}
    if "cache_control" in block:
        notice["cache_control"] = block["cache_control"]
    return notice


def _tool_addition_text(tool: dict[str, object]) -> str:
    definition = tool.get("definition")
    if not isinstance(definition, dict):
        definition = tool
    name = definition.get("name")
    description = definition.get("description")
    text = f"Tool now available: {name}" if name else "A tool is now available."
    if description:
        text += f" — {description}"
    return text


def _tool_removal_text(tool: dict[str, object]) -> str:
    name = tool.get("name")
    return (
        f"Tool no longer available: {name}"
        if name
        else "A tool is no longer available."
    )


class Tool(_AnthropicBlockBase):
    name: str
    type: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None


class ThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool | None = True
    type: str | None = None
    budget_tokens: int | None = None
    display: Literal["summarized", "omitted", "updates"] | None = None


class MessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    original_model: str | None = Field(default=None, exclude=True)
    resolved_provider_model: str | None = Field(default=None, exclude=True)
    max_tokens: int | None = None
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[Tool] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: ThinkingConfig | None = None
    context_management: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    extra_body: dict[str, Any] | None = None
    betas: list[str] | None = Field(default=None, exclude=True)
    # Session / request identity forwarded as x-opencode-* headers by the
    # OpenCode Zen provider so the billing dashboard correlates requests.
    fcc_session_id: str | None = Field(default=None, exclude=True)
    fcc_request_id: str | None = Field(default=None, exclude=True)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _reject_boolean_output_limit(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("max_tokens must not be a boolean")
        return value


class TokenCountRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    original_model: str | None = Field(default=None, exclude=True)
    resolved_provider_model: str | None = Field(default=None, exclude=True)
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    tools: list[Tool] | None = None
    thinking: ThinkingConfig | None = None
    tool_choice: dict[str, Any] | None = None
    context_management: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    betas: list[str] | None = Field(default=None, exclude=True)


class TokenCountResponse(BaseModel):
    input_tokens: int


class NativeTokenCountMessage(BaseModel):
    """The outer message shape needed by local estimation, with opaque blocks."""

    model_config = ConfigDict(extra="allow", strict=True)

    role: str
    content: str | list[dict[str, Any]]


class NativeTokenCountRequest(BaseModel):
    """Count native inputs without applying compatibility protocol validation."""

    model_config = ConfigDict(extra="allow", strict=True)

    model: str = Field(min_length=1)
    messages: list[NativeTokenCountMessage]
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: list[
        ContentBlockText
        | ContentBlockToolUse
        | ContentBlockThinking
        | ContentBlockRedactedThinking
        | dict[str, Any]
    ]
    type: Literal["message"] = "message"
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: Usage
