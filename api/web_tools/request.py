"""Detect forced Anthropic web server tool requests."""

from __future__ import annotations

from api.models.anthropic import MessagesRequest, Tool


def request_text(request: MessagesRequest) -> str:
    """Join all user/assistant message content into one string for tool input parsing."""
    from .parsers import content_text

    return "\n".join(content_text(message.content) for message in request.messages)


def forced_tool_turn_text(request: MessagesRequest) -> str:
    """Text for parsing forced server-tool inputs: latest user turn only (avoids stale history)."""
    if not request.messages:
        return ""

    from .parsers import content_text

    for message in reversed(request.messages):
        if message.role == "user":
            return content_text(message.content)
    return ""


def forced_server_tool_name(request: MessagesRequest) -> str | None:
    """Return web_search or web_fetch only when tool_choice forces that server tool."""
    tc = request.tool_choice
    if not isinstance(tc, dict):
        return None
    if tc.get("type") != "tool":
        return None
    name = tc.get("name")
    if name in {"web_search", "web_fetch"}:
        return str(name)
    return None


def has_tool_named(request: MessagesRequest, name: str) -> bool:
    return any(tool.name == name for tool in request.tools or [])


def is_web_server_tool_request(request: MessagesRequest) -> bool:
    """True when the client forces a web server tool via tool_choice (not merely listed)."""
    forced = forced_server_tool_name(request)
    if forced is not None:
        return has_tool_named(request, forced)
    return False


def is_listed_web_server_tool_request(request: MessagesRequest) -> str | None:
    """Return tool name ('web_search' or 'web_fetch') when listed (not forced) AND
    the last user turn contains a plausible query or URL.  This lets the gateway
    eagerly execute server tools for OpenAI Chat upstreams that cannot process them.
    """
    from .parsers import extract_query, extract_url

    if not has_listed_anthropic_server_tools(request):
        return None
    text = forced_tool_turn_text(request)
    # Prefer web_fetch when the user explicitly supplied a URL.
    if has_tool_named(request, "web_fetch") and extract_url(text) != text.strip():
        return "web_fetch"
    # Otherwise, if web_search is listed and there's a non-trivial query, use it.
    if has_tool_named(request, "web_search"):
        query = extract_query(text)
        if len(query) >= 2:
            return "web_search"
    return None


def is_anthropic_server_tool_definition(tool: Tool) -> bool:
    """Whether ``tool`` refers to an Anthropic server tool (web_search / web_fetch family)."""
    name = (tool.name or "").strip()
    if name in ("web_search", "web_fetch"):
        return True
    typ = tool.type
    if isinstance(typ, str):
        return typ.startswith("web_search") or typ.startswith("web_fetch")
    return False


def has_listed_anthropic_server_tools(request: MessagesRequest) -> bool:
    """True when tools include web_search / web_fetch-style entries (listed, forced or not)."""
    return any(is_anthropic_server_tool_definition(t) for t in (request.tools or []))


def openai_chat_upstream_server_tool_error(
    request: MessagesRequest, *, web_tools_enabled: bool
) -> str | None:
    """Return a user-facing error when OpenAI Chat upstream cannot satisfy server-tool semantics."""
    forced = forced_server_tool_name(request)

    if not web_tools_enabled:
        # Web tools disabled entirely — reject any server tool usage.
        if forced:
            return (
                f"tool_choice forces Anthropic server tool {forced!r}, but local web server tools are "
                "disabled (ENABLE_WEB_SERVER_TOOLS=false). Enable them or use a native Anthropic "
                "Messages transport (e.g. open_router, ollama, lmstudio)."
            )
        if has_listed_anthropic_server_tools(request):
            return (
                "OpenAI Chat upstreams cannot use listed Anthropic server tools "
                "(web_search / web_fetch) without the local web server tool handler. Use a native "
                "Anthropic transport, set ENABLE_WEB_SERVER_TOOLS=true and force the tool with "
                "tool_choice, or remove these tools from the request."
            )
        return None

    # Web tools enabled — listed or forced are both OK (interceptor handles them).
    if forced and not has_tool_named(request, forced):
        return (
            f"tool_choice forces {forced!r} but that tool is not in the tools list"
        )
    return None
