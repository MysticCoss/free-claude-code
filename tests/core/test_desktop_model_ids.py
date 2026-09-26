import pytest

from free_claude_code.core.gateway_model_ids import (
    decode_gateway_model_id,
    desktop_model_id,
    munge_desktop_ref,
    unmunge_desktop_ref,
)


@pytest.mark.parametrize(
    "ref",
    [
        "nvidia_nim/nemotron-3.5",
        "open_router/qwen/qwen3",
        "openai/gpt-5",
        "open_router/模型/one",
        "deepseek/deepseek-chat",
        "azure_openai/gpt-4o",
        "ollama/llama3.1",
        "kimi/k2-thinking",
        "zai/glm-5",
        "m/ab",
        "opencode_go/muse-spark-1.3-contributor-free",
    ],
)
@pytest.mark.parametrize("no_thinking", [False, True])
def test_desktop_ids_roundtrip_without_vendor_fragments(ref, no_thinking):
    wire = desktop_model_id(ref, no_thinking=no_thinking)
    assert wire.startswith("claude-")
    assert "[" not in wire
    assert not any(part in wire for part in ("gpt", "qwen", "nemotron", "openai"))
    decoded = decode_gateway_model_id(wire)
    assert decoded is not None
    assert f"{decoded.provider_id}/{decoded.provider_model}" == ref
    assert decoded.force_reasoning_off is no_thinking
    assert desktop_model_id(ref, no_thinking=no_thinking) == wire


@pytest.mark.parametrize(
    ("ref", "munged"),
    [
        ("deepseek/deepseek-chat", "d-iepseek/d-iepseek-chat"),
        ("zai/glm-5", "z-ei/g-lm-5"),
        ("openai/gpt-5", "o-pinai/g-pt-5"),
        ("m/ab", "m+/a-b"),
    ],
)
def test_desktop_munging_matches_documented_scheme(ref, munged):
    assert munge_desktop_ref(ref) == munged
    assert unmunge_desktop_ref(munged) == ref
    assert munge_desktop_ref(unmunge_desktop_ref(munged)) == munged


@pytest.mark.parametrize(
    "bad",
    [
        # Missing hyphen mark.
        "deepseek",
        "d",
        # A hyphen where the codec would never emit one.
        "-eepseek",
        # Non-ref payload that must not decode to a provider/model pair.
        "noprovider",
    ],
)
def test_malformed_munged_refs_are_rejected(bad):
    with pytest.raises(ValueError):
        unmunge_desktop_ref(bad)


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "zz",
        "not-a-ref",
        "d-iepseek",
        "d-iepseek+",
        "-eepseek/deepseek-chat",
        "deepseek+",
    ],
)
@pytest.mark.parametrize("prefix", ["claude-fcc/", "claude-3-fcc/"])
def test_malformed_reserved_ids_do_not_fall_through(prefix, suffix):
    with pytest.raises(ValueError):
        decode_gateway_model_id(prefix + suffix)
