"""Application-owned model metadata."""

from dataclasses import dataclass

from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningCapability


@dataclass(frozen=True, slots=True)
class ProviderModelInfo:
    """Provider model metadata used to shape the application model catalog."""

    model_id: str
    supports_thinking: bool | None = None
    input_modalities: frozenset[ModelInputModality] | None = None
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    reasoning_capability: ReasoningCapability = ReasoningCapability.UNKNOWN
    # Whether the provider accepts Requests that include
    # ``reasoning.encrypted_content`` and tolerates replaying reasoning items
    # carrying it back on later turns. Models whose upstream issues
    # encrypted reasoning bound to its own caller (and 400s on replay, e.g.
    # opencode gateways' muse-spark lane) must set this to False.
    supports_encrypted_reasoning: bool = True


@dataclass(frozen=True, slots=True)
class ProviderModelRefreshResult:
    """Per-provider outcome of one model-catalog refresh."""

    refreshed_provider_ids: tuple[str, ...] = ()
    failed_provider_ids: tuple[str, ...] = ()
