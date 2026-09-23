"""Prove Admin UI settings take effect from valid Settings values.

Each test builds a fully validated ``Settings`` value (the same shape the
Admin UI persists) and asserts the setting reaches the component that must
honor it. Network clients are mocked, no fixed ports are used, and ``HOME``
is redirected so no live ``~/.fcc`` state is touched.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.settings import Settings
from free_claude_code.messaging.platforms.ports import (
    InboundMessageHandler,
    MessagingPlatformComponents,
)
from free_claude_code.messaging.session import SessionStore
from free_claude_code.providers.nvidia_nim.voice import (
    _NIM_ASR_MODEL_MAP,
    NvidiaNimTranscriber,
)
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.bootstrap import _create_transcriber
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


@pytest.fixture(autouse=True)
def _redirect_fcc_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


class _TrackingMessagingRuntime:
    name = "tracking"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def start(self) -> None:
        self.events.append("messaging.start")

    async def quiesce(self) -> None:
        self.events.append("messaging.quiesce")

    async def close(self) -> None:
        self.events.append("messaging.close")

    def on_message(self, handler: InboundMessageHandler) -> None:
        assert callable(handler)

    @property
    def is_connected(self) -> bool:
        return True


class _SessionStoreSpy:
    """Capture ``SessionStore`` constructor kwargs while building the real store."""

    def __init__(self) -> None:
        self.managed_message_cap: object = "unset"

    def __call__(
        self,
        storage_path: str = "sessions.json",
        *,
        managed_message_cap: int | None = None,
    ) -> SessionStore:
        self.managed_message_cap = managed_message_cap
        return SessionStore(
            storage_path=storage_path, managed_message_cap=managed_message_cap
        )


async def _start_workflow_with_settings(
    settings: Settings,
    store_spy: _SessionStoreSpy,
) -> tuple[ApplicationRuntime, MagicMock, MagicMock]:
    manager = ProviderRuntimeManager(settings)
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=None,
    )
    events: list[str] = []
    components = MessagingPlatformComponents(
        name="tracking",
        runtime=_TrackingMessagingRuntime(events),
        outbound=MagicMock(),
    )
    workflow = MagicMock()
    workflow.repair_restored_statuses = AsyncMock()
    workflow.close = AsyncMock()
    cli_manager = MagicMock()
    with (
        patch(
            "free_claude_code.cli.managed.ManagedClaudeSessionManager",
            return_value=cli_manager,
        ) as cli_constructor,
        patch(
            "free_claude_code.messaging.session.SessionStore",
            new=store_spy,
        ),
        patch(
            "free_claude_code.messaging.workflow.MessagingWorkflow",
            return_value=workflow,
        ) as workflow_constructor,
    ):
        runtime.http_started()
        await runtime._start_messaging_workflow(components)
    return runtime, cli_constructor, workflow_constructor


@pytest.mark.asyncio
async def test_allowed_dir_reaches_messaging_workflow_as_workspace_and_allowed_dirs(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed-workspace"
    settings = Settings(allowed_dir=str(allowed))
    store_spy = _SessionStoreSpy()

    runtime, cli_constructor, _ = await _start_workflow_with_settings(
        settings, store_spy
    )
    try:
        workspace_path = cli_constructor.call_args.kwargs["workspace_path"]
        assert workspace_path == os.path.abspath(str(allowed))
        assert cli_constructor.call_args.kwargs["allowed_dirs"] == [workspace_path]
        assert Path(workspace_path).is_dir()
    finally:
        assert await runtime.close() is True


@pytest.mark.asyncio
async def test_max_message_log_entries_per_chat_wires_session_store_cap() -> None:
    settings = Settings(max_message_log_entries_per_chat=2)
    store_spy = _SessionStoreSpy()

    runtime, _, workflow_constructor = await _start_workflow_with_settings(
        settings, store_spy
    )
    try:
        assert store_spy.managed_message_cap == 2
        session_store = workflow_constructor.call_args.kwargs["session_store"]
        assert isinstance(session_store, SessionStore)
        for index in range(3):
            session_store.record_message_id(
                "telegram", "chat", f"msg-{index}", "out", "note"
            )
        assert session_store.get_tracked_message_ids_for_chat("telegram", "chat") == [
            "msg-1",
            "msg-2",
        ]
    finally:
        assert await runtime.close() is True


def _fake_riva_client(
    transcript: str,
) -> tuple[SimpleNamespace, SimpleNamespace, MagicMock, MagicMock]:
    response = SimpleNamespace(
        results=[
            SimpleNamespace(
                alternatives=[SimpleNamespace(transcript=transcript)],
            )
        ]
    )
    asr_service = MagicMock()
    asr_service.offline_recognize.return_value = response
    auth = MagicMock()
    client = SimpleNamespace(
        Auth=MagicMock(return_value=auth),
        ASRService=MagicMock(return_value=asr_service),
        RecognitionConfig=MagicMock(return_value=object()),
    )
    riva = SimpleNamespace(__path__=[], client=client)
    return riva, client, asr_service, auth


@pytest.mark.asyncio
async def test_nim_whisper_model_selects_matching_riva_function_id(
    tmp_path: Path,
) -> None:
    model = "nvidia/parakeet-ctc-0.6b-es"
    settings = Settings(
        voice_note_enabled=True,
        whisper_device="nvidia_nim",
        whisper_model=model,
        nvidia_nim_api_key="test-nim-key",
    )
    expected_function_id, expected_language = _NIM_ASR_MODEL_MAP[model]

    transcriber = await _create_transcriber(settings)
    assert isinstance(transcriber, NvidiaNimTranscriber)
    try:
        wav = tmp_path / "note.wav"
        wav.write_bytes(b"audio bytes")
        riva, client, asr_service, auth = _fake_riva_client("hola mundo")
        with patch.dict(
            "sys.modules",
            {"riva": riva, "riva.client": client},
        ):
            assert await transcriber.transcribe(wav) == "hola mundo"
        assert expected_function_id == "a9eeee8f-b509-4712-b19d-194361fa5f31"
        assert expected_language == "es-US"
        client.Auth.assert_called_once_with(
            use_ssl=True,
            uri="grpc.nvcf.nvidia.com:443",
            metadata_args=[
                ["function-id", expected_function_id],
                ["authorization", "Bearer test-nim-key"],
            ],
        )
        client.RecognitionConfig.assert_called_once_with(
            language_code=expected_language,
            max_alternatives=1,
            verbatim_transcripts=True,
        )
        asr_service.offline_recognize.assert_called_once()
        auth.channel.close.assert_called_once_with()
    finally:
        await transcriber.close()
