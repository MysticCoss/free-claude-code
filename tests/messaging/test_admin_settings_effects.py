"""Prove Admin UI settings take effect from valid Settings values.

Each test builds a fully validated ``Settings`` value (the same shape the
Admin UI persists) and asserts the setting reaches the component that must
honor it. Network clients are mocked, no fixed ports are used, and ``HOME``
is redirected so no live ``~/.fcc`` state is touched.
"""

import asyncio
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
from free_claude_code.providers.nvidia_nim.voice import (
    _NIM_ASR_MODEL_MAP,
    NvidiaNimTranscriber,
)
from free_claude_code.runtime.bootstrap import _create_transcriber
from free_claude_code.runtime.messaging_service import MessagingService
from free_claude_code.runtime.messaging_sqlite import SQLiteMessagingStore
from free_claude_code.runtime.sqlite_database import SQLiteDatabase


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


async def _start_workflow_with_settings(
    settings: Settings,
    tmp_path: Path,
) -> tuple[MessagingService, SQLiteDatabase, MagicMock, MagicMock]:
    database_path = tmp_path / "fcc.db"
    database = SQLiteDatabase(database_path, database_path.with_suffix(".lock"))
    http_ready = asyncio.Event()
    http_ready.set()
    service = MessagingService(
        settings=lambda: settings,
        database=database,
        http_ready=http_ready,
        is_draining=lambda: False,
        transcriber=None,
        transcriber_factory=None,
    )
    events: list[str] = []
    components = MessagingPlatformComponents(
        name="tracking",
        runtime=_TrackingMessagingRuntime(events),
        outbound=MagicMock(),
    )
    workflow = MagicMock()
    workflow.restore = AsyncMock()
    workflow.repair_restored_statuses = AsyncMock()
    workflow.close = AsyncMock()
    cli_manager = MagicMock()
    with (
        patch(
            "free_claude_code.cli.managed.ManagedClaudeSessionManager",
            return_value=cli_manager,
        ) as cli_constructor,
        patch(
            "free_claude_code.messaging.workflow.MessagingWorkflow",
            return_value=workflow,
        ) as workflow_constructor,
    ):
        service._messaging_import_task = asyncio.create_task(
            service._initialize_messaging_storage()
        )
        await service._start_messaging_workflow(components)
    return service, database, cli_constructor, workflow_constructor


async def _close(service: MessagingService, database: SQLiteDatabase) -> None:
    service.cancel_background_tasks()
    await service.wait_background_tasks()
    await database.close()


@pytest.mark.asyncio
async def test_allowed_dir_reaches_messaging_workflow_as_workspace_and_allowed_dirs(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed-workspace"
    settings = Settings(allowed_dir=str(allowed))

    service, database, cli_constructor, _ = await _start_workflow_with_settings(
        settings, tmp_path
    )
    try:
        workspace_path = cli_constructor.call_args.kwargs["workspace_path"]
        assert workspace_path == os.path.abspath(str(allowed))
        assert cli_constructor.call_args.kwargs["allowed_dirs"] == [workspace_path]
        assert Path(workspace_path).is_dir()
    finally:
        await _close(service, database)


@pytest.mark.asyncio
async def test_max_message_log_entries_per_chat_wires_session_store_cap(
    tmp_path: Path,
) -> None:
    settings = Settings(max_message_log_entries_per_chat=2)

    service, database, _, workflow_constructor = await _start_workflow_with_settings(
        settings, tmp_path
    )
    try:
        session_store = workflow_constructor.call_args.kwargs["session_store"]
        assert isinstance(session_store, SQLiteMessagingStore)
        for index in range(3):
            await session_store.record_message_id(
                "telegram", "chat", f"msg-{index}", "out", "note"
            )
        assert await session_store.get_tracked_message_ids_for_chat(
            "telegram", "chat"
        ) == ["msg-1", "msg-2"]
    finally:
        await _close(service, database)


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
        nvidia_nim_api_keys="test-nim-key",
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
