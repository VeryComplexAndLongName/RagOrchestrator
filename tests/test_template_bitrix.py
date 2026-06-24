from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from ragflow_orchestrator.embedding import HashEmbedder
from ragflow_orchestrator.factory import create_provider
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.presets import document_preset
from ragflow_orchestrator.templates.bitrix import BitrixTemplate
from ragflow_orchestrator.templates.models import BitrixConfig, LanguageMode


def _build_orchestrator(tmp_path: Path) -> RAGOrchestrator:
    provider = create_provider("postgres+qdrant", dsn="postgresql://rag_user:rag_password@localhost:5432/rag_db", qdrant_url="http://localhost:6333", qdrant_collection="bitrix_chunks")
    preset = document_preset()
    return RAGOrchestrator(
        provider=provider,
        embedder=HashEmbedder(dimensions=64),
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )


def test_bitrix_config_normalizes_and_validates() -> None:
    config = BitrixConfig(
        domain="https://portal.bitrix24.ru/",
        user_id=42,
        token="  tok  ",
        include_contacts=False,
        include_companies=False,
        include_deals=False,
        include_leads=False,
        include_tasks=False,
        include_activities=False,
        include_im_dialogs=True,
        dialog_ids=["  chat-a  ", "", "chat-a", " chat-b "],
    )

    assert config.domain == "portal.bitrix24.ru"
    assert config.dialog_ids == ["chat-a", "chat-b"]
    assert config.base_url == "https://portal.bitrix24.ru/rest/42/tok"

    with pytest.raises(ValueError, match="domain must contain host only"):
        BitrixConfig(
            domain="https://portal.bitrix24.ru/path",
            user_id=1,
            token="tok",
            include_contacts=False,
            include_companies=False,
            include_deals=False,
            include_leads=False,
            include_tasks=False,
            include_activities=False,
        )

    with pytest.raises(ValueError, match="dialog_ids must be provided"):
        BitrixConfig(
            domain="portal.bitrix24.ru",
            user_id=1,
            token="tok",
            include_contacts=False,
            include_companies=False,
            include_deals=False,
            include_leads=False,
            include_tasks=False,
            include_activities=False,
            include_im_dialogs=True,
            dialog_ids=[],
        )


def test_bitrix_template_ingests_contact_with_stub_payload(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    template = BitrixTemplate(orchestrator)

    def fake_request_json(config: BitrixConfig, method: str, params: dict[str, object] | None) -> dict[str, object]:
        del config
        start_raw = (params or {}).get("start", 0)
        start = int(start_raw) if isinstance(start_raw, (int, str)) else 0
        if method == "crm.contact.list" and start == 0:
            return {
                "result": [
                    {
                        "ID": "100",
                        "NAME": "Ivan",
                        "SECOND_NAME": "",
                        "LAST_NAME": "Petrov",
                        "PHONE": [{"VALUE": "+79990001122"}],
                        "EMAIL": [{"VALUE": "ivan@example.com"}],
                        "COMMENTS": "VIP client",
                    }
                ]
            }
        return {"result": []}

    template._request_json = fake_request_json  # type: ignore[method-assign]

    report = template.run(
        BitrixConfig(
            domain="portal.bitrix24.ru",
            user_id=7,
            token="tok",
            include_contacts=True,
            include_companies=False,
            include_deals=False,
            include_leads=False,
            include_tasks=False,
            include_activities=False,
            include_im_dialogs=False,
            language_mode=LanguageMode.AUTO,
        )
    )

    assert len(report.ingested) == 1
    assert report.ingested[0].source_id == "bitrix:contact:100"
    assert not report.failed


def test_bitrix_template_ingests_task_with_result_tasks_key(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    template = BitrixTemplate(orchestrator)

    def fake_request_json(config: BitrixConfig, method: str, params: dict[str, object] | None) -> dict[str, object]:
        del config
        start_raw = (params or {}).get("start", 0)
        start = int(start_raw) if isinstance(start_raw, (int, str)) else 0
        if method == "tasks.task.list" and start == 0:
            return {
                "result": {
                    "tasks": [
                        {
                            "ID": "501",
                            "task": {
                                "ID": "501",
                                "TITLE": "Prepare report",
                                "DESCRIPTION": "Collect CRM stats",
                                "RESPONSIBLE_ID": "21",
                                "CREATED_BY": "11",
                                "GROUP_ID": "3",
                                "STATUS": "2",
                            },
                        }
                    ]
                }
            }
        return {"result": {"tasks": []}}

    template._request_json = fake_request_json  # type: ignore[method-assign]

    report = template.run(
        BitrixConfig(
            domain="portal.bitrix24.ru",
            user_id=7,
            token="tok",
            include_contacts=False,
            include_companies=False,
            include_deals=False,
            include_leads=False,
            include_tasks=True,
            include_activities=False,
            include_im_dialogs=False,
            language_mode=LanguageMode.AUTO,
        )
    )

    assert len(report.ingested) == 1
    assert report.ingested[0].source_id == "bitrix:task:501"
    assert not report.failed


def test_bitrix_template_marks_im_skipped_when_dialogs_missing_runtime(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    template = BitrixTemplate(orchestrator)

    # model_construct bypasses validators so we can assert runtime defensive behavior in template.run.
    config = BitrixConfig.model_construct(
        domain="portal.bitrix24.ru",
        user_id=1,
        token=SecretStr("tok"),
        language_mode=LanguageMode.AUTO,
        include_contacts=False,
        include_companies=False,
        include_deals=False,
        include_leads=False,
        include_tasks=False,
        include_activities=False,
        include_im_dialogs=True,
        max_contacts=1000,
        max_companies=1000,
        max_deals=1000,
        max_leads=1000,
        max_tasks=1000,
        max_activities=1000,
        max_dialog_messages=200,
        dialog_ids=[],
    )

    report = template.run(config)

    assert report.skipped
    assert report.skipped[0].source == "im:<no-dialog-ids>"


def test_bitrix_im_source_url_is_fully_escaped() -> None:
    config = BitrixConfig(
        domain="portal.bitrix24.ru",
        user_id=9,
        token="tok",
        include_contacts=False,
        include_companies=False,
        include_deals=False,
        include_leads=False,
        include_tasks=False,
        include_activities=False,
    )

    source_url = BitrixTemplate._bitrix_source_url(config, "im", "chat/ops?x=1")
    assert source_url == "https://portal.bitrix24.ru/online/?IM_DIALOG=chat%2Fops%3Fx%3D1"


