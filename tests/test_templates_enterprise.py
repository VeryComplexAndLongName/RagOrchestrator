from __future__ import annotations

import json
from pathlib import Path

from ragflow_orchestrator.embedding import HashEmbedder
from ragflow_orchestrator.factory import create_provider
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.presets import document_preset
from ragflow_orchestrator.templates import (
    APIReferenceConfig,
    APIReferenceTemplate,
    ConfluenceWikiConfig,
    ConfluenceWikiTemplate,
    JiraConfig,
    JiraTemplate,
    LanguageMode,
)


def _build_orchestrator(tmp_path: Path) -> RAGOrchestrator:
    provider = create_provider("sqlite+vec", db_path=str(tmp_path / "enterprise.db"), table_name="enterprise_chunks")
    preset = document_preset()
    return RAGOrchestrator(
        provider=provider,
        embedder=HashEmbedder(dimensions=64),
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )


def test_confluence_wiki_template_with_stub_data(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    template = ConfluenceWikiTemplate(orchestrator)

    template._list_space_page_ids = lambda cfg, space: ["1001"]  # type: ignore[method-assign]
    template._get_page = lambda cfg, page_id: {  # type: ignore[method-assign]
        "id": page_id,
        "title": "Architecture",
        "body": {"storage": {"value": "<h1>System</h1><p>Design notes</p>"}},
    }

    report = template.run(
        ConfluenceWikiConfig(
            base_url="https://confluence.example.com",
            space_keys=["ENG"],
            max_pages=10,
            language_mode=LanguageMode.AUTO,
        )
    )

    assert len(report.ingested) == 1
    assert not report.failed


def test_jira_template_with_stub_search(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    template = JiraTemplate(orchestrator)

    template._search_issues = lambda cfg: [  # type: ignore[method-assign]
        {
            "key": "ENG-1",
            "fields": {
                "summary": "Login issue",
                "description": {"text": "Users cannot login"},
                "comment": {"comments": [{"body": {"text": "Investigating"}}]},
                "project": {"key": "ENG"},
                "issuetype": {"name": "Bug"},
                "updated": "2026-05-27",
            },
        }
    ]

    report = template.run(
        JiraConfig(
            base_url="https://jira.example.com",
            max_issues=20,
            language_mode=LanguageMode.AUTO,
        )
    )

    assert len(report.ingested) == 1
    assert not report.failed


def test_api_reference_template_from_local_openapi(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    template = APIReferenceTemplate(orchestrator)

    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Demo API", "version": "1.0.0", "description": "Reference"},
        "paths": {
            "/users": {
                "get": {"summary": "List users", "description": "Returns users", "operationId": "listUsers"}
            }
        },
        "components": {"schemas": {"User": {"type": "object", "properties": {"id": {"type": "string"}}}}},
    }
    spec_path = tmp_path / "openapi.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=True), encoding="utf-8")

    report = template.run(
        APIReferenceConfig(
            sources=[str(spec_path)],
            include_operations=True,
            include_schemas=True,
            language_mode=LanguageMode.AUTO,
        )
    )

    assert len(report.ingested) >= 2
    assert not report.failed


def test_api_reference_template_handles_non_dict_openapi_fields(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    template = APIReferenceTemplate(orchestrator)

    # Some upstream APIs can return malformed OpenAPI-like payloads where nested fields are lists.
    spec = {
        "openapi": "3.0.0",
        "info": [],
        "paths": [],
        "components": [],
    }
    spec_path = tmp_path / "openapi_malformed.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=True), encoding="utf-8")

    report = template.run(
        APIReferenceConfig(
            sources=[str(spec_path)],
            include_operations=True,
            include_schemas=True,
            language_mode=LanguageMode.AUTO,
        )
    )

    assert len(report.ingested) == 1
    assert not report.failed


def test_api_reference_template_max_items_limits_array_payload(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    template = APIReferenceTemplate(orchestrator)

    payload = [{"id": idx, "name": f"item-{idx}"} for idx in range(1, 6)]
    payload_path = tmp_path / "api_array.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    report = template.run(
        APIReferenceConfig(
            sources=[str(payload_path)],
            include_operations=True,
            include_schemas=True,
            max_items=2,
            language_mode=LanguageMode.AUTO,
        )
    )

    assert len(report.ingested) == 3
    assert not report.failed


def test_api_reference_template_plain_text_for_non_openapi_json_object() -> None:
        payload = {
                "name": {"official": "French Republic"},
                "flags": {"png": "https://flagcdn.com/w320/fr.png"},
        }

        chunks = APIReferenceTemplate._build_chunks(payload, include_operations=True, include_schemas=True)

        assert len(chunks) == 1
        assert "name.official: French Republic" in chunks[0]
        assert "flags.png: https://flagcdn.com/w320/fr.png" in chunks[0]


def test_api_reference_template_loads_xml_as_plain_text(tmp_path: Path) -> None:
        xml_text = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<countries>
    <country code=\"FR\">
        <name>France</name>
        <flag>https://flagcdn.com/w320/fr.png</flag>
    </country>
</countries>
"""

        xml_path = tmp_path / "countries.xml"
        xml_path.write_text(xml_text, encoding="utf-8")

        loaded = APIReferenceTemplate._load_spec(str(xml_path))
        assert isinstance(loaded, str)
        assert "countries/country[1]@code: FR" in loaded
        assert "countries/country[1]/name[1]: France" in loaded
