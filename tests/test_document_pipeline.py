from __future__ import annotations

from ragflow_orchestrator.document_pipeline import AdaptiveDocumentChunker, DocumentType, detect_document_type


def test_detect_document_type_json_xml_and_csv() -> None:
    assert detect_document_type(text='{"user": {"name": "John"}}').document_type == DocumentType.JSON
    assert detect_document_type(text="<root><item>value</item></root>").document_type == DocumentType.XML
    assert detect_document_type(text="name,age\nAlice,30\nBob,31").document_type == DocumentType.CSV


def test_adaptive_document_chunker_normalizes_json_paths() -> None:
    chunker = AdaptiveDocumentChunker()

    chunks = chunker.chunk(
        source_id="payload.json",
        text='{"user": {"name": "John", "address": {"city": "NY"}}}',
        metadata={"document_type": "json"},
    )

    assert chunks
    assert "user.name" in chunks[0].text
    assert "user.address.city" in chunks[0].text
