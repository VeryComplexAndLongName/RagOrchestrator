from __future__ import annotations

from ragflow_orchestrator.document_pipeline import (
    AdaptiveDocumentChunker,
    DocumentType,
    detect_document_subtype,
    detect_document_type,
)


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


def test_detect_document_subtype_agreement() -> None:
    prediction = detect_document_subtype(
        text=(
            "Договор оказания услуг. Стороны согласовали предмет договора, срок действия "
            "и порядок подписания."
        ),
        title="Соглашение",
        document_type=DocumentType.DOCX,
    )

    assert prediction.subtype == "agreement"
    assert prediction.confidence > 0


def test_adaptive_document_chunker_sets_subtype_metadata() -> None:
    chunker = AdaptiveDocumentChunker()
    chunks = chunker.chunk(
        source_id="agreement.txt",
        text="Договор между сторонами. Предмет договора и срок действия.",
        metadata={"document_type": "txt", "title": "Agreement"},
    )

    assert chunks
    assert chunks[0].metadata.get("document_subtype")
    assert "document_subtype_confidence" in chunks[0].metadata
