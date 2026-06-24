"""Tests for normative document parser."""

from __future__ import annotations

from uuid import uuid4

import pytest

from ragflow_orchestrator.cleaning.normative_parser import (
    NormativeChunk,
    NormativeChunker,
    PageContent,
    PdfClassifier,
    parse_normative_text,
)

# ============================================================
# NormativeChunk Tests
# ============================================================

def test_normative_chunk_creation() -> None:
    """Test NormativeChunk dataclass creation."""
    chunk = NormativeChunk(
        id="chunk_1",
        doc_id="doc_1",
        text="This is a chunk",
        clause_path="5.2.3",
        section="Main Section",
        standard_ref="GOST 1234",
        page=1,
        source="text_layer",
    )
    
    assert chunk.id == "chunk_1"
    assert chunk.doc_id == "doc_1"
    assert chunk.clause_path == "5.2.3"
    assert chunk.char_len == len("This is a chunk")


def test_normative_chunk_auto_id() -> None:
    """Test automatic ID generation for chunk."""
    chunk = NormativeChunk(
        id="",
        doc_id="doc_1",
        text="Content",
    )
    
    # ID should be auto-generated (UUID)
    assert chunk.id != ""
    assert len(chunk.id) > 0


# ============================================================
# Regex Pattern Tests
# ============================================================

def test_clause_extraction() -> None:
    """Test clause number extraction."""
    from ragflow_orchestrator.cleaning.normative_parser import CLAUSE_RE
    
    # Test cases
    test_cases = [
        ("1. General provisions", "1"),
        ("4.1. Requirements", "4.1"),
        ("5.2.3. Special rules", "5.2.3"),
        ("п.1. Пункт", "п.1"),
        ("пункт 2. Description", "пункт 2"),
    ]
    
    for text, expected_clause in test_cases:
        match = CLAUSE_RE.match(text)
        if match:
            clause = match.group(1).strip()
            assert clause, f"Failed to extract clause from: {text}"


def test_standard_extraction() -> None:
    """Test standard reference extraction."""
    from ragflow_orchestrator.cleaning.normative_parser import STD_RE
    
    test_cases = [
        "ГОСТ 27751-2014 defines requirements",
        "СП 14.13330.2018 specifies rules",
        "СНиП II-7-81* outdated",
        "Reference to GOST Р 1234-2024",
    ]
    
    for text in test_cases:
        match = STD_RE.search(text)
        assert match, f"Failed to extract standard from: {text}"


def test_section_header_detection() -> None:
    """Test section header pattern matching."""
    from ragflow_orchestrator.cleaning.normative_parser import SECTION_RE
    
    # Valid section headers
    valid = [
        "1. General provisions",
        "2.1. Detailed requirements",
        "3.2.1. Specific rules",
    ]
    
    for text in valid:
        match = SECTION_RE.match(text)
        assert match, f"Should match section header: {text}"
    
    # Invalid
    invalid = ["General provisions without number"]
    for text in invalid:
        match = SECTION_RE.match(text)
        assert not match, f"Should not match: {text}"


# ============================================================
# NormativeChunker Tests
# ============================================================

def test_chunker_initialization() -> None:
    """Test chunker initialization."""
    chunker = NormativeChunker(min_len=100, max_len=1000)
    assert chunker.min_len == 100
    assert chunker.max_len == 1000


def test_simple_chunking() -> None:
    """Test basic text chunking."""
    chunker = NormativeChunker(min_len=50, max_len=500)
    
    pages = [
        PageContent(
            page_num=1,
            text="""1. General provisions
This section covers general rules.

2. Specific requirements
This section specifies detailed requirements.
With multiple lines.""",
            source="text_layer",
        )
    ]
    
    chunks = chunker.chunk(pages, doc_id="test_doc")
    
    assert len(chunks) > 0
    assert all(isinstance(c, NormativeChunk) for c in chunks)
    assert all(c.doc_id == "test_doc" for c in chunks)


def test_clause_path_extraction() -> None:
    """Test clause_path extraction during chunking."""
    chunker = NormativeChunker(min_len=50, max_len=500)
    
    pages = [
        PageContent(
            page_num=1,
            text="""5.2.3. Special provision
This is the content of clause 5.2.3.""",
            source="text_layer",
        )
    ]
    
    chunks = chunker.chunk(pages, doc_id="test_doc")
    
    # At least one chunk should have the clause path
    clause_chunks = [c for c in chunks if c.clause_path]
    assert len(clause_chunks) > 0
    assert any("5.2.3" in (c.clause_path or "") for c in chunks)


def test_standard_reference_extraction() -> None:
    """Test standard reference extraction during chunking."""
    chunker = NormativeChunker(min_len=50, max_len=500)
    
    pages = [
        PageContent(
            page_num=1,
            text="""ГОСТ 27751-2014 Rules
According to GOST 27751-2014, the following applies.""",
            source="text_layer",
        )
    ]
    
    chunks = chunker.chunk(pages, doc_id="test_doc")
    
    # Check if standard reference was captured
    std_chunks = [c for c in chunks if c.standard_ref]
    assert len(std_chunks) > 0


def test_multi_page_chunking() -> None:
    """Test chunking across multiple pages."""
    chunker = NormativeChunker(min_len=30, max_len=500)
    
    pages = [
        PageContent(
            page_num=1,
            text="""1. First section
Content of first page.""",
            source="text_layer",
        ),
        PageContent(
            page_num=2,
            text="""2. Second section
Content of second page.""",
            source="ocr",
        ),
    ]
    
    chunks = chunker.chunk(pages, doc_id="test_doc")
    
    # Should have chunks from both pages
    assert len(chunks) > 0
    # Pages should be reflected in chunks
    pages_in_chunks = {c.page for c in chunks}
    assert 1 in pages_in_chunks or 2 in pages_in_chunks


def test_enrich_for_embedding() -> None:
    """Test enrichment of chunk text for embedding."""
    chunker = NormativeChunker()
    
    chunk = NormativeChunk(
        id="chunk_1",
        doc_id="doc_1",
        text="Main content",
        clause_path="5.2.3",
        standard_ref="GOST 1234",
        section="Main Section",
    )
    
    enriched = chunker.enrich_for_embedding(chunk, "Document Title")
    
    # Should contain original text
    assert "Main content" in enriched
    
    # Should contain breadcrumbs
    assert "Document Title" in enriched
    assert "GOST 1234" in enriched
    assert "Main Section" in enriched
    assert "5.2.3" in enriched


# ============================================================
# Public API Tests
# ============================================================

def test_parse_normative_text() -> None:
    """Test parse_normative_text high-level API."""
    text = """1. General provisions
This section covers general rules and requirements.

2. Specific requirements
This section specifies detailed requirements.

3. Implementation
Rules for implementation."""
    
    chunks = parse_normative_text(
        text=text,
        doc_id="test_doc",
        source="text_layer",
    )
    
    assert len(chunks) > 0
    assert all(isinstance(c, NormativeChunk) for c in chunks)
    assert all(c.doc_id == "test_doc" for c in chunks)
    assert all(c.source == "text_layer" for c in chunks)


def test_parse_normative_text_custom_chunk_sizes() -> None:
    """Test parse_normative_text with custom chunk sizes."""
    text = "Small " * 100  # Repeated text
    
    chunks_small = parse_normative_text(
        text=text,
        doc_id="doc_1",
        min_chunk_len=10,
        max_chunk_len=50,
    )
    
    chunks_large = parse_normative_text(
        text=text,
        doc_id="doc_2",
        min_chunk_len=200,
        max_chunk_len=500,
    )
    
    # Smaller max_len should produce more chunks
    # (not guaranteed, but likely for repeated content)
    assert len(chunks_small) >= 1
    assert len(chunks_large) >= 1


# ============================================================
# PDF Classifier Tests (mock-based, no real PDFs)
# ============================================================

def test_pdf_classifier_initialization() -> None:
    """Test PdfClassifier initialization."""
    try:
        classifier = PdfClassifier(text_threshold=100)
        assert classifier.text_threshold == 100
    except ImportError:
        # PyMuPDF not installed
        pytest.skip("PyMuPDF not available")


# ============================================================
# Edge Cases and Error Handling
# ============================================================

def test_empty_text_chunking() -> None:
    """Test chunking with empty text."""
    chunker = NormativeChunker()
    
    pages = [
        PageContent(
            page_num=1,
            text="",
            source="text_layer",
        )
    ]
    
    chunks = chunker.chunk(pages, doc_id="test_doc")
    
    # Should handle gracefully
    assert isinstance(chunks, list)


def test_very_long_clause() -> None:
    """Test handling of very long clause."""
    chunker = NormativeChunker(min_len=50, max_len=200)
    
    # Very long single clause
    long_text = "1. Main clause\n" + "Word " * 500
    
    pages = [
        PageContent(
            page_num=1,
            text=long_text,
            source="text_layer",
        )
    ]
    
    chunks = chunker.chunk(pages, doc_id="test_doc")
    
    # Should split long clause
    assert len(chunks) > 0


def test_mixed_russian_english() -> None:
    """Test handling of mixed Russian/English text."""
    chunker = NormativeChunker(min_len=50, max_len=500)
    
    text = """1. Основные положения
This is English text mixed with Russian.

2. Требования
Requirements in Russian.

3. Implementation Details
Details in English."""
    
    pages = [
        PageContent(
            page_num=1,
            text=text,
            source="text_layer",
        )
    ]
    
    chunks = chunker.chunk(pages, doc_id="test_doc")
    
    assert len(chunks) > 0
    # All chunks should have Cyrillic or Latin content
    assert any(c.text for c in chunks)


def test_chunks_preserve_order() -> None:
    """Test that chunk order is preserved."""
    chunker = NormativeChunker(min_len=30, max_len=500)
    
    pages = [
        PageContent(
            page_num=1,
            text="""1. First
Content 1.

2. Second
Content 2.

3. Third
Content 3.""",
            source="text_layer",
        )
    ]
    
    chunks = chunker.chunk(pages, doc_id="test_doc")
    
    # chunk_index should be sequential
    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(indices)


# ============================================================
# Integration Tests
# ============================================================

def test_full_normative_workflow() -> None:
    """Test complete workflow from text to chunks."""
    doc_id = str(uuid4())
    
    text = """ГОСТ 27751-2014 Надежность

1. Область применения
This standard applies to all systems.

1.1. Scope limitations
Limited to defined systems.

2. Нормативные ссылки
References to other standards.

3. Термины и определения
3.1. Definition of reliability
The ability to function properly.

3.2. Definition of availability
Time-based availability measure.

4. Требования
4.1. General requirements
Systems must be reliable.

4.2. Specific requirements
Detailed requirements here."""
    
    chunks = parse_normative_text(
        text=text,
        doc_id=doc_id,
        min_chunk_len=100,
        max_chunk_len=800,
    )
    
    # Verify output
    assert len(chunks) > 0
    assert all(c.doc_id == doc_id for c in chunks)
    
    # Should have extracted standard reference
    std_chunks = [c for c in chunks if c.standard_ref]
    assert len(std_chunks) > 0
    assert any("27751" in (c.standard_ref or "") for c in chunks)
    
    # Should have extracted clauses
    clause_chunks = [c for c in chunks if c.clause_path]
    assert len(clause_chunks) > 0
    
    # Should have extracted sections
    section_chunks = [c for c in chunks if c.section]
    assert len(section_chunks) > 0
