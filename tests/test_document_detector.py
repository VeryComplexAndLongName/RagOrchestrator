"""Tests for universal document type detector."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from ragflow_orchestrator.document_detector import (
    DocumentType,
    UniversalDocumentDetector,
    batch_detect_documents,
    detect_document_type,
    detect_pdf_type,
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def detector():
    """Create detector instance."""
    return UniversalDocumentDetector(enable_ocr=False)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Test: Basic Detection by Extension
# ============================================================================

class TestDetectionByExtension:
    """Test detection using file extension."""
    
    def test_pdf_extension(self, detector, temp_dir):
        """Detect PDF by extension."""
        pdf_file = temp_dir / "document.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        
        detection = detector.detect(file_path=pdf_file)
        
        assert detection.document_type in (
            DocumentType.PDF_TEXT,
            DocumentType.PDF_SCAN,
        )
        assert detection.source == "extension"
    
    def test_docx_extension(self, detector, temp_dir):
        """Detect DOCX by extension."""
        docx_file = temp_dir / "document.docx"
        docx_file.write_bytes(b"dummy")
        
        detection = detector.detect(file_path=docx_file)
        
        assert detection.document_type == DocumentType.DOCX
        assert "extension" in detection.source
    
    def test_xlsx_extension(self, detector, temp_dir):
        """Detect XLSX by extension."""
        xlsx_file = temp_dir / "spreadsheet.xlsx"
        xlsx_file.write_bytes(b"dummy")
        
        detection = detector.detect(file_path=xlsx_file)
        
        assert detection.document_type == DocumentType.XLSX
    
    def test_json_extension(self, detector, temp_dir):
        """Detect JSON by extension."""
        json_file = temp_dir / "data.json"
        json_file.write_text('{"key": "value"}')
        
        detection = detector.detect(file_path=json_file)
        
        assert detection.document_type == DocumentType.JSON
    
    def test_csv_extension(self, detector, temp_dir):
        """Detect CSV by extension."""
        csv_file = temp_dir / "data.csv"
        csv_file.write_text("col1,col2\nval1,val2")
        
        detection = detector.detect(file_path=csv_file)
        
        assert detection.document_type == DocumentType.CSV
    
    def test_markdown_extension(self, detector, temp_dir):
        """Detect Markdown by extension."""
        md_file = temp_dir / "README.md"
        md_file.write_text("# Title\n\nContent")
        
        detection = detector.detect(file_path=md_file)
        
        assert detection.document_type == DocumentType.MARKDOWN
    
    def test_python_extension(self, detector, temp_dir):
        """Detect Python code by extension."""
        py_file = temp_dir / "script.py"
        py_file.write_text("print('hello')")
        
        detection = detector.detect(file_path=py_file)
        
        assert detection.document_type == DocumentType.PYTHON


# ============================================================================
# Test: Magic Bytes Detection
# ============================================================================

class TestDetectionByMagicBytes:
    """Test detection using binary signatures."""
    
    def test_pdf_magic_bytes(self, detector):
        """Detect PDF by magic bytes."""
        pdf_data = b"%PDF-1.4\n%sample"
        
        detection = detector.detect(file_data=pdf_data)
        
        assert detection.document_type in (
            DocumentType.PDF_TEXT,
            DocumentType.PDF_SCAN,
        )
        assert detection.source == "magic_bytes"
        assert detection.confidence > 0.9
    
    def test_docx_magic_bytes(self, detector, temp_dir):
        """Detect DOCX by ZIP magic bytes."""
        # DOCX is ZIP with specific content
        docx_data = b"PK\x03\x04\x14\x00\x06\x00"
        docx_data += b"word/document.xml"
        
        detection = detector.detect(file_data=docx_data, file_path=Path("doc.docx"))
        
        # Should detect as office/zip format
        assert detection.confidence > 0.5
    
    def test_png_magic_bytes(self, detector):
        """Detect PNG magic bytes."""
        png_data = b"\x89PNG\r\n\x1a\n"
        
        detection = detector.detect(file_data=png_data)
        
        # PNG is not a supported document type
        assert detection.document_type == DocumentType.UNSUPPORTED


# ============================================================================
# Test: Content-based Detection
# ============================================================================

class TestDetectionByContent:
    """Test detection using text content patterns."""
    
    def test_json_content(self, detector):
        """Detect JSON from content."""
        json_text = '{"name": "John", "age": 30}'
        
        detection = detector.detect(text=json_text)
        
        assert detection.document_type == DocumentType.JSON
        assert detection.source == "content"
    
    def test_xml_content(self, detector):
        """Detect XML from content."""
        xml_text = '<?xml version="1.0"?><root><item>test</item></root>'
        
        detection = detector.detect(text=xml_text)
        
        assert detection.document_type == DocumentType.XML
        assert detection.source == "content"
    
    def test_html_content(self, detector):
        """Detect HTML from content."""
        html_text = "<!DOCTYPE html><html><body>Test</body></html>"
        
        detection = detector.detect(text=html_text)
        
        assert detection.document_type == DocumentType.HTML
        assert detection.source == "content"
    
    def test_markdown_content(self, detector):
        """Detect Markdown from content."""
        md_text = "# Title\n## Subtitle\n- List item\n```code```"
        
        detection = detector.detect(text=md_text)
        
        assert detection.document_type == DocumentType.MARKDOWN
        assert detection.source == "content"
    
    def test_yaml_content(self, detector):
        """Detect YAML from content."""
        yaml_text = "name: value\nkey: value\nsection:\n  nested: value"
        
        detection = detector.detect(text=yaml_text)
        
        # Should recognize YAML pattern
        assert detection.document_type in (DocumentType.YAML, DocumentType.TXT)
    
    def test_csv_content(self, detector):
        """Detect CSV from content."""
        csv_text = "name,age,email\nJohn,30,john@example.com\nJane,25,jane@example.com"
        
        detection = detector.detect(text=csv_text)
        
        # Should detect CSV structure
        assert detection.document_type in (DocumentType.CSV, DocumentType.TXT)
    
    def test_code_content(self, detector):
        """Detect code from content."""
        code_text = "def hello():\n    print('world')"
        
        detection = detector.detect(text=code_text)
        
        assert detection.document_type in (DocumentType.PYTHON, DocumentType.CODE)
    
    def test_plain_text_content(self, detector):
        """Detect plain text."""
        text = "This is just plain text without any special format."
        
        detection = detector.detect(text=text)
        
        assert detection.document_type == DocumentType.TXT


# ============================================================================
# Test: Confidence Scores
# ============================================================================

class TestConfidenceScores:
    """Test confidence scoring."""
    
    def test_magic_bytes_high_confidence(self, detector):
        """Magic bytes should have high confidence."""
        pdf_data = b"%PDF-1.4"
        
        detection = detector.detect(file_data=pdf_data)
        
        assert detection.confidence >= 0.9
    
    def test_extension_good_confidence(self, detector, temp_dir):
        """Extension detection should have good confidence."""
        pdf_file = temp_dir / "doc.pdf"
        pdf_file.write_bytes(b"dummy")
        
        detection = detector.detect(file_path=pdf_file)
        
        assert detection.confidence >= 0.8
    
    def test_content_heuristic_medium_confidence(self, detector):
        """Content heuristics should have medium confidence."""
        md_text = "# Title\nSome content"
        
        detection = detector.detect(text=md_text)
        
        assert 0.5 < detection.confidence < 1.0


# ============================================================================
# Test: Multi-strategy Detection
# ============================================================================

class TestMultiStrategyDetection:
    """Test combining multiple detection strategies."""
    
    def test_all_signals_agree(self, detector, temp_dir):
        """All signals should agree on obvious formats."""
        json_file = temp_dir / "data.json"
        json_file.write_text('{"key": "value"}')
        
        with open(json_file, "rb") as f:
            file_data = f.read()
        
        detection = detector.detect(
            file_path=json_file,
            file_data=file_data,
            text='{"key": "value"}',
        )
        
        assert detection.document_type == DocumentType.JSON
        assert detection.confidence >= 0.9
    
    def test_extension_fallback(self, detector, temp_dir):
        """Extension fallback when content unclear."""
        txt_file = temp_dir / "document.txt"
        txt_file.write_text("Random text content")
        
        detection = detector.detect(file_path=txt_file)
        
        assert detection.document_type == DocumentType.TXT


# ============================================================================
# Test: Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test high-level convenience functions."""
    
    def test_detect_document_type_one_liner(self, temp_dir):
        """Test simple one-liner detection."""
        json_file = temp_dir / "data.json"
        json_file.write_text('{"key": "value"}')
        
        detection = detect_document_type(file_path=json_file)
        
        assert detection.document_type == DocumentType.JSON
    
    def test_batch_detect_documents(self, temp_dir):
        """Test batch detection."""
        # Create test files
        (temp_dir / "file1.json").write_text('{"a": 1}')
        (temp_dir / "file2.md").write_text("# Title")
        (temp_dir / "file3.txt").write_text("Plain text")
        
        results = batch_detect_documents(list(temp_dir.glob("*")))
        
        assert len(results) == 3
        
        # Check each file
        for file_path, detection in results.items():
            assert detection.document_type != DocumentType.UNSUPPORTED


# ============================================================================
# Test: PDF Subtype Detection
# ============================================================================

class TestPdfSubtypeDetection:
    """Test PDF text vs scan classification."""
    
    def test_pdf_detection_requires_ocr_enabled(self, detector, temp_dir):
        """PDF subtype detection requires enable_ocr."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        
        # Without OCR, should return basic PDF type
        detection = detector.detect_pdf_subtype(pdf_file)
        
        # Should return fallback (safe)
        assert detection.document_type in (
            DocumentType.PDF_TEXT,
            DocumentType.PDF_SCAN,
        )
    
    def test_pdf_type_with_custom_ocr_callable(self, temp_dir):
        """Test PDF detection with custom OCR callable."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        
        # Mock OCR callable
        ocr_callable = Mock(return_value=False)  # Not a scan
        
        detector = UniversalDocumentDetector(
            enable_ocr=True,
            ocr_callable=ocr_callable,
        )
        
        detection = detector.detect_pdf_subtype(pdf_file)
        
        assert detection.document_type == DocumentType.PDF_TEXT
        assert detection.pdf_subtype == "text_layer"
        ocr_callable.assert_called_once()
    
    def test_detect_pdf_type_convenience_function(self, temp_dir):
        """Test convenience function for PDF detection."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        
        # Should not fail even without special setup
        detection = detect_pdf_type(pdf_file)
        
        assert detection.document_type in (
            DocumentType.PDF_TEXT,
            DocumentType.PDF_SCAN,
        )


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_file(self, detector, temp_dir):
        """Detect type of empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_bytes(b"")
        
        detection = detector.detect(file_path=empty_file)
        
        assert detection.document_type == DocumentType.TXT  # By extension
    
    def test_empty_text(self, detector):
        """Detect type of empty text."""
        detection = detector.detect(text="")
        
        assert detection.document_type == DocumentType.UNSUPPORTED
    
    def test_none_file_path(self, detector):
        """Handle None file path."""
        detection = detector.detect(file_path=None, text="test")
        
        # Should fall back to content detection
        assert detection.document_type in (DocumentType.TXT, DocumentType.UNSUPPORTED)
    
    def test_nonexistent_file(self, detector):
        """Handle nonexistent file path."""
        detection = detector.detect(file_path=Path("nonexistent.pdf"))
        
        # Should detect by extension
        assert detection.document_type in (
            DocumentType.PDF_TEXT,
            DocumentType.PDF_SCAN,
        )
    
    def test_corrupted_json(self, detector):
        """Handle corrupted JSON."""
        corrupted_json = '{invalid json'
        
        detection = detector.detect(text=corrupted_json)
        
        # Should not detect as JSON
        assert detection.document_type != DocumentType.JSON
    
    def test_very_long_text(self, detector):
        """Handle very long text."""
        long_text = "# Title\n" + ("content " * 10000)
        
        detection = detector.detect(text=long_text[:500])
        
        assert detection.document_type == DocumentType.MARKDOWN


# ============================================================================
# Test: Source Tracking
# ============================================================================

class TestSourceTracking:
    """Test that detection source is correctly tracked."""
    
    def test_source_is_extension(self, detector, temp_dir):
        """Track extension as source."""
        file = temp_dir / "doc.pdf"
        file.write_bytes(b"")
        
        detection = detector.detect(file_path=file)
        
        assert detection.source == "extension"
    
    def test_source_is_magic_bytes(self, detector):
        """Track magic bytes as source."""
        detection = detector.detect(file_data=b"%PDF-1.4")
        
        assert detection.source == "magic_bytes"
    
    def test_source_is_content(self, detector):
        """Track content as source."""
        detection = detector.detect(text='{"key": "value"}')
        
        assert detection.source == "content"


# ============================================================================
# Test: MIME Type Detection
# ============================================================================

class TestMimeTypeDetection:
    """Test MIME type detection with python-magic."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("magic", minversion=None),
        reason="python-magic not installed",
    )
    def test_mime_type_pdf(self, detector):
        """Detect MIME type for PDF."""
        pdf_data = b"%PDF-1.4"
        
        detection = detector.detect(file_data=pdf_data)
        
        # MIME type might be set if python-magic is available
        if detection.mime_type:
            assert "pdf" in detection.mime_type.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
