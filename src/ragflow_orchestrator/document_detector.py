"""Universal document type detector with comprehensive format support.

Supports detection of:
- Office documents: docx, xlsx, pptx, odt
- Text formats: txt, md, csv, json, xml, html
- Code: py, js, ts, java, go, c++, rust, etc.
- PDF: with text_layer vs scan classification
- Others: yaml, toml, etc.

Detection strategy (ordered by reliability):
1. Magic bytes (binary signature)
2. File extension
3. MIME type (via python-magic if available)
4. Content heuristics
5. Text pattern analysis
"""

from __future__ import annotations

import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET


class DocumentType(str, Enum):
    """All supported document types."""
    
    # Office documents
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    ODT = "odt"
    ODS = "ods"
    
    # PDF variants
    PDF_TEXT = "pdf_text"      # PDF with text layer
    PDF_SCAN = "pdf_scan"      # PDF from scanned images
    
    # Text formats
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    XML = "xml"
    JSON = "json"
    CSV = "csv"
    
    # Code
    CODE = "code"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CSHARP = "csharp"
    CPP = "cpp"
    C = "c"
    
    # Configuration
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    PROPERTIES = "properties"
    
    # Unsupported
    UNSUPPORTED = "unsupported"


@dataclass(slots=True)
class DocumentDetection:
    """Result of document type detection."""
    document_type: DocumentType
    mime_type: str = ""
    source: str = "unknown"  # "magic_bytes", "extension", "mime", "content", "heuristic"
    confidence: float = 0.5  # 0.0-1.0, 1.0 = certain
    pdf_subtype: str = ""    # "text_layer" or "ocr" for PDFs


# ============================================================================
# Magic bytes (binary signatures) for reliable file type detection
# ============================================================================

_MAGIC_BYTES: dict[bytes, tuple[DocumentType | str, str]] = {
    # Office documents (ZIP-based)
    b"PK\x03\x04": ("ZIP", ""),  # Generic ZIP, need further inspection
    
    # PDF
    b"%PDF": (DocumentType.PDF_TEXT, "pdf"),  # Will be refined to text vs scan
    
    # Microsoft Office
    b"PK\x03\x04\x14\x00\x06\x00": ("MSOFFICE", ""),  # Modern Office (need magic bytes)
    
    # Old Office (OLE)
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1": ("OLDOFFICE", ""),
    
    # OpenDocument
    b"PK\x03\x04\x14\x00\x00\x00": ("OPENDOC", ""),
    
    # RTF
    b"{\\rtf": (DocumentType.TXT, "rtf"),
    
    # GIF
    b"GIF87a": ("IMAGE", ""),
    b"GIF89a": ("IMAGE", ""),
    
    # JPEG
    b"\xff\xd8\xff": ("IMAGE", ""),
    
    # PNG
    b"\x89PNG\r\n\x1a\n": ("IMAGE", ""),
    
    # GZIP
    b"\x1f\x8b": ("GZIP", ""),
    
    # BZIP2
    b"BZ": ("BZIP2", ""),
}


# ============================================================================
# File extensions mapping
# ============================================================================

_EXTENSION_MAP: dict[str, DocumentType] = {
    # Office
    ".docx": DocumentType.DOCX,
    ".xlsx": DocumentType.XLSX,
    ".pptx": DocumentType.PPTX,
    ".odt": DocumentType.ODT,
    ".ods": DocumentType.ODS,
    ".doc": DocumentType.DOCX,  # Legacy
    ".xls": DocumentType.XLSX,  # Legacy
    
    # PDF
    ".pdf": DocumentType.PDF_TEXT,  # Will be refined
    
    # Text
    ".txt": DocumentType.TXT,
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".mdown": DocumentType.MARKDOWN,
    ".html": DocumentType.HTML,
    ".htm": DocumentType.HTML,
    ".xml": DocumentType.XML,
    ".json": DocumentType.JSON,
    ".jsonl": DocumentType.JSON,
    ".csv": DocumentType.CSV,
    ".tsv": DocumentType.CSV,
    
    # Code
    ".py": DocumentType.PYTHON,
    ".pyw": DocumentType.PYTHON,
    ".js": DocumentType.JAVASCRIPT,
    ".mjs": DocumentType.JAVASCRIPT,
    ".ts": DocumentType.TYPESCRIPT,
    ".tsx": DocumentType.TYPESCRIPT,
    ".java": DocumentType.JAVA,
    ".class": DocumentType.JAVA,
    ".go": DocumentType.GO,
    ".rs": DocumentType.RUST,
    ".cs": DocumentType.CSHARP,
    ".cpp": DocumentType.CPP,
    ".cc": DocumentType.CPP,
    ".cxx": DocumentType.CPP,
    ".c": DocumentType.C,
    ".h": DocumentType.C,
    ".hpp": DocumentType.CPP,
    
    # Configuration
    ".yaml": DocumentType.YAML,
    ".yml": DocumentType.YAML,
    ".toml": DocumentType.TOML,
    ".ini": DocumentType.INI,
    ".cfg": DocumentType.INI,
    ".conf": DocumentType.INI,
    ".properties": DocumentType.PROPERTIES,
}


# ============================================================================
# MIME types mapping
# ============================================================================

_MIME_TYPE_MAP: dict[str, DocumentType] = {
    # Office
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentType.DOCX,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": DocumentType.XLSX,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": DocumentType.PPTX,
    "application/vnd.oasis.opendocument.text": DocumentType.ODT,
    "application/vnd.oasis.opendocument.spreadsheet": DocumentType.ODS,
    "application/msword": DocumentType.DOCX,
    "application/vnd.ms-excel": DocumentType.XLSX,
    
    # PDF
    "application/pdf": DocumentType.PDF_TEXT,
    
    # Text
    "text/plain": DocumentType.TXT,
    "text/markdown": DocumentType.MARKDOWN,
    "text/html": DocumentType.HTML,
    "application/xml": DocumentType.XML,
    "text/xml": DocumentType.XML,
    "application/json": DocumentType.JSON,
    "text/csv": DocumentType.CSV,
    
    # Code
    "text/x-python": DocumentType.PYTHON,
    "text/javascript": DocumentType.JAVASCRIPT,
    "text/typescript": DocumentType.TYPESCRIPT,
    "text/x-java": DocumentType.JAVA,
    "text/x-golang": DocumentType.GO,
    "text/x-rustsrc": DocumentType.RUST,
    "text/x-csharp": DocumentType.CSHARP,
    
    # Configuration
    "application/x-yaml": DocumentType.YAML,
    "text/x-toml": DocumentType.TOML,
}


class UniversalDocumentDetector:
    """Comprehensive document type detector using multi-strategy approach."""
    
    def __init__(
        self,
        enable_ocr: bool = False,
        ocr_callable: Callable[[Any], bool] | None = None,
    ) -> None:
        """Initialize detector.
        
        Args:
            enable_ocr: Whether to detect PDF scans (requires PyMuPDF + PaddleOCR)
            ocr_callable: Custom OCR detection function(pdf_path) -> bool (True if scan)
        """
        self.enable_ocr = enable_ocr
        self.ocr_callable = ocr_callable
        self._pdf_classifier = None
    
    def detect(
        self,
        file_path: str | Path | None = None,
        file_data: bytes | None = None,
        text: str | None = None,
    ) -> DocumentDetection:
        """Detect document type using all available strategies.
        
        Args:
            file_path: Path to file (used for extension + PDF classification)
            file_data: Raw file bytes (used for magic bytes + MIME detection)
            text: Text content (used for content heuristics)
        
        Returns:
            DocumentDetection with type, source, and confidence
        """
        file_path = Path(file_path) if file_path else None
        
        # Strategy 1: Magic bytes (most reliable)
        if file_data:
            result = self._detect_by_magic_bytes(file_data, file_path)
            if result.document_type != DocumentType.UNSUPPORTED:
                return result
        
        # Strategy 2: Extension (fast and usually reliable)
        if file_path:
            result = self._detect_by_extension(file_path)
            if result.document_type != DocumentType.UNSUPPORTED:
                return result
        
        # Strategy 3: MIME type from file data
        if file_data:
            result = self._detect_by_mime_type(file_data)
            if result.document_type != DocumentType.UNSUPPORTED:
                return result
        
        # Strategy 4: Text content analysis
        if text:
            result = self._detect_by_content(text)
            if result.document_type != DocumentType.UNSUPPORTED:
                return result
        
        # Strategy 5: Fallback
        return DocumentDetection(
            document_type=DocumentType.UNSUPPORTED,
            confidence=0.0,
            source="fallback",
        )
    
    def detect_pdf_subtype(self, file_path: str | Path) -> DocumentDetection:
        """Determine if PDF is text-based or scanned.
        
        Uses PyMuPDF to analyze page content:
        - pdf_text: Has text layer (text extraction works)
        - pdf_scan: Scanned images (needs OCR)
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            DocumentDetection with pdf_text or pdf_scan
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() != ".pdf":
            return DocumentDetection(
                document_type=DocumentType.UNSUPPORTED,
                confidence=0.0,
                source="not_pdf",
            )
        
        # Try custom OCR callable first
        if self.ocr_callable:
            try:
                is_scan = self.ocr_callable(str(file_path))
                doc_type = DocumentType.PDF_SCAN if is_scan else DocumentType.PDF_TEXT
                return DocumentDetection(
                    document_type=doc_type,
                    confidence=0.9,
                    source="ocr_callable",
                    pdf_subtype="ocr" if is_scan else "text_layer",
                )
            except Exception:
                pass
        
        # Try PyMuPDF + PaddleOCR if enabled
        if self.enable_ocr:
            result = self._classify_pdf_with_pymupdf(file_path)
            if result.document_type != DocumentType.UNSUPPORTED:
                return result
        
        # Fallback: assume text layer (safer default)
        return DocumentDetection(
            document_type=DocumentType.PDF_TEXT,
            confidence=0.5,
            source="fallback",
            pdf_subtype="text_layer",
        )
    
    # ========================================================================
    # Private detection strategies
    # ========================================================================
    
    def _detect_by_magic_bytes(
        self,
        file_data: bytes,
        file_path: Path | None = None,
    ) -> DocumentDetection:
        """Detect type from binary magic bytes (most reliable)."""
        if not file_data:
            return DocumentDetection(DocumentType.UNSUPPORTED)
        
        # Check each magic byte signature
        for magic_bytes, (doc_type_name, mime_hint) in _MAGIC_BYTES.items():
            if file_data.startswith(magic_bytes):
                # Handle compound types that need refinement
                if doc_type_name == "ZIP":
                    return self._detect_zip_variant(file_data, file_path)
                elif doc_type_name == "MSOFFICE":
                    return self._detect_office_variant(file_data)
                elif doc_type_name == "OPENDOC":
                    return self._detect_opendoc_variant(file_data)
                elif isinstance(doc_type_name, DocumentType):
                    return DocumentDetection(
                        document_type=doc_type_name,
                        confidence=0.95,
                        source="magic_bytes",
                    )
        
        return DocumentDetection(DocumentType.UNSUPPORTED)
    
    def _detect_zip_variant(
        self,
        file_data: bytes,
        file_path: Path | None = None,
    ) -> DocumentDetection:
        """Detect Office/OpenDocument formats from ZIP structure."""
        # Quick heuristics without extracting whole file
        
        # Check for Office XML structure
        if b"word/" in file_data and b"document.xml" in file_data:
            return DocumentDetection(
                document_type=DocumentType.DOCX,
                confidence=0.95,
                source="magic_bytes",
            )
        if b"xl/" in file_data and b"workbook.xml" in file_data:
            return DocumentDetection(
                document_type=DocumentType.XLSX,
                confidence=0.95,
                source="magic_bytes",
            )
        if b"ppt/" in file_data and b"presentation.xml" in file_data:
            return DocumentDetection(
                document_type=DocumentType.PPTX,
                confidence=0.95,
                source="magic_bytes",
            )
        
        # Check for OpenDocument
        if b"mimetype" in file_data:
            if b"application/vnd.oasis.opendocument.text" in file_data:
                return DocumentDetection(
                    document_type=DocumentType.ODT,
                    confidence=0.95,
                    source="magic_bytes",
                )
            if b"application/vnd.oasis.opendocument.spreadsheet" in file_data:
                return DocumentDetection(
                    document_type=DocumentType.ODS,
                    confidence=0.95,
                    source="magic_bytes",
                )
        
        return DocumentDetection(DocumentType.UNSUPPORTED)
    
    def _detect_office_variant(self, file_data: bytes) -> DocumentDetection:
        """Detect OLE-based Office formats."""
        # OLE (Old Office) detection is complex, fallback to extension
        return DocumentDetection(DocumentType.UNSUPPORTED)
    
    def _detect_opendoc_variant(self, file_data: bytes) -> DocumentDetection:
        """Detect OpenDocument formats."""
        if b"application/vnd.oasis.opendocument.text" in file_data:
            return DocumentDetection(
                document_type=DocumentType.ODT,
                confidence=0.95,
                source="magic_bytes",
            )
        if b"application/vnd.oasis.opendocument.spreadsheet" in file_data:
            return DocumentDetection(
                document_type=DocumentType.ODS,
                confidence=0.95,
                source="magic_bytes",
            )
        return DocumentDetection(DocumentType.UNSUPPORTED)
    
    def _detect_by_extension(self, file_path: Path) -> DocumentDetection:
        """Detect type from file extension."""
        ext = file_path.suffix.lower()
        
        if ext in _EXTENSION_MAP:
            doc_type = _EXTENSION_MAP[ext]
            
            # For PDF, may need refinement
            if doc_type in (DocumentType.PDF_TEXT, DocumentType.PDF_SCAN):
                return DocumentDetection(
                    document_type=doc_type,
                    confidence=0.8,
                    source="extension",
                )
            
            return DocumentDetection(
                document_type=doc_type,
                confidence=0.85,
                source="extension",
            )
        
        return DocumentDetection(DocumentType.UNSUPPORTED)
    
    def _detect_by_mime_type(self, file_data: bytes) -> DocumentDetection:
        """Detect type using MIME type from python-magic."""
        try:
            import magic  # type: ignore[import-not-found]
        except ImportError:
            return DocumentDetection(DocumentType.UNSUPPORTED)
        
        try:
            mime = str(magic.from_buffer(file_data, mime=True)).lower().strip()
        except Exception:
            return DocumentDetection(DocumentType.UNSUPPORTED)
        
        if mime in _MIME_TYPE_MAP:
            return DocumentDetection(
                document_type=_MIME_TYPE_MAP[mime],
                mime_type=mime,
                confidence=0.90,
                source="mime",
            )
        
        return DocumentDetection(DocumentType.UNSUPPORTED)
    
    def _detect_by_content(self, text: str) -> DocumentDetection:
        """Detect type from text content patterns."""
        if not text:
            return DocumentDetection(DocumentType.UNSUPPORTED)
        
        sample = text[:500].strip()
        
        # XML
        if sample.startswith("<?xml"):
            return DocumentDetection(
                document_type=DocumentType.XML,
                confidence=0.95,
                source="content",
            )
        
        # HTML
        if sample.startswith("<!DOCTYPE") or sample.startswith("<html"):
            return DocumentDetection(
                document_type=DocumentType.HTML,
                confidence=0.95,
                source="content",
            )
        
        # JSON
        if sample.startswith(("{", "[")):
            try:
                json.loads(sample)
                return DocumentDetection(
                    document_type=DocumentType.JSON,
                    confidence=0.95,
                    source="content",
                )
            except (json.JSONDecodeError, ValueError):
                pass
        
        # YAML (basic detection)
        if ":" in sample and not sample.startswith("{"):
            lines = sample.split("\n")
            yaml_like = sum(1 for line in lines if ":" in line) > len(lines) * 0.5
            if yaml_like:
                return DocumentDetection(
                    document_type=DocumentType.YAML,
                    confidence=0.7,
                    source="content",
                )
        
        # TOML
        if "[" in sample and "=" in sample:
            try:
                if self._looks_like_toml(sample):
                    return DocumentDetection(
                        document_type=DocumentType.TOML,
                        confidence=0.8,
                        source="content",
                    )
            except Exception:
                pass
        
        # Markdown
        markdown_markers = ["# ", "## ", "### ", "```", "- ", "* ", "|"]
        if any(marker in sample for marker in markdown_markers):
            return DocumentDetection(
                document_type=DocumentType.MARKDOWN,
                confidence=0.8,
                source="content",
            )
        
        # CSV (simple heuristic)
        if self._looks_like_csv(sample):
            return DocumentDetection(
                document_type=DocumentType.CSV,
                confidence=0.7,
                source="content",
            )
        
        # Code detection (basic)
        code_keywords = [
            "def ", "class ", "import ", "function ", "const ", "let ",
            "var ", "async ", "await ", "public ", "private ",
        ]
        if any(kw in sample for kw in code_keywords):
            return DocumentDetection(
                document_type=DocumentType.CODE,
                confidence=0.75,
                source="content",
            )
        
        # Default: plain text
        return DocumentDetection(
            document_type=DocumentType.TXT,
            confidence=0.5,
            source="content",
        )
    
    def _classify_pdf_with_pymupdf(self, file_path: Path) -> DocumentDetection:
        """Classify PDF as text-based or scanned using PyMuPDF."""
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError:
            return DocumentDetection(DocumentType.UNSUPPORTED)
        
        try:
            doc = fitz.open(str(file_path))
            
            # Sample first 5 pages
            text_chars = 0
            has_large_images = False
            
            for page_num in range(min(5, len(doc))):
                page = doc[page_num]
                text = page.get_text("text").strip()
                text_chars += len(text)
                
                # Check for large images (typical in scans)
                for img in page.get_images():
                    try:
                        xref = img[0]
                        image = doc.extract_image(xref)
                        if image:
                            width, height = image["width"], image["height"]
                            if width * height > 200_000:  # Large image
                                has_large_images = True
                                break
                    except Exception:
                        pass
            
            doc.close()
            
            # Decision logic
            avg_chars_per_page = text_chars / min(5, len(doc)) if len(doc) > 0 else 0
            
            if avg_chars_per_page > 100 and not has_large_images:
                # Has text layer
                return DocumentDetection(
                    document_type=DocumentType.PDF_TEXT,
                    confidence=0.9,
                    source="pymupdf",
                    pdf_subtype="text_layer",
                )
            else:
                # Likely scanned
                return DocumentDetection(
                    document_type=DocumentType.PDF_SCAN,
                    confidence=0.8,
                    source="pymupdf",
                    pdf_subtype="ocr",
                )
        
        except Exception:
            return DocumentDetection(DocumentType.UNSUPPORTED)
    
    @staticmethod
    def _looks_like_csv(text: str) -> bool:
        """Heuristic CSV detection."""
        try:
            lines = text.split("\n")[:3]  # First 3 lines
            reader = csv.reader(lines)
            for row in reader:
                if len(row) >= 2:  # At least 2 columns
                    return True
        except Exception:
            pass
        return False
    
    @staticmethod
    def _looks_like_toml(text: str) -> bool:
        """Heuristic TOML detection."""
        lines = text.split("\n")
        has_section = any(line.strip().startswith("[") for line in lines)
        has_key_value = any("=" in line and not line.strip().startswith("#") for line in lines)
        return has_section and has_key_value


# ============================================================================
# Convenience functions
# ============================================================================

def detect_document_type(
    file_path: str | Path | None = None,
    file_data: bytes | None = None,
    text: str | None = None,
) -> DocumentDetection:
    """Quick detection using default detector.
    
    Args:
        file_path: Path to file
        file_data: Raw file bytes
        text: Text content
    
    Returns:
        DocumentDetection with type, source, and confidence
    """
    detector = UniversalDocumentDetector()
    return detector.detect(file_path=file_path, file_data=file_data, text=text)


def detect_pdf_type(file_path: str | Path) -> DocumentDetection:
    """Detect if PDF has text layer or is scanned.
    
    Args:
        file_path: Path to PDF file
    
    Returns:
        DocumentDetection with pdf_text or pdf_scan
    """
    detector = UniversalDocumentDetector(enable_ocr=True)
    return detector.detect_pdf_subtype(file_path)


def batch_detect_documents(
    file_paths: list[str | Path],
    max_workers: int | None = None,
) -> dict[str, DocumentDetection]:
    """Detect types for multiple files efficiently.
    
    Args:
        file_paths: List of file paths
        max_workers: Maximum number of worker threads (auto if None)
    
    Returns:
        Mapping of file paths to DocumentDetection results
    """
    detector = UniversalDocumentDetector(enable_ocr=False)

    normalized_paths = [Path(p) for p in file_paths]
    if not normalized_paths:
        return {}

    def _process_single(path: Path) -> tuple[str, DocumentDetection]:
        # Read file data for magic bytes check
        try:
            with open(path, "rb") as f:
                file_data = f.read(8192)  # First 8KB
        except Exception:
            file_data = None

        # Read text if small file
        text = None
        if path.is_file():
            try:
                size = path.stat().st_size
                if size < 100_000:  # Less than 100KB
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read(1000)  # First 1000 chars
            except Exception:
                pass

        result = detector.detect(file_path=path, file_data=file_data, text=text)
        return str(path), result

    # Auto-tune for I/O-bound workload.
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) * 4)
    max_workers = max(1, min(max_workers, len(normalized_paths)))

    results: dict[str, DocumentDetection] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key, detection in executor.map(_process_single, normalized_paths):
            results[key] = detection

    return results
