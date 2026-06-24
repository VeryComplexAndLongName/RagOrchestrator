"""Parser for normative documents (ПП РФ, ГОСТ, СП, СНиП, etc).

Implements:
1. PDF type detection (text layer vs scans)
2. OCR for scans (PaddleOCR or Tesseract)
3. Structural chunking preserving document hierarchy
4. Standard reference and clause extraction

Based on: MyTasks/additional.md
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable

# Optional dependencies for PDF and OCR
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import numpy as np
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None
    np = None


@dataclass
class PageContent:
    """Extracted content from a single PDF page."""
    page_num: int
    text: str
    source: str  # "text_layer" | "ocr"


@dataclass
class NormativeChunk:
    """A chunk representing a logical unit in a normative document."""
    id: str
    doc_id: str
    text: str
    chunk_index: int = 0
    clause_path: str | None = None  # "5.2.3" or "пункт 3.1.1"
    section: str | None = None  # Section title
    standard_ref: str | None = None  # "СП 14.13330.2018", "ГОСТ 27751-2014"
    page: int = 1
    source: str = "text_layer"  # "text_layer" | "ocr"
    char_len: int = 0
    semantic_type: str = "generic"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.char_len:
            self.char_len = len(self.text)


# ============================================================
# Regex patterns for structure extraction
# ============================================================

# Clause numbers: 1. / 4.1 / 5.2.3 / 6.7.12 / 13.1 / п.1 / пункт 2 / раздел 2.1
CLAUSE_RE = re.compile(r'^\s*((?:раздел\s+|п[ункт]?\s+)?(\d+(?:\.\d+){0,3}))[\.:]?\s+', re.IGNORECASE)

# Standard codes: ГОСТ/GOST 27751-2014, ГОСТ/GOST Р 1234-2024, СП 14.13330.2018, СНиП II-7-81*
STD_RE = re.compile(
    r'((?:ГОСТ|GOST)\s+(?:Р\s+)?[\d\-\.]+|СП\s+[\d\.]+|СНиП\s+[\dIVX\-\.\*]+|СНИП\s+[\dIVX\-\.\*]+)',
    re.IGNORECASE
)

# Section headers: "1. Общие положения", "2.1. Requirements", "3.2.1. Specific rules"
SECTION_RE = re.compile(r'^(\d+(?:\.\d+){0,2})\.\s+([A-Za-zА-ЯЁа-яё][A-Za-zА-ЯЁа-яё\s\-,\.]+)$', re.UNICODE)


# ============================================================
# PDF Classification and Extraction
# ============================================================

class PdfClassifier:
    """Determines if PDF pages are text or scans and extracts content."""

    def __init__(self, text_threshold: int = 100) -> None:
        """Initialize classifier.

        Args:
            text_threshold: Minimum character count to consider page as text layer.
        """
        if fitz is None:
            raise ImportError("PyMuPDF required: pip install PyMuPDF")
        self.text_threshold = text_threshold
        self._ocr_engine = None

    def classify_and_extract(
        self,
        pdf_path: str,
        ocr_fn: Callable[[Any], str] | None = None,
    ) -> list[PageContent]:
        """Classify each page as text/scan and extract content.

        Args:
            pdf_path: Path to PDF file
            ocr_fn: Optional OCR function; if None, uses PaddleOCR by default for scans

        Returns:
            List of PageContent (one per page)
        """
        if ocr_fn is None:
            ocr_fn = self._default_ocr

        doc = fitz.open(pdf_path)
        pages = []

        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            char_count = len(text)

            # Heuristic: check for large images (common in scans)
            has_large_image = any(
                self._get_image_size(page, img) > 200_000
                for img in page.get_images()
            )

            # Decide: text_layer or OCR
            if char_count < self.text_threshold or has_large_image:
                try:
                    ocr_text = ocr_fn(page)
                    pages.append(PageContent(i + 1, ocr_text, "ocr"))
                except Exception:
                    # OCR failed, fall back to raw text
                    pages.append(PageContent(i + 1, text, "text_layer"))
            else:
                pages.append(PageContent(i + 1, text, "text_layer"))

        doc.close()
        return pages

    def _default_ocr(self, page: Any) -> str:
        """Default OCR using PaddleOCR."""
        if PaddleOCR is None:
            raise ImportError("PaddleOCR required: pip install paddleocr")

        if self._ocr_engine is None:
            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="ru",
                show_log=False,
            )

        # Render page to image
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        # RGBA -> RGB if needed
        if pix.n == 4:
            img = img[:, :, :3]

        # Run OCR
        result = self._ocr_engine.ocr(img, cls=True)
        lines = []

        # Sort by vertical position for correct reading order
        for block in result or []:
            if not block:
                continue
            for line in sorted(block, key=lambda x: x[0][0][1]):
                lines.append(line[1][0])

        return "\n".join(lines)

    @staticmethod
    def _get_image_size(page: Any, img_ref: Any) -> int:
        """Get image size in pixels (width * height)."""
        try:
            bbox = page.get_image_bbox(img_ref)
            if bbox:
                return int((bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0))
        except Exception:
            pass
        return 0


# ============================================================
# Structural Chunking
# ============================================================

class NormativeChunker:
    """Chunks normative document text while preserving hierarchy."""

    def __init__(self, min_len: int = 120, max_len: int = 1200) -> None:
        """Initialize chunker.

        Args:
            min_len: Minimum chunk length to keep
            max_len: Maximum chunk length before splitting
        """
        self.min_len = min_len
        self.max_len = max_len

    def chunk(
        self,
        pages: list[PageContent],
        doc_id: str,
    ) -> list[NormativeChunk]:
        """Chunk normative document pages into logical units.

        Grouping strategy:
        - Group lines by clause/section boundaries
        - Keep short clauses together with parent
        - Split long clauses by sub-clauses
        - Preserve metadata (standard, section, clause path)

        Returns:
            List of NormativeChunk
        """
        chunks: list[NormativeChunk] = []
        current_std = None
        current_section = None
        buffer_lines: list[str] = []
        buffer_clause = None
        buffer_page = pages[0].page_num if pages else 1
        buffer_source = "text_layer"

        def flush() -> None:
            nonlocal buffer_lines, buffer_clause
            text = "\n".join(buffer_lines).strip()
            if text and len(text) >= self.min_len // 3:
                chunk = NormativeChunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=text,
                    chunk_index=len(chunks),
                    clause_path=buffer_clause or None,
                    section=current_section,
                    standard_ref=current_std,
                    page=buffer_page,
                    source=buffer_source,
                )
                chunks.append(chunk)
            buffer_lines = []

        for page in pages:
            for raw_line in page.text.split("\n"):
                line = raw_line.strip()
                if not line:
                    continue

                # Extract standard reference if line starts with it
                std_match = STD_RE.search(line)
                if std_match and len(line) < 100:
                    current_std = std_match.group(1).strip()

                # Detect section header
                sec_match = SECTION_RE.match(line)
                if sec_match:
                    current_section = line.strip()

                # Check for clause boundary
                clause_match = CLAUSE_RE.match(line)
                if clause_match:
                    flush()
                    buffer_clause = clause_match.group(1).strip()
                    buffer_page = page.page_num
                    buffer_source = page.source

                buffer_lines.append(line)

                # Flush if buffer gets too large
                if sum(len(x) for x in buffer_lines) > self.max_len:
                    flush()

        # Final flush
        flush()
        return chunks

    def enrich_for_embedding(
        self,
        chunk: NormativeChunk,
        doc_title: str,
    ) -> str:
        """Add contextual information to chunk text for better embeddings.

        Adds "breadcrumbs": document title, standard, section, clause path.
        """
        parts = [doc_title]
        if chunk.standard_ref:
            parts.append(chunk.standard_ref)
        if chunk.section:
            parts.append(chunk.section)
        if chunk.clause_path:
            parts.append(f"пункт {chunk.clause_path}")

        header = " | ".join(parts)
        return f"{header}\n{chunk.text}"


# ============================================================
# Public API
# ============================================================

def parse_normative_pdf(
    pdf_path: str,
    doc_id: str,
    doc_title: str | None = None,
    text_threshold: int = 100,
    min_chunk_len: int = 120,
    max_chunk_len: int = 1200,
) -> list[NormativeChunk]:
    """Parse normative document PDF.

    High-level function combining classification, extraction, and chunking.

    Args:
        pdf_path: Path to PDF file
        doc_id: Document UUID
        doc_title: Optional document title for embedding enrichment
        text_threshold: Character threshold to classify page as text
        min_chunk_len: Minimum chunk length
        max_chunk_len: Maximum chunk length

    Returns:
        List of NormativeChunk objects
    """
    classifier = PdfClassifier(text_threshold=text_threshold)
    pages = classifier.classify_and_extract(pdf_path)

    chunker = NormativeChunker(min_len=min_chunk_len, max_len=max_chunk_len)
    chunks = chunker.chunk(pages, doc_id=doc_id)

    return chunks


def parse_normative_text(
    text: str,
    doc_id: str,
    source: str = "text_layer",
    min_chunk_len: int = 120,
    max_chunk_len: int = 1200,
) -> list[NormativeChunk]:
    """Parse normative document from raw text.

    Useful for text extracted from other sources (Confluence, databases, etc).

    Args:
        text: Raw document text
        doc_id: Document UUID
        source: Source type ("text_layer", "api", etc)
        min_chunk_len: Minimum chunk length
        max_chunk_len: Maximum chunk length

    Returns:
        List of NormativeChunk objects
    """
    pages = [PageContent(page_num=1, text=text, source=source)]
    chunker = NormativeChunker(min_len=min_chunk_len, max_len=max_chunk_len)
    return chunker.chunk(pages, doc_id=doc_id)
