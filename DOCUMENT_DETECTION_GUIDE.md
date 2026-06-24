# Universal Document Type Detector

## Overview

`UniversalDocumentDetector` is a comprehensive document type detection system with support for **25+ formats**:

### Supported Types

```
Office documents:     docx, xlsx, pptx, odt, ods
PDF variants:         pdf_text, pdf_scan
Text formats:         txt, md, html, xml, json, csv
Code:                 python, javascript, typescript, java, go, rust, c++, c, c#
Configuration:        yaml, toml, ini, properties
```

## Architecture

### Multi-layer Detection Strategy

The detector uses **5 methods** in order of reliability:

```
1️⃣  Magic Bytes (binary signature)     [Confidence: 0.95]
    └─ the most reliable method
    └─ analyzes the first bytes of the file
    └─ works for all binary formats

2️⃣  File Extension                     [Confidence: 0.85]
    └─ fast and usually reliable
    └─ uses file extension
    └─ may fail if the file was renamed

3️⃣  MIME Type (python-magic)           [Confidence: 0.90]
    └─ detects MIME type from content
    └─ requires python-magic
    └─ highly reliable for standard formats

4️⃣  Content Heuristics                 [Confidence: 0.75-0.95]
    └─ analyzes text structure
    └─ looks for format markers (<?xml, {, #, etc.)
    └─ works for text formats

5️⃣  Fallback (default)                 [Confidence: 0.5]
    └─ used if nothing else worked
    └─ returns UNSUPPORTED or TXT
```

### PDF Sub-classification

For PDF files, an additional subtype is detected:

- **pdf_text**: PDF with a text layer
  - Text can be extracted directly via PyMuPDF
  - Usually created digitally

- **pdf_scan**: PDF from scanned images
  - Requires OCR (optical character recognition)
  - May come from scanner or camera captures

## API

### Base Class: UniversalDocumentDetector

```python
from ragflow_orchestrator.document_detector import UniversalDocumentDetector

detector = UniversalDocumentDetector(
    enable_ocr=True,              # Enable OCR for PDF
    ocr_callable=my_custom_ocr,   # Optional custom OCR function
)
```

#### Method: `detect()`

```python
detection = detector.detect(
    file_path=Path("document.pdf"),    # File path
    file_data=b"...",                  # Raw file bytes
    text="...",                        # Text content
)

# Result:
# - document_type: DocumentType (enum)
# - mime_type: str (if detected)
# - source: str (detection method)
# - confidence: float (0.0-1.0)
# - pdf_subtype: str (for PDFs)
```

**Parameters:**
- `file_path` (optional): Path for quick extension-based detection
- `file_data` (optional): Raw bytes for magic bytes and MIME analysis
- `text` (optional): Text content for structural analysis

**Returns:** `DocumentDetection` with full details

#### Method: `detect_pdf_subtype()`

```python
pdf_detection = detector.detect_pdf_subtype(Path("regulations.pdf"))

# Result:
# - document_type: DocumentType.PDF_TEXT or DocumentType.PDF_SCAN
# - pdf_subtype: "text_layer" or "ocr"
# - confidence: float
```

**Uses:**
- PyMuPDF for PDF structure analysis
- Text-layer presence checks
- Large image detection (scan indicator)

### Convenience Functions

#### `detect_document_type()`

Quickly detect a single document type:

```python
detection = detect_document_type(
    file_path=Path("regulations.pdf"),
)

print(f"Type: {detection.document_type}")      # DocumentType.PDF_TEXT
print(f"Confidence: {detection.confidence}")   # 0.95
print(f"Source: {detection.source}")           # "magic_bytes"
```

#### `detect_pdf_type()`

Specialized PDF subtype detection:

```python
detection = detect_pdf_type(Path("regulations.pdf"))

if detection.document_type == DocumentType.PDF_TEXT:
    print("✓ PDF with text layer - use direct extraction")
elif detection.document_type == DocumentType.PDF_SCAN:
    print("✓ Scanned PDF - use OCR")
```

#### `batch_detect_documents()`

Batch detection for multiple files:

```python
results = batch_detect_documents([
    "doc1.pdf",
    "doc2.xlsx",
    "doc3.md",
])

for file_path, detection in results.items():
    print(f"{file_path}: {detection.document_type.value}")
```

## Usage Examples

### 1️⃣ Basic Type Detection

```python
from ragflow_orchestrator.document_detector import detect_document_type
from pathlib import Path

file_path = Path("regulations.pdf")
detection = detect_document_type(file_path=file_path)

print(f"Type: {detection.document_type}")    # pdf_text or pdf_scan
print(f"Confidence: {detection.confidence:.1%}")  # 95%
```

### 2️⃣ Detection with File Data

```python
# For reliable detection, include file bytes
with open("document.docx", "rb") as f:
    file_data = f.read()

detection = detect_document_type(
    file_path=Path("document.docx"),
    file_data=file_data,
)

print(f"Type: {detection.document_type}")  # docx
print(f"Source: {detection.source}")       # magic_bytes or extension
```

### 3️⃣ PDF Subtype Detection

```python
from ragflow_orchestrator.document_detector import UniversalDocumentDetector
from ragflow_orchestrator.cleaning.normative_parser import parse_normative_pdf

detector = UniversalDocumentDetector(enable_ocr=True)
detection = detector.detect_pdf_subtype("regulations.pdf")

if detection.document_type.value == "pdf_text":
    # Use standard normative parser
    chunks = parse_normative_pdf(
        pdf_path="regulations.pdf",
        doc_id="doc_uuid",
        doc_title="GOST 27751-2014",
    )
else:
    # OCR is required for scanned PDFs
    chunks = parse_normative_pdf(
        pdf_path="regulations.pdf",
        doc_id="doc_uuid",
        doc_title="GOST 27751-2014",
    )
```

### 4️⃣ Pipeline Integration

```python
from ragflow_orchestrator.versioned_pipeline import VersionedDocumentPipeline
from ragflow_orchestrator.document_detector import (
    UniversalDocumentDetector,
    DocumentType,
)

detector = UniversalDocumentDetector(enable_ocr=True)

def process_document(file_path, pipeline):
    """Process a document with automatic type detection."""

    # Read file bytes
    with open(file_path, "rb") as f:
        file_data = f.read()

    # Detect type
    detection = detector.detect(
        file_path=file_path,
        file_data=file_data,
    )

    # Extract text by type
    if detection.document_type == DocumentType.DOCX:
        text = extract_docx_text(file_path)
    elif detection.document_type in (DocumentType.PDF_TEXT, DocumentType.PDF_SCAN):
        text = extract_pdf_text(file_path, detection.document_type)
    elif detection.document_type == DocumentType.XLSX:
        text = extract_xlsx_text(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    # Ingest into pipeline
    result = pipeline.ingest_document(
        file_path=file_path,
        text=text,
        auto_tags=[detection.document_type.value],
    )

    return result
```

### 5️⃣ Confidence-based Processing

```python
def process_with_confidence_check(file_path):
    detector = UniversalDocumentDetector(enable_ocr=False)
    detection = detector.detect(file_path=file_path)

    if detection.confidence > 0.9:
        # High confidence - process directly
        return process_as_type(detection.document_type)

    elif detection.confidence > 0.7:
        # Medium confidence - process with warning
        print(f"Warning: Detected as {detection.document_type}, confidence: {detection.confidence:.1%}")
        return process_as_type(detection.document_type)

    else:
        # Low confidence - ask for confirmation
        raise ValueError(f"Cannot reliably detect {file_path}, confidence: {detection.confidence:.1%}")
```

### 6️⃣ Specialized Detectors by Mode

```python
# Text-only detection
text_detector = UniversalDocumentDetector(enable_ocr=False)

# OCR-enabled PDF detection
pdf_detector = UniversalDocumentDetector(enable_ocr=True)

# Custom OCR detector
def my_ocr_checker(pdf_path):
    """Checks whether a PDF is scanned."""
    # Your custom logic
    return is_scan

custom_detector = UniversalDocumentDetector(
    enable_ocr=True,
    ocr_callable=my_ocr_checker,
)
```

## Implementation Details

### Magic Bytes

The first bytes are used to identify format:

```python
# PDF
b"%PDF" -> DocumentType.PDF_TEXT

# ZIP-based (Office)
b"PK\x03\x04" -> inspect internal structure
  |- word/ + document.xml -> DOCX
  |- xl/ + workbook.xml -> XLSX
  |- ppt/ + presentation.xml -> PPTX

# OLE-based (legacy Office)
b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" -> OLDOFFICE

# Images
b"\xff\xd8\xff" -> JPEG
b"\x89PNG\r\n\x1a\n" -> PNG
b"GIF8" -> GIF
```

### MIME Type Detection

Uses `python-magic` for MIME detection:

```bash
pip install python-magic
```

Examples:
- `application/pdf` -> PDF
- `application/vnd.openxmlformats-officedocument.wordprocessingml.document` -> DOCX
- `text/plain` -> TXT
- `application/json` -> JSON

### Content Heuristics

Analyzes text patterns:

```python
# XML
if starts_with("<?xml") -> XML

# JSON
if starts_with("[" or "{") and valid_json -> JSON

# YAML
if has ":" and key:value structure -> YAML

# Markdown
if markers present: # ## ### ``` - * | -> MARKDOWN

# CSV
if multiple rows with delimiters -> CSV

# Code
if keywords present: def, class, function, import -> CODE
```

## Performance Tips

### Single File

```python
# Optimal: pass path + bytes
with open(file_path, "rb") as f:
    file_data = f.read()

detection = detector.detect(
    file_path=file_path,
    file_data=file_data,
)
# Cost: magic bytes (95% confidence) + extension
```

### Folder Batch

```python
# Use batch_detect_documents()
results = batch_detect_documents(folder_path.glob("*"))

# Benefits:
# - parallel processing (if enabled)
# - result caching
# - fewer repeated checks
```

### Large Files

```python
# Read only first 8KB instead of full file
with open(large_file, "rb") as f:
    file_data = f.read(8192)

detection = detector.detect(
    file_path=large_file,
    file_data=file_data,
    # text can be omitted for binary formats
)
```

## Troubleshooting

### "python-magic not found"

```bash
pip install python-magic-bin  # Windows
pip install python-magic      # Linux/macOS
```

### "PyMuPDF not found" (for PDF analysis)

```bash
pip install PyMuPDF
```

### "PaddleOCR not found" (for scanned PDFs)

```bash
pip install paddleocr
python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ru')"
```

### Wrong Type Detection

1. Verify the file extension.
2. Ensure `file_data` is passed for reliable analysis.
3. Check `detection.source` to understand which strategy was used.
4. If `confidence < 0.7`, request user confirmation.

## Roadmap

Potential improvements:

- [ ] ML-based classification for ambiguous cases
- [ ] Parallelization improvements for `batch_detect_documents()`
- [ ] Detection result caching
- [ ] Support for rare formats (7z, rar, iso, etc.)
- [ ] Encrypted/protected file recognition
- [ ] Antivirus integration for infected files
