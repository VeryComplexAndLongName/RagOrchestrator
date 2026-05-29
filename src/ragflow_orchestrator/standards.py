from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CodeMetadata(BaseModel):
    language: str
    repo: str | None = None
    file_path: str | None = None
    function: str | None = None


class TableMetadata(BaseModel):
    table_name: str
    schema_name: str | None = None
    row_count: int | None = None


class PdfMetadata(BaseModel):
    title: str | None = None
    page: int | None = None
    section: str | None = None


class HtmlMetadata(BaseModel):
    url: str | None = None
    heading: str | None = None
    dom_path: str | None = None


class WordMetadata(BaseModel):
    doc_title: str | None = None
    heading: str | None = None


class MixedMetadata(BaseModel):
    modality: str = Field(default="mixed")
    labels: list[str] = Field(default_factory=list)


METADATA_STANDARDS: dict[str, type[BaseModel]] = {
    "code": CodeMetadata,
    "table": TableMetadata,
    "pdf": PdfMetadata,
    "html": HtmlMetadata,
    "word": WordMetadata,
    "mixed": MixedMetadata,
}


def validate_metadata(kind: str, metadata: dict[str, Any]) -> dict[str, Any]:
    model = METADATA_STANDARDS.get(kind)
    if model is None:
        return metadata
    return model.model_validate(metadata).model_dump(exclude_none=True)
