from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class BoilerplateAggressiveness(str, Enum):
    OFF = "off"
    LIGHT = "light"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass(slots=True)
class BoilerplateRemovalResult:
    text: str
    removed_lines: int
    total_lines: int
    remover: str
    aggressiveness: BoilerplateAggressiveness


class BoilerplateRemover(Protocol):
    name: str

    def remove(
        self,
        text: str,
        *,
        document_type: str,
        aggressiveness: BoilerplateAggressiveness,
        metadata: dict[str, object] | None = None,
    ) -> BoilerplateRemovalResult:
        ...


class BoilerplateRemoverRegistry:
    def __init__(self) -> None:
        self._removers: dict[str, dict[str, BoilerplateRemover]] = {}
        self._defaults: dict[str, str] = {}

    def register(self, *, document_type: str, name: str, remover: BoilerplateRemover, default: bool = False) -> None:
        doc_type = document_type.strip().lower()
        self._removers.setdefault(doc_type, {})[name] = remover
        if default or doc_type not in self._defaults:
            self._defaults[doc_type] = name

    def unregister(self, *, document_type: str, name: str) -> None:
        doc_type = document_type.strip().lower()
        doc_removers = self._removers.get(doc_type)
        if not doc_removers:
            return
        doc_removers.pop(name, None)
        if not doc_removers:
            self._removers.pop(doc_type, None)
            self._defaults.pop(doc_type, None)
            return
        if self._defaults.get(doc_type) == name:
            self._defaults[doc_type] = next(iter(doc_removers.keys()))

    def set_default(self, *, document_type: str, name: str) -> None:
        doc_type = document_type.strip().lower()
        if name in self._removers.get(doc_type, {}):
            self._defaults[doc_type] = name

    def resolve(self, *, document_type: str, preferred_name: str | None = None) -> BoilerplateRemover | None:
        doc_type = document_type.strip().lower()
        doc_removers = self._removers.get(doc_type)
        if not doc_removers:
            return None
        if preferred_name and preferred_name in doc_removers:
            return doc_removers[preferred_name]
        default_name = self._defaults.get(doc_type)
        if default_name and default_name in doc_removers:
            return doc_removers[default_name]
        return next(iter(doc_removers.values()), None)


class NoOpBoilerplateRemover:
    name = "noop"

    def remove(
        self,
        text: str,
        *,
        document_type: str,
        aggressiveness: BoilerplateAggressiveness,
        metadata: dict[str, object] | None = None,
    ) -> BoilerplateRemovalResult:
        del document_type, metadata
        lines = text.splitlines()
        return BoilerplateRemovalResult(
            text=text,
            removed_lines=0,
            total_lines=len(lines),
            remover=self.name,
            aggressiveness=aggressiveness,
        )


class RuleBasedLineRemover:
    _url_only = re.compile(r"^https?://\S+$", re.IGNORECASE)
    _mostly_symbol = re.compile(r"^[\W_]{6,}$")
    _page_marker = re.compile(r"^\s*page\s+\d+\s*(?:of|/)\s*\d+\s*$", re.IGNORECASE)
    _generated_marker = re.compile(
        r"(generated on|generated at|exported on|exported at|report generated)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        name: str,
        keyword_pattern: str,
        extra_patterns: tuple[str, ...] = (),
        protected_pattern: str = r"$^",
    ) -> None:
        self.name = name
        self._keywords = re.compile(keyword_pattern, re.IGNORECASE)
        self._extra = [re.compile(pattern, re.IGNORECASE) for pattern in extra_patterns]
        self._protected = re.compile(protected_pattern, re.IGNORECASE)

    def remove(
        self,
        text: str,
        *,
        document_type: str,
        aggressiveness: BoilerplateAggressiveness,
        metadata: dict[str, object] | None = None,
    ) -> BoilerplateRemovalResult:
        del document_type, metadata
        lines = text.splitlines()
        if not lines or aggressiveness == BoilerplateAggressiveness.OFF:
            return BoilerplateRemovalResult(
                text=text,
                removed_lines=0,
                total_lines=len(lines),
                remover=self.name,
                aggressiveness=aggressiveness,
            )

        normalized_lines = [self._normalize(line) for line in lines]
        frequency = Counter(item for item in normalized_lines if item)
        repeated_threshold = {
            BoilerplateAggressiveness.LIGHT: 4,
            BoilerplateAggressiveness.BALANCED: 3,
            BoilerplateAggressiveness.AGGRESSIVE: 2,
            BoilerplateAggressiveness.OFF: 10**9,
        }[aggressiveness]

        cleaned: list[str] = []
        removed = 0
        for index, line in enumerate(lines):
            line_value = line.strip()
            if not line_value:
                cleaned.append("")
                continue

            normalized = normalized_lines[index]
            if self._protected.search(line_value):
                cleaned.append(line)
                continue

            if self._should_remove(
                line=line_value,
                normalized=normalized,
                frequency=frequency,
                aggressiveness=aggressiveness,
                repeated_threshold=repeated_threshold,
            ):
                removed += 1
                continue

            cleaned.append(line)

        compact = "\n".join(cleaned)
        compact = re.sub(r"\n{3,}", "\n\n", compact).strip()

        # Fail-safe: if heuristics became too aggressive, keep original content.
        if lines and removed / len(lines) > 0.55:
            return BoilerplateRemovalResult(
                text=text,
                removed_lines=0,
                total_lines=len(lines),
                remover=self.name,
                aggressiveness=aggressiveness,
            )

        return BoilerplateRemovalResult(
            text=compact,
            removed_lines=removed,
            total_lines=len(lines),
            remover=self.name,
            aggressiveness=aggressiveness,
        )

    def _should_remove(
        self,
        *,
        line: str,
        normalized: str,
        frequency: Counter[str],
        aggressiveness: BoilerplateAggressiveness,
        repeated_threshold: int,
    ) -> bool:
        if self._page_marker.search(line):
            return True
        if self._generated_marker.search(line):
            return True
        if self._mostly_symbol.match(line):
            return True

        if aggressiveness != BoilerplateAggressiveness.LIGHT and self._url_only.match(line):
            return True

        if self._keywords.search(line):
            max_len = {
                BoilerplateAggressiveness.LIGHT: 120,
                BoilerplateAggressiveness.BALANCED: 220,
                BoilerplateAggressiveness.AGGRESSIVE: 320,
                BoilerplateAggressiveness.OFF: 0,
            }[aggressiveness]
            if len(line) <= max_len:
                return True

        if any(pattern.search(line) for pattern in self._extra):
            return True

        if normalized and len(normalized) <= 240 and frequency.get(normalized, 0) >= repeated_threshold:
            return True

        return False

    @staticmethod
    def _normalize(line: str) -> str:
        return re.sub(r"\s+", " ", line.strip().lower())


class JustextHtmlRemover:
    name = "justext"

    def remove(
        self,
        text: str,
        *,
        document_type: str,
        aggressiveness: BoilerplateAggressiveness,
        metadata: dict[str, object] | None = None,
    ) -> BoilerplateRemovalResult:
        del document_type, metadata
        lines = text.splitlines()
        if not text.strip() or aggressiveness == BoilerplateAggressiveness.OFF:
            return BoilerplateRemovalResult(
                text=text,
                removed_lines=0,
                total_lines=len(lines),
                remover=self.name,
                aggressiveness=aggressiveness,
            )

        try:
            import justext  # type: ignore[import-not-found]
        except ImportError:
            return BoilerplateRemovalResult(
                text=text,
                removed_lines=0,
                total_lines=len(lines),
                remover=self.name,
                aggressiveness=aggressiveness,
            )

        stoplist = justext.get_stoplist("English")
        paragraphs = justext.justext(text, stoplist)
        kept = [paragraph.text.strip() for paragraph in paragraphs if not paragraph.is_boilerplate and paragraph.text.strip()]
        if not kept:
            return BoilerplateRemovalResult(
                text=text,
                removed_lines=0,
                total_lines=len(lines),
                remover=self.name,
                aggressiveness=aggressiveness,
            )

        cleaned = "\n\n".join(kept).strip()
        removed = max(0, len(lines) - len(cleaned.splitlines()))
        return BoilerplateRemovalResult(
            text=cleaned,
            removed_lines=removed,
            total_lines=len(lines),
            remover=self.name,
            aggressiveness=aggressiveness,
        )


def parse_aggressiveness(value: object | None) -> BoilerplateAggressiveness:
    if value is None:
        return BoilerplateAggressiveness.BALANCED
    normalized = str(value).strip().lower()
    if normalized in BoilerplateAggressiveness._value2member_map_:
        return BoilerplateAggressiveness(normalized)
    return BoilerplateAggressiveness.BALANCED


def default_boilerplate_registry() -> BoilerplateRemoverRegistry:
    registry = BoilerplateRemoverRegistry()

    html_keywords = (
        r"(cookie|privacy policy|terms of service|all rights reserved|newsletter|subscribe|accept all|"
        r"manage preferences|consent|javascript required|sign in|log in)"
    )
    generic_keywords = (
        r"(cookie|privacy policy|terms of service|all rights reserved|unsubscribe|"
        r"confidential|do not distribute|internal use only|legal notice)"
    )

    html_rules = RuleBasedLineRemover(
        name="html-rules",
        keyword_pattern=html_keywords,
        extra_patterns=(r"^(home|about|contact|login|sign up|share)$",),
    )
    markdown_rules = RuleBasedLineRemover(name="markdown-rules", keyword_pattern=generic_keywords)
    txt_rules = RuleBasedLineRemover(name="txt-rules", keyword_pattern=generic_keywords)
    pdf_rules = RuleBasedLineRemover(
        name="pdf-rules",
        keyword_pattern=generic_keywords,
        extra_patterns=(r"^chapter\s+\d+\s*$",),
    )
    docx_rules = RuleBasedLineRemover(
        name="docx-rules",
        keyword_pattern=generic_keywords,
        extra_patterns=(r"^document id:\s*\S+",),
        protected_pattern=r"^heading\b|^title\b",
    )
    xlsx_rules = RuleBasedLineRemover(
        name="xlsx-rules",
        keyword_pattern=r"(generated on|generated at|exported on|exported at|readme|instructions)",
        extra_patterns=(r"^sheet\s*:\s*", r"^page\s*\d+\s*of\s*\d+\s*$"),
    )
    xml_rules = RuleBasedLineRemover(
        name="xml-rules",
        keyword_pattern=r"(xmlns|schema|xsi:|all rights reserved|privacy policy)",
    )
    csv_rules = RuleBasedLineRemover(
        name="csv-rules",
        keyword_pattern=r"(generated on|generated at|exported on|exported at|all rights reserved)",
    )

    for doc_type, remover in (
        ("html", html_rules),
        ("markdown", markdown_rules),
        ("txt", txt_rules),
        ("pdf", pdf_rules),
        ("docx", docx_rules),
        ("xlsx", xlsx_rules),
        ("xml", xml_rules),
        ("csv", csv_rules),
    ):
        registry.register(document_type=doc_type, name=remover.name, remover=remover, default=True)

    registry.register(document_type="html", name=JustextHtmlRemover.name, remover=JustextHtmlRemover(), default=False)

    noop = NoOpBoilerplateRemover()
    for doc_type in ("json", "code", "unsupported"):
        registry.register(document_type=doc_type, name=noop.name, remover=noop, default=True)

    return registry
