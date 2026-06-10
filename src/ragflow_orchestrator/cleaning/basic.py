from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser


class BasicTextCleaner:
    """Minimal cleaner that normalizes whitespace and strips control chars."""

    _control_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
    _spaces = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        without_controls = self._control_chars.sub("", text)
        normalized = self._spaces.sub(" ", without_controls)
        return normalized.strip()


class _HtmlTextExtractor(HTMLParser):
    _block_tags = {
        "article",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "ol",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if lowered in self._block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if lowered in self._block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if data.strip():
            self.parts.append(data)

    def get_text(self) -> str:
        return unescape("".join(self.parts))


class MarkupAwareTextCleaner:
    """Cleaner for document-like text that strips HTML and common Markdown syntax."""

    _control_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
    _markdown_link = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
    _markdown_image = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
    _autolink = re.compile(r"<(https?://[^>\s]+)>")
    _reference_link = re.compile(r"^\s*\[([^\]]+)\]:\s*(https?://\S+)\s*$", re.MULTILINE)
    _fence_marker = re.compile(r"^(```|~~~)[^\n]*$", re.MULTILINE)
    _inline_code = re.compile(r"`([^`]+)`")
    _strong_emphasis = re.compile(r"(?<!\w)(\*\*|__)(?=\S)(.+?)(?<=\S)\1(?!\w)")
    _light_emphasis = re.compile(r"(?<!\w)(\*|_)(?=\S)(.+?)(?<=\S)\1(?!\w)")
    _heading_prefix = re.compile(r"^\s*#{1,6}\s+")
    _unordered_list_prefix = re.compile(r"^\s*[-*+•]\s+")
    _ordered_list_prefix = re.compile(r"^\s*\d+[.)]\s+")
    _quote_prefix = re.compile(r"^\s*>\s?")
    _horizontal_rule = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
    _table_delimiter_line = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
    _spaces = re.compile(r"[ \t]+")
    _blank_lines = re.compile(r"\n{3,}")
    _html_tag = re.compile(r"<\s*[a-zA-Z][^>]*>")
    _url_only_line = re.compile(r"^https?://\S+$", re.IGNORECASE)
    _boilerplate_line = re.compile(
        r"(cookie|privacy policy|all rights reserved|terms of service|newsletter|subscribe|accept all)",
        re.IGNORECASE,
    )

    def clean(self, text: str) -> str:
        if not text:
            return ""

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = self._control_chars.sub("", normalized)
        normalized = self._autolink.sub(r"\1", normalized)
        normalized = self._reference_link.sub(r"\1: \2", normalized)
        normalized = self._markdown_image.sub(self._replace_markdown_image, normalized)
        normalized = self._markdown_link.sub(r"\1: \2", normalized)
        normalized = self._fence_marker.sub("", normalized)
        normalized = self._inline_code.sub(r"\1", normalized)
        normalized = self._strong_emphasis.sub(r"\2", normalized)
        normalized = self._light_emphasis.sub(r"\2", normalized)
        normalized = self._normalize_markdown_lines(normalized)
        normalized = self._strip_html(normalized)
        normalized = self._remove_noisy_lines(normalized)
        normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
        normalized = self._spaces.sub(" ", normalized)
        normalized = self._blank_lines.sub("\n\n", normalized)
        return normalized.strip()

    @staticmethod
    def _replace_markdown_image(match: re.Match[str]) -> str:
        alt_text = match.group(1).strip()
        url = match.group(2).strip()
        if alt_text:
            return f"Image {alt_text}: {url}"
        return url

    def _strip_html(self, text: str) -> str:
        if not self._html_tag.search(text):
            return text
        parser = _HtmlTextExtractor()
        parser.feed(text)
        parser.close()
        return parser.get_text()

    def _normalize_markdown_lines(self, text: str) -> str:
        lines: list[str] = []
        for line in text.split("\n"):
            current = line
            if self._horizontal_rule.match(current):
                lines.append("")
                continue
            if self._table_delimiter_line.match(current):
                continue
            if self._looks_like_table_row(current):
                current = self._normalize_table_row(current)

            changed = True
            while changed:
                changed = False
                for pattern in (
                    self._heading_prefix,
                    self._unordered_list_prefix,
                    self._ordered_list_prefix,
                    self._quote_prefix,
                ):
                    updated = pattern.sub("", current, count=1)
                    if updated != current:
                        current = updated
                        changed = True
            lines.append(current)
        return "\n".join(lines)

    @staticmethod
    def _looks_like_table_row(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.count("|") >= 2

    @staticmethod
    def _normalize_table_row(line: str) -> str:
        stripped = line.strip().strip("|")
        cells = [cell.strip() for cell in stripped.split("|")]
        cells = [cell for cell in cells if cell]
        return "; ".join(cells)

    def _remove_noisy_lines(self, text: str) -> str:
        cleaned_lines: list[str] = []
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                cleaned_lines.append("")
                continue
            if self._url_only_line.match(line):
                continue
            if self._boilerplate_line.search(line) and len(line) < 180:
                continue
            cleaned_lines.append(raw_line)
        return "\n".join(cleaned_lines)
