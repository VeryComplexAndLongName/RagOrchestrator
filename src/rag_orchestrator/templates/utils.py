from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse

from rag_orchestrator.templates.models import LanguageMode


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_map = {key.lower(): value for key, value in attrs}
        href = attrs_map.get("href")
        if href:
            self.links.append(href)


class TextExtractor(HTMLParser):
    _block_tags = {"p", "div", "br", "li", "tr", "td", "th", "section", "article", "header", "footer", "main", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag in self._block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag in self._block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self.parts.append(text)

    def get_text(self) -> str:
        text = unescape(" ".join(self.parts))
        text = re.sub(r"[\t\r\f\v]+", " ", text)
        text = re.sub(r" *\n+ *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ ]{2,}", " ", text)
        return text.strip()


def normalize_url(base_url: str, href: str) -> str:
    joined = urljoin(base_url, href)
    no_fragment, _ = urldefrag(joined)
    return no_fragment


def is_same_domain(url_a: str, url_b: str) -> bool:
    return urlparse(url_a).netloc.lower() == urlparse(url_b).netloc.lower()


def extract_links(base_url: str, html: str) -> list[str]:
    parser = LinkParser()
    parser.feed(html)
    links = [normalize_url(base_url, href) for href in parser.links]
    return [link for link in links if urlparse(link).scheme in {"http", "https"}]


def extract_text_from_html(html: str) -> str:
    extractor = TextExtractor()
    extractor.feed(html)
    extractor.close()
    return extractor.get_text()


def html_to_text(text: str) -> str:
    if not text:
        return ""
    if not re.search(r"<\s*[a-zA-Z][^>]*>", text):
        return text.strip()
    return extract_text_from_html(text)


def detect_language(text: str, mode: LanguageMode) -> str:
    if mode == LanguageMode.FORCE_RU:
        return "ru"
    if mode == LanguageMode.FORCE_EN:
        return "en"
    if mode == LanguageMode.MIXED:
        return "mixed"

    cyrillic = sum(1 for char in text if "\u0400" <= char <= "\u04FF")
    latin = sum(1 for char in text if ("a" <= char.lower() <= "z"))

    if cyrillic == 0 and latin == 0:
        return "unknown"
    if cyrillic > latin * 1.2:
        return "ru"
    if latin > cyrillic * 1.2:
        return "en"
    return "mixed"
