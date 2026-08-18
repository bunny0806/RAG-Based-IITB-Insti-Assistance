"""Web document loader using requests and BeautifulSoup."""

from __future__ import annotations

from typing import Any, Dict
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .base_loader import BaseLoader, Document


class WebLoader(BaseLoader):
    """Load text content from public web pages."""

    def __init__(self, source: str) -> None:
        super().__init__(source=source, document_type="web")
        self._metadata: Dict[str, Any] = {}

    def load(self) -> Document:
        """Fetch and parse a web page into a standardized document."""
        if not self._is_valid_url(self.source):
            raise ValueError(f"Invalid web URL: {self.source}")

        try:
            response = requests.get(
                self.source,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            self.logger.error("Failed to fetch web page %s: %s", self.source, exc)
            raise RuntimeError(f"Unable to fetch web page: {self.source}") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        self._remove_noise(soup)

        title = self._extract_title(soup)
        content = self._extract_visible_text(soup)

        self._metadata = {
            "title": title,
            "url": self.source,
            "filename": title or self._default_filename(),
            "document_name": title or self._default_filename(),
            "page_number": 1,
            "total_pages": 1,
        }

        document = Document(
            content=content,
            metadata=self._metadata,
            source=self.source,
            document_type=self.document_type,
        )
        self.validate(document)
        self.logger.info("Successfully loaded web document from %s", self.source)
        return document

    def validate(self, document: Document) -> None:
        """Ensure the loaded document contains content and metadata."""
        if not isinstance(document.content, str) or not document.content.strip():
            raise ValueError("Web content is empty.")
        if not document.metadata.get("title"):
            raise ValueError("Web document metadata is missing a title.")

    def get_metadata(self) -> Dict[str, Any]:
        """Return the metadata collected for the last load operation."""
        return self._metadata

    def _extract_title(self, soup: BeautifulSoup) -> str:
        title_tag = soup.title
        if title_tag and title_tag.get_text(strip=True):
            return title_tag.get_text(strip=True)
        parsed_url = urlparse(self.source)
        return parsed_url.netloc or self._default_filename()

    def _extract_visible_text(self, soup: BeautifulSoup) -> str:
        text_blocks: list[str] = []
        for element in soup.find_all(["p", "div", "li", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "span"]):
            text = " ".join(element.get_text(" ", strip=True).split())
            if text:
                text_blocks.append(text)
        return "\n\n".join(text_blocks).strip()

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

    def _default_filename(self) -> str:
        parsed_url = urlparse(self.source)
        return parsed_url.path.strip("/") or parsed_url.netloc

    def _is_valid_url(self, value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
