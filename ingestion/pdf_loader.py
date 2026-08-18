"""PDF document loader with pdfplumber first and pypdf fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from .base_loader import BaseLoader, Document


class PDFLoader(BaseLoader):
    """Load PDF documents using pdfplumber and fall back to pypdf."""

    def __init__(self, source: str) -> None:
        super().__init__(source=source, document_type="pdf")
        self._metadata: Dict[str, Any] = {}

    def load(self) -> Document:
        """Load text content from a PDF file."""
        path = Path(self.source)

        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {self.source}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file type for PDF loader: {path.suffix}")

        try:
            text, total_pages = self._extract_with_pdfplumber(path)
        except Exception as pdfplumber_error:  # pragma: no cover - defensive fallback
            self.logger.warning(
                "pdfplumber failed for %s: %s. Falling back to pypdf.",
                self.source,
                pdfplumber_error,
            )
            try:
                text, total_pages = self._extract_with_pypdf(path)
            except Exception as pypdf_error:
                self.logger.error("Failed to load PDF %s: %s", self.source, pypdf_error)
                raise RuntimeError(f"Unable to read PDF document: {self.source}") from pypdf_error

        self._metadata = {
            "filename": path.name,
            "document_name": path.stem,
            "page_number": 1,
            "total_pages": total_pages,
        }

        document = Document(
            content=text,
            metadata=self._metadata,
            source=self.source,
            document_type=self.document_type,
        )
        self.validate(document)
        self.logger.info("Successfully loaded PDF document from %s", self.source)
        return document

    def validate(self, document: Document) -> None:
        """Ensure the loaded document contains readable content."""
        if not isinstance(document.content, str) or not document.content.strip():
            raise ValueError("PDF content is empty.")
        if not document.metadata.get("filename"):
            raise ValueError("PDF metadata is missing filename.")

    def get_metadata(self) -> Dict[str, Any]:
        """Return the metadata collected for the last load operation."""
        return self._metadata

    def _extract_with_pdfplumber(self, path: Path) -> Tuple[str, int]:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            text_parts: list[str] = []
            for page in pdf.pages:
                extracted_text = page.extract_text() or ""
                if extracted_text.strip():
                    text_parts.append(extracted_text.strip())
            text = "\n\n".join(text_parts).strip()
            return text, len(pdf.pages)

    def _extract_with_pypdf(self, path: Path) -> Tuple[str, int]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text_parts: list[str] = []
        for page in reader.pages:
            extracted_text = page.extract_text() or ""
            if extracted_text.strip():
                text_parts.append(extracted_text.strip())
        text = "\n\n".join(text_parts).strip()
        return text, len(reader.pages)
