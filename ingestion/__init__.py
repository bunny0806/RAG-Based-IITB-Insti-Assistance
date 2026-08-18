"""Ingestion package for document loading workflows."""

from .base_loader import BaseLoader, Document
from .ingestion_pipeline import IngestionPipeline
from .loader_factory import LoaderFactory
from .pdf_loader import PDFLoader
from .web_loader import WebLoader

__all__ = [
    "BaseLoader",
    "Document",
    "IngestionPipeline",
    "LoaderFactory",
    "PDFLoader",
    "WebLoader",
]
