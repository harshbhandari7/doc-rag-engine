from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class LoadStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    SKIPPED = "skipped"


@dataclass
class PageContent:
    """Text extracted from a single document page."""

    page_number: int
    text: str


@dataclass
class DocumentMetadata:
    """Metadata about a loaded document."""

    source: Path
    filename: str
    format: str
    file_size_bytes: int
    total_pages: int
    loaded_pages: int  # count after filtering empty pages


@dataclass
class LoadedDocument:
    """Result of loading and converting a single document.

    *content* holds the full document as markdown.
    *pages* holds per-page text with empty pages already filtered out.
    """

    content: str
    pages: list[PageContent]
    metadata: DocumentMetadata
    status: LoadStatus
    errors: list[str] = field(default_factory=list)
