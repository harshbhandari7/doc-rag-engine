from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.exceptions import ConversionError

from configs.settings import AppConfig
from src.ingestion.models import (
    DocumentMetadata,
    LoadedDocument,
    LoadStatus,
    PageContent,
)

logger = logging.getLogger(__name__)

# Extensions that docling can handle natively.
_EXT_TO_FORMAT: dict[str, InputFormat] = {
    ".pdf": InputFormat.PDF,
    ".docx": InputFormat.DOCX,
    ".pptx": InputFormat.PPTX,
    ".html": InputFormat.HTML,
    ".htm": InputFormat.HTML,
    ".md": InputFormat.MD,
    ".csv": InputFormat.CSV,
    ".xlsx": InputFormat.XLSX,
}

# Plain-text extensions that we read directly (no docling).
_PLAINTEXT_EXTS: set[str] = {".txt", ".text", ".log"}

# Union of every extension we accept.
_SUPPORTED_EXTS: set[str] = set(_EXT_TO_FORMAT) | _PLAINTEXT_EXTS


class DocumentLoader:
    """Load documents from files or directories and return structured results.

    Supported formats:
        PDF, DOCX, PPTX, HTML, Markdown, CSV, XLSX  (via docling)
        TXT, TEXT, LOG                                (read directly)

    Usage::

        loader = DocumentLoader()
        doc = loader.load("data/raw/arxiv_papers/attention_need.pdf")
        print(doc.status, len(doc.pages))

        # Directory scan
        for doc in loader.load_directory("data/raw"):
            ...

        # Batch / streaming
        for doc in loader.load_batch(list_of_paths):
            ...
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self._converter = self._build_converter()
        logger.info(
            "DocumentLoader initialised  ocr=%s  table_structure=%s  timeout=%s",
            self.config.ocr_enabled,
            self.config.table_structure_enabled,
            self.config.document_timeout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> LoadedDocument:
        """Load a single file and return a ``LoadedDocument``."""
        path = Path(path).resolve()
        logger.info("Loading %s", path)

        if not path.exists():
            logger.error("File not found: %s", path)
            return self._failure(path, f"File not found: {path}")

        if not path.is_file():
            logger.error("Path is not a file: %s", path)
            return self._failure(path, f"Path is not a file: {path}")

        ext = path.suffix.lower()

        if ext in _PLAINTEXT_EXTS:
            return self._load_plaintext(path)

        if ext not in _EXT_TO_FORMAT:
            logger.warning("Unsupported format: %s", ext)
            return self._failure(path, f"Unsupported file extension: {ext}")

        return self._load_with_docling(path)

    def load_directory(
        self,
        dir_path: str | Path,
        pattern: str = "**/*",
    ) -> Iterator[LoadedDocument]:
        """Yield a ``LoadedDocument`` for every supported file under *dir_path*.

        Files are discovered with ``Path.glob(pattern)`` and sorted by name
        for deterministic ordering.
        """
        dir_path = Path(dir_path).resolve()
        if not dir_path.is_dir():
            logger.error("Not a directory: %s", dir_path)
            return

        files = sorted(
            p
            for p in dir_path.glob(pattern)
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS
        )
        logger.info("Found %d supported file(s) in %s", len(files), dir_path)
        yield from self.load_batch(files)

    def load_batch(self, paths: Iterable[str | Path]) -> Iterator[LoadedDocument]:
        """Process multiple documents, yielding results as they complete.

        Docling-compatible files are handed to ``convert_all`` (which may
        process them concurrently).  Plain-text files are read sequentially.
        """
        docling_paths: list[Path] = []
        plaintext_paths: list[Path] = []

        for p in paths:
            p = Path(p).resolve()
            ext = p.suffix.lower()
            if ext in _PLAINTEXT_EXTS:
                plaintext_paths.append(p)
            elif ext in _EXT_TO_FORMAT:
                docling_paths.append(p)
            else:
                logger.warning("Skipping unsupported file: %s", p)
                yield self._failure(p, f"Unsupported file extension: {ext}")

        # --- docling batch (streaming iterator) ---
        if docling_paths:
            logger.info("Batch-converting %d document(s) via docling", len(docling_paths))
            try:
                results = self._converter.convert_all(
                    [str(p) for p in docling_paths],
                    raises_on_error=False,
                )
                for result in results:
                    source = Path(str(result.input.file))
                    yield self._process_result(result, source)
            except Exception:
                logger.exception("Batch conversion failed")
                for p in docling_paths:
                    yield self._failure(p, "Batch conversion failed unexpectedly")

        # --- plain text ---
        for p in plaintext_paths:
            yield self._load_plaintext(p)

    # ------------------------------------------------------------------
    # Docling conversion
    # ------------------------------------------------------------------

    def _build_converter(self) -> DocumentConverter:
        pipeline_opts = PdfPipelineOptions(
            do_ocr=self.config.ocr_enabled,
            do_table_structure=self.config.table_structure_enabled,
        )
        if self.config.document_timeout is not None:
            pipeline_opts.document_timeout = self.config.document_timeout

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_opts,
                ),
            },
        )

    def _load_with_docling(self, path: Path) -> LoadedDocument:
        try:
            result = self._converter.convert(str(path), raises_on_error=False)
            return self._process_result(result, path)
        except ConversionError as exc:
            logger.error("Conversion error for %s: %s", path, exc)
            return self._failure(path, str(exc))
        except Exception:
            logger.exception("Unexpected error loading %s", path)
            return self._failure(path, f"Unexpected error loading {path}")

    def _process_result(self, result, path: Path) -> LoadedDocument:
        """Turn a docling ``ConversionResult`` into a ``LoadedDocument``."""
        errors = [e.error_message for e in result.errors]

        status = {
            ConversionStatus.SUCCESS: LoadStatus.SUCCESS,
            ConversionStatus.PARTIAL_SUCCESS: LoadStatus.PARTIAL,
            ConversionStatus.FAILURE: LoadStatus.FAILURE,
            ConversionStatus.SKIPPED: LoadStatus.SKIPPED,
        }.get(result.status, LoadStatus.FAILURE)

        if status == LoadStatus.FAILURE:
            logger.error("Conversion failed for %s: %s", path, errors)
            return self._failure(path, *errors)

        if errors:
            logger.warning("Conversion of %s produced warnings: %s", path, errors)

        doc = result.document

        # Full document content as markdown.
        content = doc.export_to_markdown()

        # Per-page plain text, skipping empty / near-empty pages.
        pages: list[PageContent] = []
        total_pages = len(doc.pages) if doc.pages else 0

        for page_no in sorted(doc.pages) if doc.pages else []:
            page_text = doc.export_to_text(page_no=page_no).strip()
            if len(page_text) < self.config.min_page_chars:
                logger.debug(
                    "Skipping near-empty page %d of %s (%d chars)",
                    page_no,
                    path.name,
                    len(page_text),
                )
                continue
            pages.append(PageContent(page_number=page_no, text=page_text))

        # Merge adjacent tiny pages so downstream chunking gets usable segments.
        if self.config.merge_min_chars > 0:
            pages = self._merge_small_pages(pages)

        metadata = DocumentMetadata(
            source=path,
            filename=path.name,
            format=path.suffix.lower().lstrip("."),
            file_size_bytes=path.stat().st_size if path.exists() else 0,
            total_pages=total_pages,
            loaded_pages=len(pages),
        )

        logger.info(
            "Loaded %s  pages=%d/%d  status=%s",
            path.name,
            metadata.loaded_pages,
            metadata.total_pages,
            status.value,
        )
        return LoadedDocument(
            content=content,
            pages=pages,
            metadata=metadata,
            status=status,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Plain-text fallback
    # ------------------------------------------------------------------

    def _load_plaintext(self, path: Path) -> LoadedDocument:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        except Exception as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return self._failure(path, str(exc))

        content = text.strip()
        if not content:
            logger.warning("Empty text file: %s", path)
            return self._failure(path, "File is empty")

        pages = [PageContent(page_number=1, text=content)]
        metadata = DocumentMetadata(
            source=path,
            filename=path.name,
            format="txt",
            file_size_bytes=path.stat().st_size,
            total_pages=1,
            loaded_pages=1,
        )
        logger.info("Loaded plain-text file %s (%d bytes)", path.name, metadata.file_size_bytes)
        return LoadedDocument(
            content=content,
            pages=pages,
            metadata=metadata,
            status=LoadStatus.SUCCESS,
        )

    # ------------------------------------------------------------------
    # Chunk merging
    # ------------------------------------------------------------------

    def _merge_small_pages(self, pages: list[PageContent]) -> list[PageContent]:
        """Merge adjacent pages whose text is shorter than *merge_min_chars*.

        A short page is folded into the previous page.  If the very last page
        is still below the threshold after the forward pass it is folded into
        the one before it.
        """
        if len(pages) <= 1:
            return pages

        threshold = self.config.merge_min_chars
        merged: list[PageContent] = []

        for page in pages:
            if merged and len(merged[-1].text) < threshold:
                prev = merged[-1]
                merged[-1] = PageContent(
                    page_number=prev.page_number,
                    text=prev.text + "\n\n" + page.text,
                )
                logger.debug(
                    "Merged page %d into page %d (previous had %d chars)",
                    page.page_number,
                    prev.page_number,
                    len(prev.text),
                )
            else:
                merged.append(page)

        # Trailing short page → fold into the one before it.
        if len(merged) > 1 and len(merged[-1].text) < threshold:
            tail = merged.pop()
            merged[-1] = PageContent(
                page_number=merged[-1].page_number,
                text=merged[-1].text + "\n\n" + tail.text,
            )
            logger.debug(
                "Merged trailing page %d into page %d",
                tail.page_number,
                merged[-1].page_number,
            )

        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _failure(self, path: Path, *errors: str) -> LoadedDocument:
        metadata = DocumentMetadata(
            source=path,
            filename=path.name,
            format=path.suffix.lower().lstrip("."),
            file_size_bytes=path.stat().st_size if path.exists() else 0,
            total_pages=0,
            loaded_pages=0,
        )
        return LoadedDocument(
            content="",
            pages=[],
            metadata=metadata,
            status=LoadStatus.FAILURE,
            errors=list(errors),
        )
