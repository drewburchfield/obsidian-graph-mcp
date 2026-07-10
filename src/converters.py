"""
Document converters: turn any supported file into embeddable text.

This is the only genuinely new piece versus the markdown-only pipeline. Everything
downstream (chunking, embedding, pgvector) is unchanged - it just needs a string.

Routing:
  - Plain text (.md, .markdown, .txt): read as-is.
  - Rich documents (.pdf, .docx, .pptx, .html/.htm): Microsoft markitdown ->
    markdown text.
  - Spreadsheets (.csv, .xlsx, .xls): a deterministic "table card" (purpose,
    column schema, dtypes, row count, sample rows) rather than a raw cell dump.
    Small labeled tables also get their full markdown appended; large data dumps
    stay card-only so the file is findable without flooding the index with a
    meaningless wall-of-numbers vector. (See table-RAG research: TableRAG,
    LlamaIndex recursive retriever, PIPER.)

convert_file() returns the text to embed, or None to skip the file.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

# Extension routing
PASSTHROUGH_EXTS = {".md", ".markdown", ".txt"}
MARKITDOWN_EXTS = {".pdf", ".docx", ".pptx", ".html", ".htm"}
TABULAR_EXTS = {".csv", ".xlsx", ".xls"}
SUPPORTED_EXTS = PASSTHROUGH_EXTS | MARKITDOWN_EXTS | TABULAR_EXTS

# A spreadsheet whose full markdown render is below this many characters is a
# "small labeled table" and gets embedded whole in addition to its card.
SMALL_TABLE_CHAR_LIMIT = 6000
SAMPLE_ROWS = 3

# Lazily-initialized singletons (imports are heavy / optional)
_markitdown = None

# PostgreSQL TEXT columns reject NUL (0x00); some PDFs/exports leak control bytes.
# Strip all C0 control chars except tab/newline/carriage-return.
_BAD_CONTROL_CHARS = {c: None for c in range(0x20) if c not in (0x09, 0x0A, 0x0D)}


def _sanitize(text: str) -> str:
    """Remove NUL and other disallowed control bytes so Postgres can store it."""
    return text.translate(_BAD_CONTROL_CHARS)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def _get_markitdown():
    global _markitdown
    if _markitdown is None:
        from markitdown import MarkItDown  # imported lazily

        _markitdown = MarkItDown()
    return _markitdown


def _convert_passthrough(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        # Fall back to a lenient read for stray encodings
        return path.read_text(encoding="utf-8", errors="replace").strip()


def _convert_markitdown(path: Path) -> str | None:
    md = _get_markitdown()
    result = md.convert(str(path))
    text = (result.text_content or "").strip()
    return text or None


def _convert_pdf(path: Path) -> str | None:
    """Extract PDF text with MarkItDown, then pdfium for malformed font structures."""
    try:
        return _convert_markitdown(path)
    except Exception as primary_error:  # noqa: BLE001 - pdfium handles pdfminer failures
        logger.warning(f"Primary PDF parser failed for {path.name}; trying pdfium: {primary_error}")

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    pages: list[str] = []
    try:
        for page in document:
            text_page = page.get_textpage()
            try:
                text = (text_page.get_text_range() or "").strip()
                if text:
                    pages.append(text)
            finally:
                close = getattr(text_page, "close", None)
                if close:
                    close()
    finally:
        close = getattr(document, "close", None)
        if close:
            close()
    return "\n\n---\n\n".join(pages) or None


def _table_card(df, title: str, sheet: str | None = None) -> str:
    """Build a deterministic, high-signal description of a dataframe."""
    rows, cols = df.shape
    header = f"# {title}" if sheet is None else f"# {title} — sheet: {sheet}"

    col_lines = []
    for name in df.columns:
        dtype = str(df[name].dtype)
        series = df[name].dropna()
        hint = ""
        if len(series):
            if dtype.startswith(("int", "float")):
                try:
                    hint = f" (min={series.min()}, max={series.max()})"
                except TypeError:
                    hint = ""
            else:
                top = series.astype(str).value_counts().head(3).index.tolist()
                hint = f" (e.g. {', '.join(top)})"
        col_lines.append(f"- {name}: {dtype}{hint}")

    parts = [
        header,
        f"\nSpreadsheet with {rows} rows and {cols} columns.",
        "\nColumns:",
        "\n".join(col_lines),
    ]

    # Sample rows as a small markdown table (always cheap, high-signal)
    try:
        sample_md = df.head(SAMPLE_ROWS).to_markdown(index=False)
        if sample_md:
            parts.append(f"\nSample rows:\n{sample_md}")
    except Exception:  # noqa: BLE001 - to_markdown needs tabulate; degrade gracefully
        parts.append(f"\nSample rows:\n{df.head(SAMPLE_ROWS).to_string(index=False)}")

    card = "\n".join(parts)

    # Small labeled table: append the full content so cells are retrievable too.
    try:
        full_md = df.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        full_md = df.to_string(index=False)
    if full_md and len(full_md) <= SMALL_TABLE_CHAR_LIMIT:
        card += f"\n\nFull table:\n{full_md}"
    else:
        card += "\n\n(Large table — full data not embedded; summarized above.)"

    return card.strip()


def _convert_tabular(path: Path) -> str | None:
    import pandas as pd

    title = path.stem
    ext = path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)
        return _table_card(df, title)

    # Excel: one card per sheet, concatenated into a single document.
    sheets = pd.read_excel(path, sheet_name=None)
    cards = []
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        cards.append(_table_card(df, title, sheet=sheet_name))
    return "\n\n---\n\n".join(cards) if cards else None


def convert_file(path: Path, *, raise_errors: bool = False) -> str | None:
    """
    Convert a file to embeddable text, or return None to skip it.

    Never raises on a single bad file - logs and returns None so one unreadable
    document can't abort a whole index run.
    """
    ext = path.suffix.lower()
    try:
        if ext in PASSTHROUGH_EXTS:
            text = _convert_passthrough(path)
        elif ext == ".pdf":
            text = _convert_pdf(path)
        elif ext in MARKITDOWN_EXTS:
            text = _convert_markitdown(path)
        elif ext in TABULAR_EXTS:
            text = _convert_tabular(path)
        else:
            return None
        return _sanitize(text) if text else None
    except Exception as e:  # noqa: BLE001 - resilience: skip bad files, keep going
        if raise_errors:
            raise
        logger.warning(f"Failed to convert {path.name}: {e}")
        return None
