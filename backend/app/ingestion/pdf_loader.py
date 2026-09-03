"""pdf loading: extract text per page with page-number provenance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Page:
    page_number: int  # 1-indexed
    text: str


def load_pdf(path: str) -> List[Page]:
    """Return a list of 1-indexed pages with extracted text.

    Uses ``pypdf``. Image-only (scanned) PDFs will yield empty text and should be
    OCR'd first - we surface that in the README rather than silently ingesting
    nothing.
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages: List[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append(Page(page_number=i, text=page.extract_text() or ""))
    return pages
