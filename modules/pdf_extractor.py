"""
PDF Text Extraction Module
Extracts text from digital PDFs using pdfplumber.
Detects scanned pages that need OCR.
"""

import pdfplumber
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class PageData:
    page_number: int
    text: str
    word_count: int
    char_count: int
    is_scanned: bool
    tables: list = field(default_factory=list)


@dataclass
class DocumentData:
    pages: List[PageData]
    metadata: Dict
    full_text: str
    num_pages: int
    total_words: int
    total_chars: int
    scanned_pages: List[int]
    file_name: str


class PDFExtractor:
    """Extracts text from digital/normal PDF files."""

    SCANNED_THRESHOLD = 15  # min words per page to NOT be "scanned"

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self._pages: List[PageData] = []
        self._metadata: Dict = {}
        self._extracted = False

    # ------------------------------------------------------------------
    def extract(self) -> DocumentData:
        """Main extraction entry-point. Returns a DocumentData object."""
        try:
            with pdfplumber.open(self.file_path) as pdf:
                self._metadata = {
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "subject": pdf.metadata.get("Subject", ""),
                    "creator": pdf.metadata.get("Creator", ""),
                    "producer": pdf.metadata.get("Producer", ""),
                    "creation_date": str(pdf.metadata.get("CreationDate", "")),
                    "num_pages": len(pdf.pages),
                }

                for idx, page in enumerate(pdf.pages):
                    raw_text = page.extract_text() or ""
                    raw_text = raw_text.strip()

                    # Try to extract tables
                    tables = []
                    try:
                        page_tables = page.extract_tables() or []
                        for tbl in page_tables:
                            tables.append(tbl)
                    except Exception:
                        pass

                    word_count = len(raw_text.split()) if raw_text else 0
                    char_count = len(raw_text)

                    self._pages.append(
                        PageData(
                            page_number=idx + 1,
                            text=raw_text,
                            word_count=word_count,
                            char_count=char_count,
                            is_scanned=word_count < self.SCANNED_THRESHOLD,
                            tables=tables,
                        )
                    )

            self._extracted = True
            full_text = "\n\n".join(p.text for p in self._pages if p.text)
            scanned = [p.page_number for p in self._pages if p.is_scanned]

            return DocumentData(
                pages=self._pages,
                metadata=self._metadata,
                full_text=full_text,
                num_pages=len(self._pages),
                total_words=sum(p.word_count for p in self._pages),
                total_chars=sum(p.char_count for p in self._pages),
                scanned_pages=scanned,
                file_name=self.file_name,
            )

        except Exception as e:
            raise RuntimeError(f"PDF extraction failed: {e}")

    # ------------------------------------------------------------------
    def has_scanned_pages(self) -> bool:
        if not self._extracted:
            self.extract()
        return any(p.is_scanned for p in self._pages)

    def get_page_text(self, page_number: int) -> str:
        if not self._extracted:
            self.extract()
        for p in self._pages:
            if p.page_number == page_number:
                return p.text
        return ""

    def get_pages_text_dict(self) -> Dict[int, str]:
        if not self._extracted:
            self.extract()
        return {p.page_number: p.text for p in self._pages}