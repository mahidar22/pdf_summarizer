"""
OCR Engine Module
Handles scanned PDFs and images using Tesseract OCR.
Falls back to basic extraction if Tesseract is unavailable.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageFilter, ImageEnhance

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_path

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


class OCREngine:
    """Optical Character Recognition for scanned PDFs and images."""

    SUPPORTED_LANGS = {
        "eng": "English",
        "fra": "French",
        "deu": "German",
        "spa": "Spanish",
        "ita": "Italian",
        "por": "Portuguese",
        "hin": "Hindi",
        "ara": "Arabic",
        "chi_sim": "Chinese (Simplified)",
        "jpn": "Japanese",
        "kor": "Korean",
    }

    def __init__(self, tesseract_path: Optional[str] = None, lang: str = "eng"):
        self.lang = lang
        if tesseract_path and TESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self._check_availability()

    # ------------------------------------------------------------------
    def _check_availability(self):
        if not TESSERACT_AVAILABLE:
            raise ImportError(
                "pytesseract is not installed. "
                "Run: pip install pytesseract  and install Tesseract binary."
            )
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError(
                "pdf2image is not installed. "
                "Run: pip install pdf2image  and install poppler-utils."
            )

    # ------------------------------------------------------------------
    @staticmethod
    def preprocess_image(image: Image.Image) -> Image.Image:
        """Enhance image for better OCR accuracy."""
        # Convert to grayscale
        img = image.convert("L")
        # Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        # Sharpen
        img = img.filter(ImageFilter.SHARPEN)
        # Binarize (threshold)
        img = img.point(lambda x: 0 if x < 140 else 255, "1")
        return img

    # ------------------------------------------------------------------
    def extract_from_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        dpi: int = 300,
        preprocess: bool = True,
    ) -> Dict[int, str]:
        """
        OCR specific pages of a PDF.
        Returns {page_number: extracted_text}.
        """
        results: Dict[int, str] = {}

        try:
            if pages:
                # pdf2image uses 1-based indexing
                images = convert_from_path(
                    pdf_path, dpi=dpi, first_page=min(pages), last_page=max(pages)
                )
                page_range = range(min(pages), max(pages) + 1)
            else:
                images = convert_from_path(pdf_path, dpi=dpi)
                page_range = range(1, len(images) + 1)

            for img, page_num in zip(images, page_range):
                if pages and page_num not in pages:
                    continue

                if preprocess:
                    img = self.preprocess_image(img)

                text = pytesseract.image_to_string(img, lang=self.lang)
                results[page_num] = text.strip()

        except Exception as e:
            raise RuntimeError(f"OCR extraction failed: {e}")

        return results

    # ------------------------------------------------------------------
    def extract_from_image(
        self, image_path: str, preprocess: bool = True
    ) -> str:
        """OCR a single image file."""
        try:
            img = Image.open(image_path)
            if preprocess:
                img = self.preprocess_image(img)
            text = pytesseract.image_to_string(img, lang=self.lang)
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Image OCR failed: {e}")

    # ------------------------------------------------------------------
    def extract_from_pil_image(
        self, image: Image.Image, preprocess: bool = True
    ) -> str:
        """OCR a PIL Image object."""
        if preprocess:
            image = self.preprocess_image(image)
        text = pytesseract.image_to_string(image, lang=self.lang)
        return text.strip()

    # ------------------------------------------------------------------
    def ocr_full_pdf(self, pdf_path: str, dpi: int = 300) -> Tuple[str, Dict[int, str]]:
        """OCR every page. Returns (full_text, page_dict)."""
        page_dict = self.extract_from_pdf(pdf_path, pages=None, dpi=dpi)
        full_text = "\n\n".join(
            page_dict[k] for k in sorted(page_dict.keys())
        )
        return full_text, page_dict