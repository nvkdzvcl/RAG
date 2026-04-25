"""Optional OCR helpers for scanned/image-based PDF content."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import fitz
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    fitz = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageFilter, ImageOps
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    Image = Any  # type: ignore[assignment,misc]
    ImageFilter = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

try:
    import pytesseract
    from pytesseract import Output
    from pytesseract import TesseractNotFoundError
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    pytesseract = None  # type: ignore[assignment]
    Output = None  # type: ignore[assignment]

    class TesseractNotFoundError(RuntimeError):
        """Fallback exception when pytesseract is unavailable."""


def configure_tesseract_cmd(cmd: str | None) -> None:
    """Set explicit tesseract binary path when provided."""
    if not cmd or pytesseract is None:
        return
    pytesseract.pytesseract.tesseract_cmd = cmd.strip()


def is_tesseract_available() -> bool:
    """Return whether pytesseract and the Tesseract binary are usable."""
    if pytesseract is None:
        return False
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except (OSError, RuntimeError, TesseractNotFoundError):
        return False


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Apply lightweight OCR-oriented preprocessing."""
    if ImageOps is None or ImageFilter is None:
        raise RuntimeError("OCR preprocessing requires Pillow to be installed.")

    # Heuristic-only preprocessing:
    # 1) grayscale, 2) autocontrast for scan background cleanup, 3) sharpen text edges.
    grayscale = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(grayscale)
    return contrasted.filter(ImageFilter.SHARPEN)


def _parse_confidence(raw_value: Any) -> float | None:
    try:
        parsed = float(str(raw_value).strip())
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _line_from_words(words: list[dict[str, Any]], *, image_width: int, x_gap_ratio: float) -> str:
    if not words:
        return ""
    ordered = sorted(words, key=lambda item: item["left"])
    gap_threshold = max(24, int(image_width * x_gap_ratio))

    pieces: list[str] = [ordered[0]["text"]]
    prev_right = ordered[0]["left"] + ordered[0]["width"]

    for word in ordered[1:]:
        gap = word["left"] - prev_right
        if gap > gap_threshold:
            pieces.append(" | ")
        elif gap > max(8, gap_threshold // 3):
            pieces.append("  ")
        else:
            pieces.append(" ")
        pieces.append(word["text"])
        prev_right = word["left"] + word["width"]

    return "".join(pieces).strip()


def _extract_text_from_data_output(
    data: dict[str, list[Any]],
    *,
    image_width: int,
    confidence_threshold: float,
    x_gap_ratio: float,
) -> str:
    texts = data.get("text", [])
    if not texts:
        return ""

    confidences = data.get("conf", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])

    words: list[dict[str, Any]] = []
    for idx, text_value in enumerate(texts):
        text = str(text_value).strip()
        if not text:
            continue

        conf = _parse_confidence(confidences[idx] if idx < len(confidences) else None)
        if conf is None or conf < confidence_threshold:
            continue

        try:
            left = int(float(str(lefts[idx] if idx < len(lefts) else 0)))
            top = int(float(str(tops[idx] if idx < len(tops) else 0)))
            width = int(float(str(widths[idx] if idx < len(widths) else 0)))
            height = int(float(str(heights[idx] if idx < len(heights) else 0)))
        except (TypeError, ValueError):
            continue

        words.append(
            {
                "text": text,
                "conf": conf,
                "left": left,
                "top": top,
                "width": max(width, 1),
                "height": max(height, 1),
            }
        )

    if not words:
        return ""

    # Group words into visual lines by Y position (heuristic).
    median_height = int(mean(item["height"] for item in words))
    line_tolerance = max(8, median_height // 2)

    line_buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sorted_words = sorted(words, key=lambda item: (item["top"], item["left"]))

    for word in sorted_words:
        assigned_key: int | None = None
        for key in list(line_buckets.keys()):
            if abs(word["top"] - key) <= line_tolerance:
                assigned_key = key
                break
        if assigned_key is None:
            assigned_key = word["top"]
        line_buckets[assigned_key].append(word)

    lines = [
        _line_from_words(line_words, image_width=image_width, x_gap_ratio=x_gap_ratio)
        for _, line_words in sorted(line_buckets.items(), key=lambda item: item[0])
    ]
    return "\n".join(line for line in lines if line).strip()


def ocr_image(
    image: Image.Image,
    lang: str = "vie+eng",
    *,
    confidence_threshold: float = 40.0,
    x_gap_ratio: float = 0.18,
) -> str:
    """Run OCR on a PIL image and return extracted text."""
    if pytesseract is None or Output is None:
        raise RuntimeError("OCR requires pytesseract to be installed.")

    preprocessed = preprocess_image_for_ocr(image)
    data = pytesseract.image_to_data(preprocessed, lang=lang, output_type=Output.DICT)
    text = _extract_text_from_data_output(
        data,
        image_width=preprocessed.width,
        confidence_threshold=confidence_threshold,
        x_gap_ratio=x_gap_ratio,
    )
    if text:
        return text

    fallback_text = pytesseract.image_to_string(preprocessed, lang=lang)
    return (fallback_text or "").strip()


def ocr_pdf_page_with_pymupdf(
    file_path: str | Path,
    page_index: int,
    dpi: int = 216,
    *,
    lang: str = "vie+eng",
    confidence_threshold: float = 40.0,
) -> str:
    """Render a PDF page with PyMuPDF and OCR it."""
    if fitz is None:
        raise RuntimeError("PDF OCR rendering requires the 'pymupdf' package.")
    if ImageOps is None:
        raise RuntimeError("PDF OCR rendering requires Pillow to be installed.")

    resolved_path = Path(file_path)
    zoom = max(float(dpi), 72.0) / 72.0
    transform = fitz.Matrix(zoom, zoom)

    document = fitz.open(str(resolved_path))
    try:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=transform, alpha=False)
    finally:
        document.close()

    image_mode = "RGB" if pixmap.n >= 3 else "L"
    image = Image.frombytes(image_mode, [pixmap.width, pixmap.height], pixmap.samples)
    return ocr_image(
        image,
        lang=lang,
        confidence_threshold=confidence_threshold,
    )
