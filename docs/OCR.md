# OCR (Tùy Chọn) Cho PDF Scan

OCR trong dự án này được thiết kế theo hướng:

- tùy chọn (disabled by default)
- không phá vỡ parser architecture hiện có
- fail-safe nếu thiếu Tesseract hoặc OCR lỗi

## Cấu Hình

Thêm vào `.env`:

```bash
OCR_ENABLED=false
OCR_LANGUAGE=vie+eng
OCR_MIN_TEXT_CHARS=100
OCR_RENDER_DPI=216
TESSERACT_CMD=
OCR_CONFIDENCE_THRESHOLD=40
```

Ý nghĩa nhanh:

- `OCR_ENABLED`: bật/tắt OCR fallback cho PDF
- `OCR_LANGUAGE`: ngôn ngữ OCR (khuyến nghị `vie+eng`)
- `OCR_MIN_TEXT_CHARS`: nếu text trích xuất từ PDF page ngắn hơn ngưỡng này thì thử OCR
- `OCR_RENDER_DPI`: DPI render ảnh trang PDF trước OCR
- `TESSERACT_CMD`: đường dẫn nhị phân Tesseract (đặc biệt hữu ích trên Windows)
- `OCR_CONFIDENCE_THRESHOLD`: ngưỡng confidence để lọc từ từ `image_to_data` (heuristic)

## Cài Đặt Tesseract

Ubuntu:

```bash
sudo apt install tesseract-ocr tesseract-ocr-vie
```

macOS:

```bash
brew install tesseract tesseract-lang
```

Windows:

1. Cài Tesseract OCR.
2. Cấu hình `TESSERACT_CMD` tới `tesseract.exe` trong `.env`.

## Hành Vi Runtime

PDF parser vẫn chạy luồng bình thường:

1. extract text/table/image bằng `pdfplumber`
2. nếu text page quá ngắn và OCR bật:
   - render page bằng `PyMuPDF`
   - OCR bằng `pytesseract`
   - thêm block text metadata OCR (`block_type=ocr_text`, `ocr=true`)

Nếu OCR fail, parser ghi warning và tiếp tục xử lý tài liệu.
Sau khi thay đổi `OCR_ENABLED` hoặc tham số OCR, cần upload lại PDF để tái tạo chunks/index theo cấu hình mới.

## Giới Hạn Hiện Tại

- OCR làm ingestion chậm hơn (đặc biệt PDF nhiều trang).
- Chất lượng phụ thuộc scan quality, DPI, font, độ nhiễu.
- OCR line/table formatting là heuristic (group theo tọa độ), không thay thế table parser chuyên dụng.
- DOCX image OCR chưa bật trong phase này.
