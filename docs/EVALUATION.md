# EVALUATION.md

## Mục Đích

Định nghĩa quy trình đánh giá có thể lặp lại cho:

- `standard` mode
- `advanced` mode
- `compare` mode

## Định Dạng Dataset

Đường dẫn golden dataset:

- `data/eval/golden_dataset.jsonl`
- `data/eval/golden.jsonl` (alias cũ vẫn hợp lệ để tương thích CLI)

Mỗi dòng JSONL gồm:

- `id` (bắt buộc)
- `question` (bắt buộc)
- `expected_behavior` (bắt buộc): `answer`, `abstain`, hoặc `retry`
- `reference_answer` (tùy chọn)
- `gold_sources` (tùy chọn)
  - hỗ trợ chuỗi legacy như `docs/MODES.md`
  - hỗ trợ selector có key để match ổn định hơn:
    - `chunk_id=...`
    - `doc_id=...`
    - `source=...` hoặc `path=...`
    - `title=...`, `section=...`
- `category` (bắt buộc): `simple`, `multi_hop`, `ambiguous`, `insufficient_context`, `conflicting_sources`, `vietnamese`
- `notes` (tùy chọn)

## Nhóm Bao Phủ Bắt Buộc

- simple
- multi_hop
- ambiguous
- insufficient_context
- conflicting_sources
- vietnamese

## Runner

Module CLI:

- `python scripts/run_eval.py`

Tùy chọn:

- `--dataset`: đường dẫn dataset (mặc định `data/eval/golden_dataset.jsonl`)
- `--modes standard advanced compare`
- `--output-dir data/eval/results`
- `--predictor workflow|stub` (`stub` chạy theo logic xác định, không cần workflow thực tế)

Ví dụ:

```bash
python scripts/run_eval.py --dataset data/eval/golden_dataset.jsonl --modes standard advanced compare
python -m app.evaluation.runner --dataset data/eval/golden.jsonl --predictor stub
```

## Regression Check Hiện Tại

Test tự động đang kiểm tra:

- nạp dataset và kiểm tra hình dạng dữ liệu
- tính toán metric
- sinh báo cáo
- chạy evaluation runner với workflow đã mock

## Các Metric Được Báo Cáo

- số lượng citation và tỉ lệ có citation
- độ khớp abstain và tỉ lệ abstain
- mức sử dụng retry và tỉ lệ retry ở advanced
- tổng hợp confidence và latency
- số lượng context retrieved/selected
- các chỉ báo heuristic:
  - câu trả lời không rỗng
  - độ trùng từ khóa với `reference_answer`
  - độ trùng với `gold_sources`
  - groundedness proxy dựa trên mức chồng lấp từ vựng với selected context
