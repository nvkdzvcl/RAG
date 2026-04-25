# MODES.md

## Tổng Quan

Dự án hỗ trợ 3 chế độ truy vấn:

- **standard**
- **advanced**
- **compare**

Thiết kế tách phần hạ tầng dùng chung để tái sử dụng, đồng thời cô lập logic điều phối theo từng workflow.

## Chế Độ Standard

Pipeline:

`query -> retrieve -> rerank -> select context -> generate -> cite -> return`

Đặc điểm:

- luôn dựa trên retrieval
- không có vòng critique lặp
- độ trễ/chi phí thấp hơn, phù hợp làm baseline

## Chế Độ Advanced

Pipeline:

`query -> retrieval gate -> query rewrite -> retrieve -> rerank -> generate draft -> critique -> retry/refine/abstain -> return`

Đặc điểm:

- có vòng tự kiểm tra nhưng bị chặn bởi giới hạn số vòng
- có thể retry retrieval
- có thể abstain khi bằng chứng yếu
- độ trễ/chi phí cao hơn để đổi lấy độ tin cậy

## Chế Độ Compare

Pipeline:

`query -> run standard + advanced -> aggregate outputs -> compute summary -> return`

Đặc điểm:

- chạy standard và advanced độc lập cho cùng truy vấn
- trả về kết quả song song để đối chiếu trực tiếp
- có phần tóm tắt so sánh (ví dụ chênh lệch confidence/latency)

## Thành Phần Dùng Chung

Cả 3 mode dùng chung:

- loader/chunking/indexing
- retriever/reranker/context selector
- hợp đồng generation và citation
- configuration/logging/schema
- định dạng dataset đánh giá

Điểm khác biệt chính chỉ nằm ở lớp orchestration (cách điều phối workflow).
