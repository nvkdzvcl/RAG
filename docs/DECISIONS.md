# DECISIONS.md

Tài liệu này ghi lại các quyết định kiến trúc và kỹ thuật quan trọng.

Mỗi quyết định bao gồm:
- bối cảnh
- quyết định
- các phương án đã cân nhắc
- lý do lựa chọn
- hệ quả

Mục tiêu là lưu giữ lập luận kỹ thuật của dự án để người đóng góp sau này không phải đoán vì sao các lựa chọn hiện tại được đưa ra.

---

## QUYẾT ĐỊNH 001 — Xây Dựng 3 Chế Độ Trong Một Hệ Thống

### Bối cảnh
Dự án cần luồng RAG baseline, luồng Self-RAG nâng cao, và luồng so sánh trực tiếp.

### Quyết định
Xây dựng một hệ thống dùng chung với 3 workflow:
- standard mode
- advanced mode
- compare mode

### Các phương án đã cân nhắc
- chỉ xây advanced mode
- tách codebase riêng cho từng mode
- làm standard trước rồi thiết kế lại sau

### Lý do lựa chọn
Kiến trúc dùng chung giúp giảm trùng lặp và làm cho việc so sánh trở nên trực tiếp hơn.

### Hệ quả
Ưu điểm:
- benchmark dễ hơn
- demo rõ ràng hơn
- giảm code trùng lặp

Nhược điểm:
- yêu cầu kỷ luật kiến trúc tốt ngay từ đầu

---

## QUYẾT ĐỊNH 002 — Dùng Self-RAG Thực Dụng Thay Vì Tái Hiện Mức Paper

### Bối cảnh
Paper Self-RAG gốc dùng huấn luyện ở mức mô hình và hành vi phản tư (reflection-style).

### Quyết định
Triển khai Self-RAG ở mức hệ thống với các bước:
- retrieval gate
- critique
- retry
- refine
- abstain

### Các phương án đã cân nhắc
- tái hiện Self-RAG đầy đủ theo paper
- chỉ dùng standard RAG

### Lý do lựa chọn
Đây là hướng thực tế hơn cho một dự án phần mềm học thuật mã nguồn mở.

### Hệ quả
Ưu điểm:
- khả thi để triển khai
- dễ debug
- dễ trình bày

Nhược điểm:
- không phải bản tái hiện paper đầy đủ

---

## QUYẾT ĐỊNH 003 — Dùng Hybrid Retrieval

### Bối cảnh
Dense-only có thể bỏ sót truy vấn nặng từ khóa hoặc cần khớp chính xác.

### Quyết định
Dùng hybrid retrieval:
- dense retrieval
- BM25 retrieval
- fusion

### Các phương án đã cân nhắc
- chỉ dense retrieval
- chỉ sparse retrieval

### Lý do lựa chọn
Hybrid retrieval cải thiện recall cho nhiều dạng câu hỏi khác nhau.

### Hệ quả
Ưu điểm:
- độ bền vững truy hồi tốt hơn
- hỗ trợ tốt hơn cho thuật ngữ kỹ thuật/chuyên ngành

Nhược điểm:
- tăng số thành phần cần bảo trì

---

## QUYẾT ĐỊNH 004 — Bổ Sung Reranker

### Bối cảnh
Kết quả retrieval ban đầu có thể chứa chunk nhiễu hoặc liên quan yếu.

### Quyết định
Áp dụng reranking sau retrieval.

### Các phương án đã cân nhắc
- không dùng reranker
- chỉ sắp xếp theo điểm cơ bản

### Lý do lựa chọn
Context được xếp hạng tốt hơn sẽ cải thiện chất lượng generation và giảm nhiễu.

### Hệ quả
Ưu điểm:
- chất lượng context tốt hơn
- câu trả lời downstream tốt hơn

Nhược điểm:
- tăng độ trễ
- thêm phụ thuộc runtime

---

## QUYẾT ĐỊNH 005 — Giới Hạn Context Khi Generation

### Bối cảnh
Truyền quá nhiều chunk có thể làm giảm chất lượng trả lời và tăng chi phí.

### Quyết định
Chỉ dùng top chunk đã chọn sau reranking cho generation.

### Các phương án đã cân nhắc
- truyền toàn bộ chunk retrieve được
- context động không giới hạn

### Lý do lựa chọn
Context tập trung thường cho câu trả lời grounded tốt hơn.

### Hệ quả
Ưu điểm:
- giảm chi phí token
- câu trả lời gọn hơn

Nhược điểm:
- có nguy cơ bỏ sót bằng chứng hữu ích nếu chọn context chưa tốt

---

## QUYẾT ĐỊNH 006 — Tách Workflow Standard và Advanced

### Bối cảnh
Standard và advanced dùng chung nhiều thành phần nhưng khác nhau ở luồng điều khiển.

### Quyết định
Tách module workflow:
- standard.py
- advanced.py
- shared.py
- router.py

### Các phương án đã cân nhắc
- một file workflow lớn với nhiều cờ
- tách pipeline riêng nhưng trùng lặp

### Lý do lựa chọn
Cách này giúp orchestration dễ đọc và có tính module.

### Hệ quả
Ưu điểm:
- dễ bảo trì
- dễ debug
- dễ test

Nhược điểm:
- cần đầu tư cấu trúc ban đầu nhiều hơn một chút

---

## QUYẾT ĐỊNH 007 — Thêm Retrieval Gate Ở Advanced Mode

### Bối cảnh
Không phải truy vấn nào cũng thực sự cần retrieval.

### Quyết định
Thêm bước retrieval-gate trước retrieval ở advanced mode.

### Các phương án đã cân nhắc
- luôn retrieve
- quyết định retrieve chỉ bằng rule

### Lý do lựa chọn
Phù hợp hơn với Self-RAG thực dụng và giảm công việc không cần thiết.

### Hệ quả
Ưu điểm:
- có thể giảm chi phí trong một số trường hợp
- hành vi hệ thống linh hoạt hơn

Nhược điểm:
- có rủi ro quyết định sai ở nhánh không-retrieval

---

## QUYẾT ĐỊNH 008 — Dùng Critique Output Có Cấu Trúc

### Bối cảnh
Critique dạng văn bản tự do khó parse và khó debug.

### Quyết định
Yêu cầu critique tuân theo schema có cấu trúc.

### Các phương án đã cân nhắc
- critique tự do
- critique chỉ theo rule

### Lý do lựa chọn
Output có cấu trúc dễ validate và dễ tiêu thụ ở mức lập trình.

### Hệ quả
Ưu điểm:
- logic workflow đáng tin cậy hơn
- debug dễ hơn
- log rõ ràng hơn

Nhược điểm:
- cần thiết kế prompt cẩn thận

---

## QUYẾT ĐỊNH 009 — Giới Hạn Vòng Lặp Retry Ở Advanced

### Bối cảnh
Vòng lặp không giới hạn gây tăng chi phí và độ trễ khó dự đoán.

### Quyết định
Đặt số vòng retry tối đa mặc định là 2.

### Các phương án đã cân nhắc
- retry không giới hạn
- không retry

### Lý do lựa chọn
Phần lớn lợi ích đến từ 1-2 lần retry, không phải lặp vô hạn.

### Hệ quả
Ưu điểm:
- độ phức tạp có giới hạn
- thời gian chạy dễ dự đoán

Nhược điểm:
- một số ca khó có thể vẫn chưa giải quyết triệt để

---

## QUYẾT ĐỊNH 010 — Cho Phép Abstain

### Bối cảnh
Hệ thống không nên hallucinate khi bằng chứng không đủ.

### Quyết định
Cho phép hệ thống abstain hoặc trả về phản hồi "thiếu bằng chứng" rõ ràng.

### Các phương án đã cân nhắc
- luôn trả lời
- đoán ngầm không thông báo

### Lý do lựa chọn
Độ tin cậy quan trọng hơn sự đầy đủ cưỡng ép.

### Hệ quả
Ưu điểm:
- độ tin cậy cao hơn
- tăng niềm tin người dùng

Nhược điểm:
- một số người dùng có thể thích câu trả lời suy đoán hơn

---

## QUYẾT ĐỊNH 011 — Dùng Schema State Dùng Chung Giữa Các Mode

### Bối cảnh
Tất cả workflow cần cách quản lý state nhất quán, nhất là cho logging và trace ở frontend.

### Quyết định
Dùng một workflow state schema chung; một số field có thể không dùng ở standard mode.

### Các phương án đã cân nhắc
- model state tách biệt hoàn toàn
- dictionary không typed

### Lý do lựa chọn
State dùng chung tăng tính nhất quán và giảm độ phức tạp serialization.

### Hệ quả
Ưu điểm:
- tích hợp API đơn giản hơn
- logging đơn giản hơn
- render trace đơn giản hơn

Nhược điểm:
- standard mode có thể mang vài field không dùng

---

## QUYẾT ĐỊNH 012 — Lưu Prompt Thành File Riêng

### Bối cảnh
Prompt nhúng trực tiếp trong code khó bảo trì.

### Quyết định
Lưu prompt trong thư mục `/prompts`.

### Các phương án đã cân nhắc
- prompt inline trong code
- hằng số prompt ẩn trong các module

### Lý do lựa chọn
Prompt file riêng dễ versioning, so sánh và cải tiến.

### Hệ quả
Ưu điểm:
- code gọn hơn
- vòng lặp chỉnh prompt dễ hơn
- onboarding cộng tác viên dễ hơn

Nhược điểm:
- thêm một thư mục cần bảo trì

---

## QUYẾT ĐỊNH 013 — Làm Frontend Sớm Nhưng Ở Mức Vừa Phải

### Bối cảnh
Dự án cần giao diện để demo, nhưng frontend không được lấn át backend.

### Quyết định
Xây frontend đủ hoàn thiện sau khi backend baseline ổn định.

### Các phương án đã cân nhắc
- không làm frontend
- làm frontend nặng ngay từ đầu

### Lý do lựa chọn
Frontend mức vừa phải phục vụ demo và khả năng dùng mà không làm lệch ưu tiên backend.

### Hệ quả
Ưu điểm:
- trình bày tốt hơn
- đánh giá và demo dễ hơn

Nhược điểm:
- cần phối hợp thêm giữa frontend và backend

---

## QUYẾT ĐỊNH 014 — Dùng React + Vite + Tailwind + shadcn/ui

### Bối cảnh
Dự án cần frontend đẹp, thực dụng và phát triển nhanh.

### Quyết định
Dùng stack:
- React
- Vite
- Tailwind CSS
- shadcn/ui
- lucide-react
- recharts

### Các phương án đã cân nhắc
- React thuần không design system
- Next.js
- các UI framework nặng hơn

### Lý do lựa chọn
Stack này có tốc độ phát triển nhanh, chất lượng giao diện tốt và phù hợp với dự án mã nguồn mở.

### Hệ quả
Ưu điểm:
- làm UI đẹp nhanh
- trải nghiệm dev hiện đại
- linh hoạt thành phần

Nhược điểm:
- có overhead thiết lập frontend

---

## QUYẾT ĐỊNH 015 — Giữ Cài Đặt Đơn Giản Với requirements.txt

### Bối cảnh
Người đóng góp mã nguồn mở cần cài phụ thuộc backend nhanh và rõ ràng.

### Quyết định
Cung cấp:
- requirements.txt
- requirements-dev.txt
- .env.example

Không giữ shim `requirement.txt` riêng để tránh gây mơ hồ.

### Các phương án đã cân nhắc
- chỉ dùng pyproject
- cài đặt thủ công không tài liệu hóa

### Lý do lựa chọn
Giảm ma sát setup cho contributor và người đánh giá.

### Hệ quả
Ưu điểm:
- onboarding dễ hơn
- setup local dễ hơn

Nhược điểm:
- cần bảo trì phụ thuộc cẩn thận

---

## QUYẾT ĐỊNH 016 — Ưu Tiên Tính Dễ Đọc Hơn Tối Ưu Quá Sớm

### Bối cảnh
Đây là dự án phần mềm học thuật và tài liệu tham chiếu mã nguồn mở.

### Quyết định
Ưu tiên code module, dễ đọc hơn là tối ưu mạnh ở phiên bản đầu.

### Các phương án đã cân nhắc
- tối ưu sớm
- gộp nhiều mối quan tâm vào ít file

### Lý do lựa chọn
Khả năng bảo trì và tính rõ ràng là nền tảng cho học tập và cộng tác.

### Hệ quả
Ưu điểm:
- contributor dễ hiểu hệ thống
- thuận lợi cho chấm điểm và demo

Nhược điểm:
- một số tối ưu hiệu năng có thể được dời sang giai đoạn sau

---

## QUYẾT ĐỊNH 017 — Thêm Compare Mode

### Bối cảnh
Dự án cần cách trực quan để so sánh baseline RAG và Advanced Self-RAG.

### Quyết định
Thêm compare mode chạy standard và advanced song song cho cùng một truy vấn.

### Các phương án đã cân nhắc
- so sánh thủ công ngoài hệ thống
- chỉ giữ standard và advanced

### Lý do lựa chọn
Nâng chất lượng demo, tăng khả năng quan sát khi evaluation và gia tăng giá trị học thuật.

### Hệ quả
Ưu điểm:
- benchmark dễ hơn
- demo tốt hơn
- giải thích trade-off rõ hơn

Nhược điểm:
- schema response phức tạp hơn
- frontend cũng phức tạp hơn một chút
