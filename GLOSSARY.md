# Giải thích các khái niệm và các giai đoạn trong hệ thống RAG

> Tài liệu này giải thích các khái niệm, công nghệ và luồng hoạt động chính của hệ thống RAG trong repo. Nội dung được viết theo hướng dễ học, dễ thuyết trình, không đi sâu vào tên hàm hoặc chi tiết triển khai code.

---

## 1. RAG là gì?

**RAG** là viết tắt của **Retrieval-Augmented Generation**, nghĩa là **sinh câu trả lời có tăng cường truy xuất tài liệu**.

Một mô hình ngôn ngữ lớn như Qwen, GPT, Llama có thể trả lời câu hỏi rất tự nhiên, nhưng bản thân nó không luôn biết dữ liệu riêng của người dùng, tài liệu nội bộ, file PDF mới upload, hoặc nội dung nằm ngoài dữ liệu huấn luyện. RAG giải quyết vấn đề này bằng cách kết hợp hai phần:

1. **Retrieval**: tìm những đoạn tài liệu liên quan đến câu hỏi.
2. **Generation**: đưa những đoạn tài liệu đó cho LLM để tạo câu trả lời.

Thay vì để LLM trả lời hoàn toàn dựa trên “trí nhớ” của nó, RAG buộc LLM trả lời dựa trên phần ngữ cảnh vừa được truy xuất từ tài liệu.

### RAG dùng để làm gì?

RAG thường dùng cho các hệ thống:

- Hỏi đáp trên PDF, DOCX, Markdown, tài liệu nội bộ.
- Trợ lý học tập trả lời dựa trên giáo trình.
- Chatbot doanh nghiệp trả lời theo quy định công ty.
- Hệ thống phân tích báo cáo, hợp đồng, biên bản, chính sách.
- Tìm kiếm thông minh hơn keyword search thông thường.
- Giảm tình trạng LLM tự bịa câu trả lời.

### Vì sao không đưa toàn bộ tài liệu vào LLM?

Vì tài liệu thường rất dài. LLM có giới hạn độ dài ngữ cảnh, gọi là **context window**. Nếu đưa toàn bộ tài liệu vào, hệ thống sẽ chậm, tốn tài nguyên và dễ bị nhiễu thông tin.

RAG chỉ lấy ra vài đoạn quan trọng nhất rồi đưa vào LLM. Nhờ vậy hệ thống nhanh hơn, rẻ hơn và câu trả lời bám tài liệu hơn.

---

## 2. Tổng quan hệ thống trong repo

Repo này là một hệ thống **Self-RAG Pipeline** có nhiều tầng xử lý. Hệ thống không chỉ làm RAG cơ bản, mà còn có các cơ chế nâng cao như:

- Nhận diện cấu trúc tài liệu khi ingestion.
- Chia tài liệu thành chunk thông minh.
- Tạo embedding cho chunk.
- Tìm kiếm kết hợp dense retrieval và sparse retrieval.
- Dùng BM25 cho tìm kiếm theo từ khóa.
- Dùng vector embedding cho tìm kiếm theo ngữ nghĩa.
- Dùng hybrid retrieval để kết hợp nhiều nguồn điểm.
- Dùng reranker để sắp xếp lại kết quả truy xuất.
- Dùng grounding để kiểm tra câu trả lời có dựa trên tài liệu hay không.
- Dùng chế độ advanced để rewrite, critique, refine hoặc abstain.
- Có evaluation framework để đo chất lượng retrieval và generation.
- Có frontend để thao tác với hệ thống.

Có thể hình dung luồng tổng quát như sau:

```text
Tài liệu upload
    ↓
Ingestion
    ↓
Chunking
    ↓
Embedding + Indexing
    ↓
User đặt câu hỏi
    ↓
Retrieval
    ↓
Reranking
    ↓
Context Selection
    ↓
Generation bằng LLM
    ↓
Grounding / Critique / Refine
    ↓
Trả lời cho người dùng
```

---

## 3. Các giai đoạn chính của RAG

## 3.1. Giai đoạn Ingestion

**Ingestion** là giai đoạn đưa tài liệu vào hệ thống để chuẩn bị cho việc hỏi đáp.

Nói đơn giản: ingestion biến file như PDF, DOCX, TXT, Markdown thành dữ liệu văn bản có cấu trúc để máy có thể tìm kiếm được.

### Ingestion dùng để làm gì?

Giai đoạn này dùng để:

- Đọc nội dung từ tài liệu.
- Tách text, bảng, hình ảnh hoặc metadata.
- Làm sạch nội dung bị lỗi, thừa khoảng trắng, ký tự nhiễu.
- Chia tài liệu thành các đoạn nhỏ hơn gọi là chunk.
- Gắn thông tin ngữ cảnh cho từng chunk.
- Chuẩn bị dữ liệu cho embedding và indexing.

Nếu ingestion làm sai, các bước sau cũng sẽ sai. Ví dụ: nếu PDF bị đọc thiếu nội dung, retrieval không thể tìm được thông tin đúng. Nếu bảng bị cắt đôi, LLM có thể hiểu sai dữ liệu trong bảng.

### Ingestion hoạt động như thế nào?

Thông thường ingestion diễn ra theo các bước:

1. **Nhận file**: người dùng upload tài liệu vào hệ thống.
2. **Xác định định dạng**: hệ thống kiểm tra file là PDF, DOCX, TXT, MD hay Markdown.
3. **Parse nội dung**: hệ thống dùng parser phù hợp để trích xuất nội dung.
4. **Làm sạch dữ liệu**: loại bỏ lỗi định dạng, khoảng trắng thừa, ký tự không cần thiết.
5. **Nhận diện cấu trúc**: giữ lại thông tin như tiêu đề, section, bảng, ảnh hoặc vị trí trang.
6. **Chunking**: chia nội dung thành các đoạn nhỏ.
7. **Gắn metadata**: mỗi chunk giữ thông tin về nguồn, trang, tiêu đề, section, loại nội dung.
8. **Chuyển sang indexing**: chunk được đưa sang bước embedding và lưu chỉ mục.

### Structure-Aware Ingestion là gì?

**Structure-Aware Ingestion** nghĩa là ingestion có nhận thức về cấu trúc tài liệu.

Một hệ thống RAG cơ bản có thể chỉ đọc text rồi cắt đoạn theo số ký tự. Cách này dễ làm mất ý nghĩa của tài liệu. Ví dụ:

- Cắt đôi một bảng.
- Tách tiêu đề khỏi phần nội dung bên dưới.
- Làm mất thông tin trang hoặc section.
- Gom nhiều phần không liên quan vào cùng một chunk.

Structure-aware ingestion cố gắng giữ lại cấu trúc gốc. Ví dụ, nếu một đoạn nằm trong section “Chi phí vận hành”, chunk nên biết nó thuộc section đó. Nếu một bảng đang nằm liền nhau, hệ thống nên giữ bảng như một khối thay vì cắt vụn.

### Chunk là gì?

**Chunk** là một đoạn nhỏ được cắt ra từ tài liệu lớn.

Ví dụ một file PDF 50 trang không nên đưa nguyên vào LLM. Hệ thống sẽ chia nó thành nhiều chunk, mỗi chunk có kích thước vừa đủ để tìm kiếm và đưa vào prompt.

Một chunk tốt cần:

- Đủ ngắn để xử lý nhanh.
- Đủ dài để giữ được ý nghĩa.
- Không cắt ngang một ý quan trọng.
- Có metadata để biết chunk đến từ đâu.

### Chunk size là gì?

**Chunk size** là kích thước mục tiêu của mỗi chunk. Kích thước có thể tính bằng ký tự, từ hoặc token tùy hệ thống.

Chunk size quá nhỏ sẽ làm mất ngữ cảnh. Chunk size quá lớn sẽ làm retrieval kém chính xác và tốn context window.

Ví dụ:

- Chunk nhỏ: dễ tìm chính xác nhưng thiếu bối cảnh.
- Chunk lớn: nhiều bối cảnh hơn nhưng có thể chứa nhiều thông tin nhiễu.

### Chunk overlap là gì?

**Chunk overlap** là phần nội dung được lặp lại giữa hai chunk liên tiếp.

Mục đích của overlap là tránh mất thông tin ở ranh giới giữa hai chunk. Nếu một ý quan trọng nằm ở cuối chunk trước và đầu chunk sau, overlap giúp cả hai chunk vẫn giữ được ngữ cảnh.

Ví dụ:

```text
Chunk 1: A B C D E
Chunk 2: D E F G H
```

Ở đây “D E” là phần overlap.

### Metadata là gì?

**Metadata** là thông tin mô tả dữ liệu, không nhất thiết là nội dung chính.

Trong RAG, metadata của chunk có thể gồm:

- Tên file.
- Trang tài liệu.
- Tiêu đề.
- Section.
- Loại block: text, table, image, OCR text.
- Thời điểm upload.
- ID tài liệu.

Metadata giúp hệ thống truy xuất chính xác hơn, hiển thị nguồn rõ hơn và hỗ trợ lọc kết quả.

---

## 3.2. Giai đoạn OCR

**OCR** là viết tắt của **Optical Character Recognition**, nghĩa là nhận dạng ký tự từ hình ảnh.

Trong nhiều file PDF scan, nội dung không phải là text thật mà là ảnh chụp của trang giấy. Nếu chỉ dùng parser văn bản thông thường, hệ thống có thể không đọc được gì. OCR giúp chuyển chữ trong ảnh thành text.

### OCR dùng để làm gì?

OCR dùng để:

- Đọc PDF scan.
- Đọc ảnh chụp tài liệu.
- Trích xuất chữ từ trang không có text layer.
- Làm cho tài liệu scan có thể search được.

### OCR hoạt động như thế nào?

Quy trình OCR thường là:

1. Render trang PDF thành ảnh.
2. Đưa ảnh vào công cụ OCR.
3. Công cụ OCR nhận diện chữ trong ảnh.
4. Kết quả text được đưa vào pipeline như text bình thường.
5. Gắn metadata để biết đoạn này đến từ OCR.

### Tesseract là gì?

**Tesseract** là một công cụ OCR mã nguồn mở. Nó nhận ảnh đầu vào và cố gắng đọc chữ trong ảnh.

Trong hệ thống này, Tesseract có thể được dùng khi PDF không trích xuất được đủ text bằng parser thông thường.

### pytesseract là gì?

**pytesseract** là thư viện Python dùng để gọi Tesseract từ code Python. Nó đóng vai trò cầu nối giữa pipeline Python và công cụ OCR Tesseract.

### PyMuPDF là gì?

**PyMuPDF** là thư viện Python dùng để xử lý PDF. Trong ngữ cảnh OCR, nó thường được dùng để render trang PDF thành ảnh trước khi đưa ảnh đó vào OCR.

### pdfplumber là gì?

**pdfplumber** là thư viện Python dùng để trích xuất text, bảng và layout từ PDF. Nó hữu ích khi PDF có text layer thật và cần giữ cấu trúc nội dung.

### pypdf là gì?

**pypdf** là thư viện Python dùng để đọc và xử lý PDF. Nó thường dùng cho các thao tác PDF cơ bản như đọc trang, metadata hoặc text.

### Khi nào OCR nên bật?

Nên bật OCR khi tài liệu là:

- PDF scan.
- Ảnh chụp văn bản.
- File có rất ít text trích xuất được.
- Tài liệu tiếng Việt/tiếng Anh cần nhận dạng từ ảnh.

Không nên bật OCR nếu không cần, vì OCR làm ingestion chậm hơn và kết quả phụ thuộc chất lượng ảnh.

---

## 3.3. Giai đoạn Embedding

**Embedding** là quá trình biến văn bản thành vector số.

Máy tính không hiểu trực tiếp câu văn như con người. Để tìm kiếm theo ngữ nghĩa, hệ thống cần biểu diễn câu hoặc đoạn văn dưới dạng một danh sách số. Danh sách số đó gọi là vector embedding.

Ví dụ, hai câu có ý nghĩa gần nhau sẽ có vector gần nhau trong không gian vector:

- “Sinh viên cần nộp học phí khi nào?”
- “Thời hạn đóng học phí là ngày nào?”

Dù dùng từ khác nhau, ý nghĩa gần nhau nên embedding có thể giúp hệ thống nhận ra chúng liên quan.

### Embedding dùng để làm gì?

Embedding dùng để:

- Tìm kiếm theo ý nghĩa, không chỉ theo từ khóa.
- So sánh độ giống nhau giữa câu hỏi và chunk.
- Lưu tài liệu vào vector index.
- Hỗ trợ dense retrieval.

### Sentence Transformers là gì?

**Sentence Transformers** là nhóm mô hình và thư viện dùng để tạo embedding cho câu, đoạn văn hoặc tài liệu.

Trong hệ thống RAG, Sentence Transformers thường nhận một đoạn text và trả về vector. Vector này được dùng để tìm các đoạn tài liệu có ý nghĩa gần với câu hỏi của người dùng.

### multilingual-e5-base là gì?

**multilingual-e5-base** là một mô hình embedding đa ngôn ngữ. “Multilingual” nghĩa là hỗ trợ nhiều ngôn ngữ, phù hợp với tài liệu có tiếng Việt, tiếng Anh hoặc pha trộn cả hai.

Mô hình kiểu E5 thường phân biệt text dùng cho tài liệu và text dùng cho truy vấn. Điều này giúp mô hình hiểu rằng một bên là câu hỏi, một bên là đoạn tài liệu cần so khớp.

### Vector là gì?

**Vector** là một mảng số đại diện cho ý nghĩa của văn bản.

Ví dụ đơn giản:

```text
"học phí sinh viên" → [0.12, -0.34, 0.88, ...]
```

Mảng số này không có ý nghĩa trực tiếp với người đọc, nhưng máy tính có thể dùng nó để tính độ gần nhau giữa các đoạn văn.

### Vector similarity là gì?

**Vector similarity** là độ giống nhau giữa hai vector. Nếu vector của câu hỏi gần vector của một chunk, chunk đó có khả năng liên quan đến câu hỏi.

Một cách phổ biến để đo độ gần là cosine similarity. Kết quả càng cao thì hai đoạn càng giống nhau về mặt ngữ nghĩa.

---

## 3.4. Giai đoạn Indexing

**Indexing** là giai đoạn tạo chỉ mục để truy xuất tài liệu nhanh hơn.

Nếu không có index, mỗi lần người dùng hỏi, hệ thống phải quét toàn bộ tài liệu từ đầu đến cuối. Cách này rất chậm. Index giúp hệ thống tìm nhanh các chunk liên quan.

### Indexing dùng để làm gì?

Indexing dùng để:

- Lưu chunk đã xử lý.
- Lưu embedding của từng chunk.
- Lưu chỉ mục BM25 cho tìm kiếm từ khóa.
- Lưu metadata đi kèm chunk.
- Cho phép retrieval chạy nhanh khi người dùng đặt câu hỏi.

### Vector index là gì?

**Vector index** là chỉ mục dùng để tìm các vector gần nhau.

Trong RAG, mỗi chunk có một vector. Khi người dùng hỏi, câu hỏi cũng được chuyển thành vector. Vector index giúp tìm các chunk có vector gần với vector câu hỏi.

### BM25 index là gì?

**BM25 index** là chỉ mục dùng cho tìm kiếm từ khóa. Nó đánh giá một tài liệu có liên quan đến query hay không dựa trên tần suất và độ hiếm của từ khóa.

BM25 mạnh khi câu hỏi chứa:

- Tên riêng.
- Mã số.
- Thuật ngữ chính xác.
- Từ khóa hiếm.
- Cụm từ cần match đúng.

Ví dụ query “QĐ-123”, “Điều 5”, “học phí học kỳ 2” có thể được BM25 xử lý tốt hơn pure vector search.

### Persistence là gì?

**Persistence** là việc lưu dữ liệu đã xử lý xuống ổ đĩa để dùng lại.

Nếu không có persistence, mỗi lần khởi động lại hệ thống phải ingestion và indexing lại toàn bộ tài liệu. Persistence giúp tiết kiệm thời gian bằng cách lưu index, chunk và metadata sau khi xử lý.

---

## 3.5. Giai đoạn Retrieval

**Retrieval** là giai đoạn tìm các chunk liên quan đến câu hỏi của người dùng.

Đây là trái tim của RAG. Nếu retrieval lấy sai tài liệu, LLM có thể trả lời sai dù model rất mạnh. Trong nhiều hệ thống RAG, vấn đề không nằm ở LLM mà nằm ở retrieval.

### Retrieval dùng để làm gì?

Retrieval dùng để:

- Tìm đoạn tài liệu liên quan nhất.
- Giảm lượng context cần đưa vào LLM.
- Cung cấp bằng chứng cho generation.
- Giúp câu trả lời bám sát tài liệu.

### Dense Retrieval là gì?

**Dense Retrieval** là tìm kiếm dựa trên vector embedding.

Cách hoạt động:

1. Biến câu hỏi thành vector.
2. So sánh vector câu hỏi với vector các chunk.
3. Chọn những chunk có vector gần nhất.

Dense retrieval mạnh khi người dùng hỏi bằng cách diễn đạt khác với tài liệu. Ví dụ tài liệu viết “đóng học phí”, người dùng hỏi “nộp tiền học kỳ”, dense retrieval vẫn có thể tìm được.

### Sparse Retrieval là gì?

**Sparse Retrieval** là tìm kiếm dựa trên từ khóa. BM25 là một dạng sparse retrieval phổ biến.

Sparse retrieval mạnh khi cần match chính xác từ, mã, tên riêng hoặc thuật ngữ hiếm.

Ví dụ:

- “qwen2.5:3b”
- “BAAI/bge-reranker-v2-m3”
- “Điều 12”
- “Mã học phần INT2204”

Những nội dung này đôi khi vector search có thể bỏ sót, nhưng BM25 thường bắt tốt.

### Hybrid Retrieval là gì?

**Hybrid Retrieval** là truy xuất lai, kết hợp nhiều phương pháp retrieval.

Trong repo, ý tưởng chính là kết hợp:

- Dense retrieval: tìm theo ý nghĩa.
- Sparse retrieval/BM25: tìm theo từ khóa.

Kết quả từ nhiều nguồn được hợp nhất để tăng khả năng lấy đúng chunk.

### Vì sao cần Hybrid Retrieval?

Vì mỗi phương pháp có điểm mạnh và điểm yếu:

| Phương pháp | Mạnh ở đâu | Yếu ở đâu |
|---|---|---|
| Dense retrieval | Hiểu ngữ nghĩa, hiểu diễn đạt khác nhau | Có thể bỏ sót mã, tên riêng, thuật ngữ chính xác |
| Sparse retrieval/BM25 | Bắt từ khóa chính xác tốt | Không hiểu tốt khi người dùng diễn đạt khác tài liệu |
| Hybrid retrieval | Kết hợp cả hai | Phức tạp hơn, cần hợp nhất điểm |

Hybrid retrieval giúp hệ thống ổn định hơn trong tài liệu thực tế, nhất là tài liệu có cả văn bản tự nhiên, bảng, thuật ngữ, mã số và tên riêng.

### RRF là gì?

**RRF** là viết tắt của **Reciprocal Rank Fusion**. Đây là kỹ thuật hợp nhất kết quả xếp hạng từ nhiều retriever.

Thay vì chỉ cộng điểm thô từ BM25 và vector search, RRF quan tâm đến thứ hạng của chunk trong từng danh sách kết quả. Chunk nào xuất hiện ở vị trí cao trong nhiều danh sách sẽ được ưu tiên.

Ví dụ:

- Chunk A đứng hạng 1 trong dense retrieval.
- Chunk A cũng đứng hạng 3 trong BM25.
- Chunk B chỉ đứng hạng 1 trong BM25 nhưng không xuất hiện trong dense retrieval.

RRF có thể ưu tiên Chunk A vì nó được nhiều phương pháp đồng thuận là liên quan.

---

## 3.6. Giai đoạn Reranking

**Reranking** là bước sắp xếp lại các chunk đã truy xuất để chọn ra context tốt nhất.

Retrieval thường lấy ra một danh sách ứng viên ban đầu. Danh sách này có thể chứa cả chunk tốt và chunk nhiễu. Reranker đọc kỹ hơn từng cặp câu hỏi - chunk để đánh giá chunk nào thật sự liên quan.

### Reranker dùng để làm gì?

Reranker dùng để:

- Tăng độ chính xác của context cuối cùng.
- Đẩy chunk liên quan nhất lên đầu.
- Loại bớt chunk chỉ giống bề mặt nhưng không trả lời đúng câu hỏi.
- Cải thiện chất lượng câu trả lời cuối.

### Cross-Encoder là gì?

**Cross-Encoder** là một loại mô hình đánh giá trực tiếp mức liên quan giữa một query và một đoạn văn.

Khác với embedding model thường tạo vector riêng cho query và chunk, cross-encoder nhận cả query và chunk cùng lúc, rồi xuất ra một điểm liên quan. Vì đọc hai bên cùng lúc, cross-encoder thường chính xác hơn nhưng chậm hơn.

### BGE Reranker là gì?

**BGE Reranker** là một dòng mô hình reranking dùng để chấm điểm mức liên quan giữa câu hỏi và đoạn tài liệu. Trong hệ thống này, reranker được dùng sau retrieval để cải thiện thứ tự các chunk trước khi đưa vào LLM.

### Vì sao không dùng reranker cho toàn bộ tài liệu?

Vì reranker chậm hơn retrieval. Nếu tài liệu có hàng nghìn chunk, chấm điểm tất cả bằng cross-encoder sẽ tốn thời gian.

Quy trình hợp lý là:

1. Retrieval lấy ra một số ứng viên ban đầu.
2. Reranker chỉ xử lý top candidate.
3. Context selector chọn một số chunk tốt nhất.

---

## 3.7. Giai đoạn Context Selection

**Context Selection** là bước chọn những chunk cuối cùng để đưa vào prompt cho LLM.

Sau retrieval và reranking, hệ thống vẫn không nên đưa tất cả chunk vào LLM. Nó cần chọn một tập context vừa đủ.

### Context Selection dùng để làm gì?

Context selection dùng để:

- Giữ prompt ngắn gọn.
- Tránh nhồi quá nhiều thông tin nhiễu.
- Ưu tiên chunk có điểm cao.
- Đảm bảo context nằm trong giới hạn token.
- Tăng khả năng LLM trả lời đúng trọng tâm.

### Lost-in-the-middle là gì?

**Lost-in-the-middle** là hiện tượng LLM bỏ sót thông tin nằm giữa một context quá dài.

Nếu prompt chứa quá nhiều đoạn, thông tin quan trọng có thể bị “chìm” ở giữa. Vì vậy context selector cần chọn ít nhưng chất lượng, và sắp xếp context hợp lý.

---

## 3.8. Giai đoạn Generation

**Generation** là giai đoạn LLM tạo câu trả lời dựa trên câu hỏi và context đã chọn.

LLM không tự đọc trực tiếp toàn bộ tài liệu. Nó chỉ nhận:

- Câu hỏi của người dùng.
- Các đoạn context liên quan.
- Chỉ dẫn trong prompt.
- Một số cấu hình như temperature, max tokens.

### Generation dùng để làm gì?

Generation dùng để:

- Tổng hợp thông tin từ nhiều chunk.
- Trả lời bằng ngôn ngữ tự nhiên.
- Diễn giải nội dung tài liệu dễ hiểu hơn.
- Tạo câu trả lời có cấu trúc.
- Nêu rõ khi không đủ bằng chứng.

### Prompt là gì?

**Prompt** là phần chỉ dẫn gửi cho LLM.

Prompt thường bao gồm:

- Vai trò của mô hình.
- Quy tắc trả lời.
- Câu hỏi người dùng.
- Context từ tài liệu.
- Yêu cầu không bịa thông tin.
- Yêu cầu trả lời theo định dạng nhất định.

Trong RAG, prompt rất quan trọng vì nó điều khiển cách LLM dùng context.

### LLM là gì?

**LLM** là viết tắt của **Large Language Model**, nghĩa là mô hình ngôn ngữ lớn.

LLM có khả năng hiểu và sinh ngôn ngữ tự nhiên. Trong hệ thống này, LLM được dùng để tạo câu trả lời cuối cùng sau khi retrieval đã lấy được tài liệu liên quan.

### Qwen là gì?

**Qwen** là một họ mô hình ngôn ngữ lớn. Trong repo, Qwen là lựa chọn thường dùng cho local demo, nhưng hệ thống không khóa cứng vào riêng Qwen.

Backend gọi LLM qua chuẩn **OpenAI-compatible API** và có cơ chế fallback an toàn. Vì vậy có thể đổi model/runtime mà không phải viết lại toàn bộ workflow.

Qwen có nhiều kích thước khác nhau. Model nhỏ như 3B phù hợp chạy local nhanh hơn, nhưng chất lượng có thể kém hơn model lớn. Model lớn hơn như 7B hoặc 14B thường trả lời tốt hơn nhưng cần nhiều RAM/VRAM hơn.

### Ollama là gì?

**Ollama** là công cụ giúp chạy LLM local trên máy cá nhân hoặc server.

Thay vì gọi API cloud, Ollama cho phép tải model về máy và chạy trực tiếp. Trong hệ thống này, Ollama là một runtime local phổ biến, nhưng không phải lựa chọn bắt buộc.

Ollama hữu ích vì:

- Chạy được LLM trên máy cá nhân.
- Không cần gửi dữ liệu ra dịch vụ cloud.
- Phù hợp demo, học tập, thử nghiệm RAG local.
- Có API local để backend gọi model.

### OpenAI-compatible API là gì?

**OpenAI-compatible API** là kiểu API có giao diện tương tự API của OpenAI.

Nhiều runtime như Ollama, vLLM, SGLang có thể cung cấp endpoint tương thích OpenAI. Nhờ vậy backend chỉ cần nói chuyện với một chuẩn API quen thuộc, còn phía sau có thể đổi model hoặc runtime.

Lợi ích:

- Dễ đổi từ Ollama sang vLLM hoặc SGLang.
- Dễ đổi model Qwen nhỏ sang Qwen lớn.
- Backend không cần viết lại toàn bộ logic gọi LLM.
- Khi runtime chính lỗi, hệ thống có thể fallback an toàn thay vì crash toàn bộ pipeline.

### vLLM là gì?

**vLLM** là framework phục vụ LLM hiệu năng cao, thường dùng trên GPU server. Nó tối ưu throughput và latency khi chạy model lớn hoặc phục vụ nhiều request.

Trong bối cảnh repo, vLLM là lựa chọn thay thế Ollama khi cần triển khai mạnh hơn, nhanh hơn hoặc chuyên nghiệp hơn.

### SGLang là gì?

**SGLang** là framework dùng để phục vụ và điều phối LLM. Nó cũng có thể cung cấp API tương thích OpenAI, phù hợp khi muốn chạy LLM trên server với hiệu năng tốt hơn môi trường local đơn giản.

---

## 3.9. Temperature là gì?

**Temperature** là tham số điều khiển mức độ ngẫu nhiên của câu trả lời từ LLM.

- Temperature thấp: câu trả lời ổn định, ít sáng tạo, ít lệch.
- Temperature cao: câu trả lời đa dạng, sáng tạo hơn, nhưng dễ sai hoặc bịa hơn.

Trong RAG, thường dùng temperature thấp vì mục tiêu là trả lời chính xác dựa trên tài liệu, không phải sáng tác tự do.

Ví dụ:

| Temperature | Hành vi thường gặp |
|---|---|
| 0.0 - 0.2 | Rất ổn định, phù hợp hỏi đáp tài liệu |
| 0.3 - 0.7 | Có chút linh hoạt, phù hợp diễn giải |
| 0.8 trở lên | Sáng tạo hơn, rủi ro bịa cao hơn |

Với RAG học thuật, pháp lý, tài chính hoặc tài liệu nội bộ, nên ưu tiên temperature thấp.

---

## 3.10. Max tokens là gì?

**Max tokens** là giới hạn độ dài tối đa của câu trả lời hoặc một phần output mà LLM được phép sinh ra.

Nếu max tokens quá thấp, câu trả lời có thể bị cụt. Nếu quá cao, câu trả lời dài, tốn tài nguyên và có thể lan man.

Trong RAG, max tokens cần cân bằng giữa:

- Đủ dài để trả lời đầy đủ.
- Không quá dài để tránh chậm và tốn tài nguyên.
- Phù hợp độ phức tạp của câu hỏi.

---

## 3.11. Grounding là gì?

**Grounding** là kiểm tra câu trả lời có bám vào tài liệu được truy xuất hay không.

Một câu trả lời grounded là câu trả lời có cơ sở từ context. Ngược lại, nếu LLM nói điều không có trong tài liệu, đó là dấu hiệu hallucination.

### Grounding dùng để làm gì?

Grounding dùng để:

- Giảm câu trả lời bịa.
- Kiểm tra câu trả lời có bằng chứng không.
- Quyết định có nên trả lời hay từ chối.
- Tăng độ tin cậy của hệ thống.

### Grounding hoạt động như thế nào?

Có nhiều cách grounding:

- So khớp từ vựng giữa answer và context.
- So khớp ngữ nghĩa giữa answer và context.
- Dùng LLM hoặc heuristic để kiểm tra mức hỗ trợ.
- Kiểm tra có citation hoặc source liên quan không.

Trong hệ thống này, grounding có thể được cấu hình linh hoạt giữa ưu tiên tốc độ và ưu tiên độ chặt.

---

## 3.12. Hallucination là gì?

**Hallucination** là hiện tượng LLM tạo ra thông tin nghe có vẻ đúng nhưng thực tế không có cơ sở hoặc không nằm trong tài liệu.

Ví dụ người dùng hỏi “Ngày nộp học phí là ngày nào?”, nhưng tài liệu không có thông tin này. Nếu LLM tự bịa “ngày 15/09” thì đó là hallucination.

RAG giảm hallucination bằng cách cung cấp context thật. Tuy nhiên RAG không tự động loại bỏ hoàn toàn hallucination. Vì vậy hệ thống cần thêm grounding, abstain, critique và evaluation.

---

## 3.13. Citation là gì?

**Citation** là thông tin nguồn cho câu trả lời.

Trong RAG, citation cho biết câu trả lời dựa trên chunk nào, tài liệu nào hoặc trang nào. Citation giúp người dùng kiểm tra lại thông tin.

Ngay cả khi người dùng không cần hiển thị citation, hệ thống vẫn nên có logic quản lý nguồn ở bên trong để tránh trả lời vô căn cứ.

---

## 3.14. Abstain là gì?

**Abstain** nghĩa là hệ thống từ chối trả lời khi không đủ bằng chứng.

Đây là hành vi rất quan trọng trong RAG chất lượng cao. Một hệ thống tốt không phải lúc nào cũng cố trả lời. Nếu tài liệu không chứa thông tin cần thiết, câu trả lời tốt nhất có thể là:

> Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn.

Abstain giúp giảm hallucination và tăng độ tin cậy.

---

## 3.15. Query Rewrite là gì?

**Query Rewrite** là bước viết lại câu hỏi của người dùng để retrieval tốt hơn.

Người dùng có thể hỏi mơ hồ, quá ngắn hoặc dùng cách diễn đạt không giống tài liệu. Query rewrite giúp biến câu hỏi thành phiên bản rõ hơn, dễ tìm kiếm hơn.

Ví dụ:

- Câu hỏi gốc: “cái này tính sao?”
- Query rewrite: “Cách tính chi phí vận hành trong báo cáo là gì?”

Trong advanced mode, query rewrite giúp hệ thống truy xuất lại khi lần tìm kiếm đầu chưa đủ tốt.

---

## 3.16. Critique là gì?

**Critique** là bước tự đánh giá câu trả lời hoặc context.

Sau khi sinh câu trả lời nháp, hệ thống có thể kiểm tra:

- Câu trả lời có bám context không?
- Có thiếu bằng chứng không?
- Có mâu thuẫn với tài liệu không?
- Có cần retrieval lại không?
- Có nên refine hoặc abstain không?

Critique giúp advanced mode đáng tin cậy hơn standard mode, nhưng cũng làm tăng độ trễ.

---

## 3.17. Refine là gì?

**Refine** là bước chỉnh sửa câu trả lời sau khi critique phát hiện vấn đề.

Ví dụ câu trả lời ban đầu quá chung chung, thiếu nguồn hoặc diễn đạt chưa đúng. Refine sẽ tạo phiên bản tốt hơn, bám context hơn.

Refine khác với rewrite:

- Rewrite: viết lại câu hỏi để truy xuất tốt hơn.
- Refine: chỉnh lại câu trả lời sau khi đã có draft.

---

## 3.18. Retry là gì?

**Retry** là thử lại một bước trong pipeline, thường là retrieval hoặc generation.

Trong RAG nâng cao, nếu context ban đầu yếu, hệ thống có thể:

1. Viết lại query.
2. Retrieval lại.
3. Rerank lại.
4. Sinh câu trả lời mới.

Retry giúp tăng chất lượng nhưng phải có giới hạn vòng lặp. Nếu không giới hạn, hệ thống có thể chạy lâu, tốn tài nguyên hoặc lặp vô hạn.

---

## 3.19. Fallback là gì?

**Fallback** là cơ chế dự phòng khi một thành phần lỗi hoặc không khả dụng.

Ví dụ:

- Model embedding không tải được thì dùng embedding đơn giản hơn.
- Reranker lỗi thì dùng điểm retrieval ban đầu.
- LLM lỗi thì trả output an toàn hoặc thông báo không xử lý được.
- OCR lỗi thì bỏ qua OCR và tiếp tục pipeline bình thường.

Fallback giúp hệ thống không bị crash toàn bộ khi một phần bị lỗi.

---

## 4. Các mode hoạt động trong repo

Repo hỗ trợ ba chế độ chính: **standard**, **advanced**, và **compare**.

---

## 4.1. Standard Mode

**Standard Mode** là pipeline RAG nền tảng.

Luồng khái quát:

```text
Câu hỏi
  → retrieval
  → rerank
  → chọn context
  → generation
  → trả lời
```

### Standard Mode dùng để làm gì?

Standard mode phù hợp khi:

- Câu hỏi tương đối rõ.
- Tài liệu có thông tin trực tiếp.
- Cần phản hồi nhanh.
- Muốn dùng làm baseline để so sánh.

### Ưu điểm

- Nhanh hơn advanced mode.
- Ít bước hơn.
- Dễ debug hơn.
- Phù hợp truy vấn đơn giản.

### Nhược điểm

- Không có vòng tự kiểm tra sâu.
- Khả năng xử lý câu hỏi mơ hồ kém hơn advanced mode.
- Ít cơ chế retry/refine hơn.

---

## 4.2. Advanced Mode

**Advanced Mode** là chế độ Self-RAG nâng cao.

Luồng khái quát:

```text
Câu hỏi
  → kiểm tra có cần retrieval không
  → rewrite query nếu cần
  → retrieval
  → rerank
  → sinh draft
  → critique
  → retry/refine/abstain
  → trả lời
```

### Advanced Mode dùng để làm gì?

Advanced mode phù hợp khi:

- Câu hỏi mơ hồ.
- Câu hỏi cần nhiều bước suy luận.
- Có khả năng thiếu bằng chứng.
- Tài liệu có thông tin mâu thuẫn.
- Cần độ tin cậy cao hơn tốc độ.

### Ưu điểm

- Có thể tự kiểm tra câu trả lời.
- Có thể từ chối nếu thiếu bằng chứng.
- Có thể rewrite query để tìm tốt hơn.
- Có thể refine câu trả lời.
- Giảm hallucination tốt hơn.

### Nhược điểm

- Chậm hơn.
- Tốn nhiều tài nguyên hơn.
- Pipeline phức tạp hơn.
- Dễ khó debug hơn standard mode.

---

## 4.3. Compare Mode

**Compare Mode** chạy cả standard mode và advanced mode cho cùng một câu hỏi để so sánh trực tiếp.

Trong implementation hiện tại, compare mode mặc định chạy **tuần tự** (standard trước, advanced sau) để có thể tái sử dụng kết quả standard cho advanced, giúp giảm trùng lặp xử lý. Có thể bật chạy song song bằng cấu hình khi cần.

Luồng khái quát:

```text
Câu hỏi
  → chạy standard
  → chạy advanced
  → so sánh kết quả
  → trả về tổng hợp
```

### Compare Mode dùng để làm gì?

Compare mode phù hợp khi:

- Muốn so sánh chất lượng hai pipeline.
- Muốn kiểm tra advanced mode có cải thiện không.
- Muốn xem sự khác biệt về confidence, latency, câu trả lời.
- Muốn demo hệ thống rõ ràng hơn.

### Ưu điểm

- Dễ quan sát sự khác nhau giữa hai mode.
- Hữu ích cho đánh giá và debug.
- Phù hợp thuyết trình vì minh họa được tradeoff.

### Nhược điểm

- Chậm hơn vì phải chạy nhiều nhánh.
- Tốn tài nguyên hơn.
- Không phải mode tối ưu cho người dùng cuối nếu chỉ cần một câu trả lời nhanh.

---

## 5. Các công nghệ backend trong repo

## 5.1. Python

**Python** là ngôn ngữ chính của backend. Python phổ biến trong AI, machine learning và xử lý dữ liệu vì có nhiều thư viện hỗ trợ NLP, PDF, embedding, LLM và API.

Trong hệ thống này, Python đảm nhiệm:

- Xử lý tài liệu.
- Tạo embedding.
- Xây dựng index.
- Truy xuất dữ liệu.
- Gọi LLM.
- Chạy workflow RAG.
- Cung cấp API backend.

---

## 5.2. FastAPI

**FastAPI** là framework Python dùng để xây dựng API backend.

FastAPI phù hợp cho hệ thống RAG vì:

- Tốc độ tốt.
- Hỗ trợ kiểu dữ liệu rõ ràng.
- Dễ viết API upload tài liệu và query.
- Tích hợp tốt với Pydantic.
- Dễ triển khai với Uvicorn.

Trong repo, FastAPI đóng vai trò cổng giao tiếp giữa frontend hoặc client và pipeline RAG.

---

## 5.3. Uvicorn

**Uvicorn** là server dùng để chạy ứng dụng FastAPI.

FastAPI là framework định nghĩa API, còn Uvicorn là server thực sự nhận request HTTP và chuyển vào ứng dụng.

Trong môi trường dev, Uvicorn thường chạy với chế độ reload để tự cập nhật khi code thay đổi. Khi benchmark hoặc chạy ổn định hơn, nên tắt reload để đo latency chính xác hơn.

---

## 5.4. Pydantic

**Pydantic** là thư viện Python dùng để định nghĩa và kiểm tra dữ liệu.

Trong backend, Pydantic giúp:

- Kiểm tra request gửi lên API.
- Định nghĩa cấu trúc response.
- Quản lý cấu hình từ biến môi trường.
- Giảm lỗi do dữ liệu sai kiểu.

Ví dụ, một request query cần có câu hỏi, mode, hoặc tham số liên quan. Pydantic giúp đảm bảo dữ liệu đó đúng định dạng trước khi pipeline xử lý.

---

## 5.5. pydantic-settings

**pydantic-settings** là phần mở rộng giúp đọc cấu hình từ môi trường, ví dụ file `.env`.

Trong hệ thống RAG, cấu hình rất nhiều:

- Model LLM.
- Model embedding.
- Kích thước chunk.
- Số lượng top-k retrieval.
- Bật/tắt OCR.
- Bật/tắt cache.
- Chính sách grounding.

Dùng settings giúp thay đổi hành vi hệ thống mà không cần sửa code.

---

## 5.6. python-dotenv

**python-dotenv** dùng để đọc biến môi trường từ file `.env`.

File `.env` giúp lưu cấu hình local như địa chỉ Ollama, tên model, temperature, đường dẫn dữ liệu, cấu hình OCR và cache.

---

## 5.7. httpx

**httpx** là thư viện Python dùng để gửi HTTP request.

Trong hệ thống này, httpx có thể dùng khi backend cần gọi LLM server qua API tương thích OpenAI, ví dụ Ollama, vLLM hoặc SGLang.

---

## 6. Các công nghệ xử lý tài liệu

## 6.1. python-docx

**python-docx** là thư viện đọc file DOCX trong Python.

Nó giúp hệ thống trích xuất nội dung từ tài liệu Word, sau đó đưa vào pipeline ingestion giống như các loại tài liệu khác.

---

## 6.2. pdfplumber

**pdfplumber** dùng để trích xuất text, bảng và layout từ PDF.

Nó hữu ích khi cần hiểu cấu trúc PDF, nhất là các tài liệu có bảng hoặc format phức tạp.

---

## 6.3. PyMuPDF

**PyMuPDF** dùng để thao tác PDF nâng cao, đặc biệt hữu ích khi cần render trang PDF thành ảnh cho OCR.

---

## 6.4. pypdf

**pypdf** là thư viện xử lý PDF cơ bản, thường dùng cho việc đọc trang, metadata hoặc text.

---

## 6.5. Pillow

**Pillow** là thư viện xử lý ảnh trong Python.

Trong pipeline OCR hoặc xử lý PDF scan, Pillow có thể hỗ trợ thao tác với ảnh trước khi đưa vào OCR.

---

## 6.6. pytesseract

**pytesseract** là thư viện Python để gọi Tesseract OCR. Nó giúp backend nhận dạng chữ từ ảnh.

---

## 6.7. underthesea

**underthesea** là thư viện xử lý ngôn ngữ tự nhiên tiếng Việt.

Trong RAG tiếng Việt, xử lý từ, câu hoặc token tiếng Việt có thể khó hơn tiếng Anh vì cách tách từ phức tạp. Thư viện tiếng Việt giúp pipeline xử lý nội dung Việt tốt hơn ở một số bước.

---

## 7. Các công nghệ RAG / AI trong repo

## 7.1. Sentence Transformers

Sentence Transformers dùng để tạo embedding cho câu hỏi và chunk tài liệu.

Nó là nền tảng của dense retrieval. Nếu embedding tốt, hệ thống có thể tìm được tài liệu liên quan ngay cả khi câu hỏi không trùng từ khóa với tài liệu.

---

## 7.2. BM25

BM25 là thuật toán tìm kiếm từ khóa. Nó đánh giá một chunk có liên quan đến query hay không dựa trên tần suất từ khóa, độ hiếm của từ và độ dài tài liệu.

BM25 rất hữu ích trong hệ thống RAG vì không phải câu hỏi nào cũng nên tìm bằng ngữ nghĩa. Nhiều câu hỏi cần match chính xác mã, tên, thuật ngữ hoặc số hiệu.

---

## 7.3. Qwen

Qwen là họ LLM thường dùng trong ví dụ local để sinh câu trả lời. Tuy nhiên backend không khóa cứng vào Qwen: chỉ cần runtime cung cấp API tương thích OpenAI là có thể thay model/provider.

Qwen nhỏ giúp chạy nhanh hơn trên máy cá nhân. Qwen lớn cho chất lượng tốt hơn nhưng cần phần cứng mạnh hơn.

---

## 7.4. Ollama

Ollama là runtime giúp chạy LLM local. Nó cho phép tải model, khởi động server local và để backend gọi model qua API.

Trong demo RAG, Ollama rất tiện vì:

- Cài đặt đơn giản.
- Chạy được trên máy cá nhân.
- Dễ dùng với Qwen.
- Không cần API key cloud thật.
- Phù hợp môi trường học tập và thử nghiệm.

Lưu ý: Ollama là lựa chọn local phổ biến, không phải thành phần bắt buộc của kiến trúc. Có thể thay bằng vLLM/SGLang hoặc endpoint OpenAI-compatible khác.

---

## 7.5. torch

**torch** là thư viện nền tảng của PyTorch, dùng cho deep learning.

Các mô hình embedding, reranker hoặc transformer thường dựa trên PyTorch để chạy inference trên CPU hoặc GPU.

---

## 7.6. transformers

**transformers** là thư viện của Hugging Face dùng để làm việc với các mô hình Transformer.

Nhiều model LLM, embedding hoặc reranking có thể dùng hệ sinh thái transformers để tải model, tokenizer và chạy inference.

---

## 7.7. tokenizers

**tokenizers** dùng để chia text thành token cho model.

LLM và embedding model không đọc trực tiếp chuỗi ký tự như con người. Chúng xử lý token. Tokenizer quyết định cách văn bản được chia nhỏ trước khi vào model.

---

## 7.8. safetensors

**safetensors** là định dạng lưu trọng số model an toàn và nhanh hơn một số định dạng cũ.

Nó thường xuất hiện khi làm việc với model từ Hugging Face.

---

## 8. Các công nghệ frontend trong repo

## 8.1. React

**React** là thư viện JavaScript dùng để xây dựng giao diện người dùng.

Trong hệ thống RAG, frontend React có thể dùng để:

- Upload tài liệu.
- Chọn mode truy vấn.
- Nhập câu hỏi.
- Hiển thị câu trả lời.
- Hiển thị context, citation, trạng thái hoặc thống kê.

---

## 8.2. TypeScript

**TypeScript** là JavaScript có thêm kiểu dữ liệu tĩnh.

Nó giúp frontend dễ bảo trì hơn vì giảm lỗi sai kiểu dữ liệu, đặc biệt khi frontend nhận response phức tạp từ backend.

---

## 8.3. Vite

**Vite** là công cụ build và dev server cho frontend hiện đại.

Vite giúp chạy frontend nhanh trong lúc phát triển và build project để triển khai.

---

## 8.4. Tailwind CSS

**Tailwind CSS** là framework CSS theo hướng utility-first.

Thay vì viết nhiều CSS riêng, developer dùng các class tiện ích để tạo layout, màu sắc, spacing, font, border, responsive UI.

---

## 8.5. Recharts

**Recharts** là thư viện vẽ biểu đồ cho React.

Trong hệ thống RAG, Recharts có thể dùng để hiển thị:

- Latency.
- Số lượng request.
- So sánh mode standard/advanced.
- Kết quả evaluation.
- Metric retrieval.

---

## 8.6. lucide-react

**lucide-react** là thư viện icon cho React.

Nó giúp giao diện có các biểu tượng đẹp, nhẹ và nhất quán.

---

## 8.7. clsx và tailwind-merge

**clsx** giúp ghép class CSS có điều kiện. **tailwind-merge** giúp xử lý xung đột class Tailwind.

Hai thư viện này thường dùng để viết component UI gọn và dễ kiểm soát style hơn.

---

## 9. Caching trong hệ thống RAG

**Caching** là lưu tạm kết quả của một bước xử lý để dùng lại sau.

Trong RAG, nhiều bước có thể tốn thời gian:

- Tạo embedding.
- Retrieval.
- Reranking.
- Gọi LLM.
- Grounding.

Nếu cùng một input được xử lý nhiều lần, cache giúp trả kết quả nhanh hơn.

### LRU Cache là gì?

**LRU** là viết tắt của **Least Recently Used**. Đây là cơ chế cache loại bỏ mục ít được dùng gần đây nhất khi cache đầy.

Ví dụ cache chỉ chứa 3 mục:

```text
A, B, C
```

Nếu thêm D và A là mục lâu nhất không dùng, cache sẽ bỏ A:

```text
B, C, D
```

LRU phù hợp khi hệ thống có nhiều request lặp lại hoặc người dùng hỏi các câu gần giống nhau.

### Cache có lợi gì?

- Giảm latency.
- Giảm số lần gọi LLM.
- Giảm tải CPU/GPU.
- Tăng trải nghiệm khi demo local.

### Cache có rủi ro gì?

- Có thể trả kết quả cũ nếu dữ liệu đã thay đổi.
- Cần quản lý kích thước cache.
- Cần invalidation khi upload lại tài liệu hoặc thay đổi index.

---

## 10. Evaluation trong hệ thống RAG

**Evaluation** là quá trình đo chất lượng hệ thống.

RAG không nên chỉ đánh giá bằng cảm giác “câu trả lời có vẻ đúng”. Cần metric để biết retrieval có lấy đúng tài liệu không, generation có trả lời đúng không, advanced mode có tốt hơn standard không.

### Evaluation dùng để làm gì?

Evaluation dùng để:

- Kiểm tra chất lượng retrieval.
- Kiểm tra khả năng abstain khi thiếu bằng chứng.
- So sánh standard, advanced và compare mode.
- Phát hiện regression khi sửa code.
- Đo latency trước và sau tối ưu.

### Golden dataset là gì?

**Golden dataset** là bộ câu hỏi mẫu có đáp án hoặc hành vi mong đợi.

Ví dụ mỗi dòng có thể gồm:

- Câu hỏi.
- Loại câu hỏi.
- Đáp án tham khảo.
- Nguồn đúng.
- Hành vi mong đợi: trả lời, retry hoặc abstain.

Golden dataset giúp đánh giá hệ thống một cách lặp lại.

### Hit Rate là gì?

**Hit Rate** đo xem trong top-k chunk được retrieval có ít nhất một chunk đúng hay không.

Nếu hệ thống cần tìm tài liệu đúng trong top 5 kết quả, Hit Rate@5 cho biết tỷ lệ câu hỏi mà tài liệu đúng xuất hiện trong 5 kết quả đầu.

Hit Rate cao nghĩa là retrieval thường lấy được ít nhất một bằng chứng đúng.

### MRR là gì?

**MRR** là viết tắt của **Mean Reciprocal Rank**.

MRR không chỉ quan tâm có tìm được chunk đúng hay không, mà còn quan tâm chunk đúng đứng ở vị trí thứ mấy.

Nếu chunk đúng đứng hạng 1, điểm rất cao. Nếu đứng hạng 5, điểm thấp hơn. MRR cao nghĩa là kết quả đúng thường nằm rất gần đầu danh sách.

### nDCG là gì?

**nDCG** là metric đánh giá chất lượng xếp hạng khi có nhiều mức độ liên quan.

Không phải chunk nào cũng chỉ “đúng” hoặc “sai”. Có chunk rất liên quan, có chunk hơi liên quan. nDCG đánh giá hệ thống có xếp các chunk liên quan cao lên đầu hay không.

### Latency là gì?

**Latency** là độ trễ, tức thời gian từ lúc gửi request đến khi nhận được phản hồi.

Trong RAG, latency đến từ nhiều bước:

- Embedding query.
- Retrieval.
- Reranking.
- Gọi LLM.
- Grounding.
- Streaming output.

### p50, p90, p95 là gì?

Đây là các thống kê latency theo percentile:

- **p50**: 50% request nhanh hơn mức này.
- **p90**: 90% request nhanh hơn mức này.
- **p95**: 95% request nhanh hơn mức này.

p95 quan trọng vì nó cho thấy trải nghiệm của các request chậm, không chỉ request trung bình.

---

## 11. CI/CD và kiểm soát chất lượng code

## 11.1. GitHub Actions

**GitHub Actions** là hệ thống CI/CD của GitHub.

Nó có thể tự động chạy test, lint, type check và kiểm tra bảo mật mỗi khi có thay đổi code.

### CI/CD dùng để làm gì?

- Tự động kiểm tra code trước khi merge.
- Phát hiện lỗi sớm.
- Đảm bảo test vẫn pass.
- Đảm bảo chất lượng code ổn định.
- Giảm rủi ro regression.

---

## 11.2. Pytest

**Pytest** là framework test cho Python.

Trong hệ thống RAG, test có thể kiểm tra:

- Ingestion đọc file đúng không.
- Chunking có giữ metadata không.
- Retrieval có trả kết quả hợp lệ không.
- Workflow standard/advanced/compare có chạy đúng không.
- API response có đúng schema không.

---

## 11.3. Ruff

**Ruff** là công cụ lint và format code Python rất nhanh.

Nó giúp phát hiện lỗi style, import thừa, pattern code không tốt và có thể tự format một số phần.

---

## 11.4. Mypy

**Mypy** là công cụ kiểm tra kiểu dữ liệu tĩnh cho Python.

Python là ngôn ngữ dynamic, nên lỗi sai kiểu có thể chỉ xuất hiện lúc chạy. Mypy giúp phát hiện sớm các lỗi liên quan đến kiểu dữ liệu.

---

## 11.5. Bandit

**Bandit** là công cụ quét bảo mật cho code Python.

Nó tìm các pattern có thể gây rủi ro bảo mật, ví dụ dùng lệnh hệ thống không an toàn hoặc xử lý dữ liệu nhạy cảm không đúng.

---

## 11.6. pip-audit

**pip-audit** kiểm tra các dependency Python có lỗ hổng bảo mật đã biết hay không.

Điều này quan trọng vì hệ thống AI thường phụ thuộc vào nhiều thư viện bên ngoài.

---

## 11.7. Workflow State và Critique Schema

Repo dùng một state object dùng chung để điều phối workflow nâng cao, thay vì truyền dictionary rời rạc.

State này gồm các trường cốt lõi như:

- `mode`, `user_query`, `normalized_query`, `response_language`
- `chat_history`, `need_retrieval`
- `rewritten_queries`, `retrieved_docs`, `reranked_docs`, `selected_context`
- `draft_answer`, `final_answer`, `citations`
- `confidence`, `loop_count`, `stop_reason`
- tín hiệu an toàn như `grounded_score`, `grounding_reason`, `hallucination_detected`, `language_mismatch`

Kết quả critique cũng có schema rõ ràng:

- `grounded`, `enough_evidence`, `has_conflict`
- `missing_aspects`, `should_retry_retrieval`, `should_refine_answer`
- `better_queries`, `confidence`, `note`

Điểm quan trọng: schema chặt giúp pipeline dễ test, dễ log, và giảm lỗi parse khi advanced mode cần ra quyết định retry/refine/abstain.

---

## 11.8. API Contract chính trong repo

Backend dùng FastAPI với prefix `/api/v1`.

Các endpoint chính:

- `POST /api/v1/query`
- `POST /api/v1/query/stream` (SSE)
- `POST /api/v1/documents/upload`
- `GET /api/v1/documents`
- `GET /api/v1/documents/{document_id}/status`
- `POST /api/v1/documents/reindex`
- `POST /api/v1/settings/chunking`
- `POST /api/v1/settings/retrieval`

`QueryRequest` ngoài `query` và `mode` còn hỗ trợ filter:

- `doc_ids`, `filenames`, `file_types`
- `uploaded_after`, `uploaded_before`
- `include_ocr`

`QueryResponse` là union schema theo mode:

- `StandardQueryResponse`
- `AdvancedQueryResponse`
- `CompareQueryResponse`

Compare response trả về đủ 3 phần:

- nhánh `standard`
- nhánh `advanced`
- phần `comparison` (winner, reasons, delta confidence/latency/citation/groundedness)

---

## 11.9. Prompt Contract trong thư mục `/prompts`

Các prompt dài không nên hardcode trong business logic.

Những prompt file cốt lõi đang có:

- `standard_answer.md`
- `advanced_answer.md`
- `retrieval_gate.md`
- `query_rewrite.md`
- `critique.md`
- `refine.md`

Ngoài ra repo còn có prompt hỗ trợ strict grounding:

- `refine_grounded.md`

---

## 11.10. Cấu hình thực thi quan trọng (hiện tại)

Để đọc đúng hành vi runtime, cần chú ý vài cờ:

- Compare mode mặc định chạy tuần tự (`COMPARE_PARALLEL_ENABLED=false`), có thể bật song song khi cần.
- Advanced loop hiện mặc định `MAX_ADVANCED_LOOPS=1`.
- Nếu muốn hành vi gần với thiết lập Self-RAG chặt hơn, có thể tăng loop (ví dụ `MAX_ADVANCED_LOOPS=2`) nhưng chi phí/độ trễ sẽ tăng.
- LLM provider mặc định trong config là `stub`; khi dùng model thật thì thường chuyển sang endpoint OpenAI-compatible (Ollama/vLLM/SGLang...).

---

## 12. Luồng hoạt động chi tiết từ lúc upload tài liệu đến lúc trả lời

## 12.1. Khi người dùng upload tài liệu

1. Frontend hoặc client gửi file lên backend.
2. Backend lưu file vào khu vực dữ liệu.
3. Hệ thống xác định loại file.
4. Parser phù hợp trích xuất nội dung.
5. Nếu PDF scan và OCR được bật, hệ thống thử OCR các trang thiếu text.
6. Nội dung được làm sạch.
7. Tài liệu được chia thành chunk.
8. Chunk được gắn metadata.
9. Hệ thống tạo embedding cho chunk.
10. Chunk được đưa vào vector index.
11. Chunk cũng được đưa vào BM25 index.
12. Index được lưu lại để truy vấn sau.
13. Trạng thái tài liệu chuyển dần từ upload sang sẵn sàng.

---

## 12.2. Khi người dùng đặt câu hỏi ở standard mode

1. Backend nhận câu hỏi.
2. Câu hỏi được chuyển thành embedding.
3. Dense retrieval tìm chunk gần về ngữ nghĩa.
4. Sparse retrieval/BM25 tìm chunk trùng từ khóa.
5. Hybrid retrieval hợp nhất kết quả.
6. Reranker sắp xếp lại top candidate.
7. Context selector chọn chunk tốt nhất.
8. Prompt được tạo từ câu hỏi và context.
9. LLM sinh câu trả lời.
10. Hệ thống kiểm tra mức độ đủ bằng chứng cơ bản.
11. Kết quả được trả về frontend.

---

## 12.3. Khi người dùng đặt câu hỏi ở advanced mode

1. Backend nhận câu hỏi.
2. Hệ thống đánh giá câu hỏi có cần retrieval, rewrite hoặc xử lý nâng cao không.
3. Nếu câu hỏi mơ hồ, query có thể được viết lại.
4. Retrieval chạy với query đã chuẩn hóa hoặc rewrite.
5. Kết quả được rerank.
6. Context tốt nhất được chọn.
7. LLM tạo câu trả lời nháp.
8. Critique kiểm tra câu trả lời có đủ bằng chứng không.
9. Nếu bằng chứng yếu, hệ thống có thể retry retrieval.
10. Nếu trả lời chưa tốt, hệ thống refine câu trả lời.
11. Nếu không đủ thông tin, hệ thống abstain.
12. Câu trả lời cuối được trả về.

---

## 12.4. Khi người dùng đặt câu hỏi ở compare mode

1. Backend nhận câu hỏi.
2. Mặc định: Standard mode xử lý trước.
3. Advanced mode xử lý cùng câu hỏi, có thể tái sử dụng một phần kết quả từ standard để giảm trùng lặp.
4. Tùy cấu hình, compare mode có thể chạy song song hai nhánh.
5. Hệ thống so sánh hai kết quả.
6. Response trả về gồm kết quả từng mode và phần tổng hợp so sánh.

---

## 13. Các trạng thái xử lý tài liệu

Khi upload tài liệu, hệ thống không phải lúc nào cũng sẵn sàng ngay. Tài liệu có thể đi qua nhiều trạng thái:

| Trạng thái | Ý nghĩa |
|---|---|
| uploaded | File đã được nhận |
| splitting | Đang chia tài liệu thành chunk |
| embedding | Đang tạo vector embedding |
| indexing | Đang đưa chunk vào index |
| ready | Tài liệu đã sẵn sàng để query |
| failed | Xử lý thất bại |

Trạng thái này giúp frontend hiển thị tiến độ và tránh query tài liệu khi index chưa sẵn sàng.

---

## 14. Fast mode và strict mode

Trong hệ thống RAG, luôn có đánh đổi giữa tốc độ và độ tin cậy.

### Fast mode

Fast mode ưu tiên phản hồi nhanh, phù hợp demo local hoặc máy cấu hình vừa phải.

Đặc điểm:

- Dùng model nhỏ hơn.
- Giới hạn max tokens thấp hơn.
- Có thể bỏ qua một số kiểm tra nặng khi tín hiệu đã đủ tốt.
- Reranker có thể được skip trong trường hợp đơn giản.
- Semantic grounding có thể giảm độ chặt ở standard mode.

Fast mode phù hợp khi:

- Chạy trên laptop.
- Demo nhanh.
- Câu hỏi đơn giản.
- Chấp nhận tradeoff nhỏ về độ chặt.

### Strict mode

Strict mode ưu tiên độ tin cậy và kiểm tra chặt hơn.

Đặc điểm:

- Grounding nghiêm ngặt hơn.
- Có thể dùng reranker nhiều hơn.
- Advanced gate, critique, refine được dùng tích cực hơn.
- Tốn thời gian và tài nguyên hơn.

Strict mode phù hợp khi:

- Dữ liệu quan trọng.
- Câu hỏi phức tạp.
- Cần giảm hallucination tối đa.
- Chạy trên server mạnh hơn.

---

## 15. Vì sao hệ thống dùng kiến trúc module?

Kiến trúc module nghĩa là mỗi phần có trách nhiệm riêng:

- Ingestion xử lý tài liệu.
- Indexing tạo chỉ mục.
- Retrieval tìm chunk.
- Reranking sắp xếp lại.
- Generation tạo câu trả lời.
- Workflow điều phối các bước.
- Evaluation đo chất lượng.
- API giao tiếp với frontend.
- Frontend hiển thị cho người dùng.

Lợi ích của kiến trúc module:

- Dễ thay model embedding.
- Dễ thay LLM runtime.
- Dễ thay reranker.
- Dễ debug từng tầng.
- Dễ test riêng từng phần.
- Dễ mở rộng lên production.

Ví dụ, nếu muốn đổi từ Ollama sang vLLM, chỉ cần đảm bảo backend vẫn gọi qua chuẩn API tương thích. Nếu muốn đổi vector store sang Qdrant hoặc Milvus, tầng indexing/retrieval có thể thay đổi mà không cần sửa toàn bộ frontend.

---

## 16. Các vấn đề RAG mà repo đang cố giải quyết

## 16.1. Retrieval sai

Nếu retrieval lấy sai chunk, LLM sẽ trả lời sai hoặc thiếu. Repo dùng hybrid retrieval và reranker để giảm vấn đề này.

## 16.2. Hallucination

LLM có thể bịa khi thiếu context. Repo dùng grounding, abstain, critique và citation để giảm rủi ro.

## 16.3. Lost-in-the-middle

Context quá dài có thể làm LLM bỏ sót thông tin giữa prompt. Repo dùng chunking, reranking và context selection để chọn context gọn hơn.

## 16.4. Tài liệu có cấu trúc phức tạp

PDF, bảng, heading, scan và tài liệu tiếng Việt có thể khó parse. Repo có ingestion nhận diện cấu trúc, OCR tùy chọn và metadata để giữ ngữ cảnh.

## 16.5. Suy giảm chất lượng âm thầm

Khi sửa code, retrieval có thể kém đi mà không ai nhận ra. Evaluation và regression check giúp phát hiện sớm.

## 16.6. Độ trễ cao

RAG có nhiều bước tốn thời gian. Repo dùng cache, fast profile, dynamic budget và cascade reranking để tối ưu latency.

---

## 17. Bảng thuật ngữ nhanh

| Thuật ngữ | Giải thích ngắn |
|---|---|
| RAG | Kỹ thuật trả lời bằng cách truy xuất tài liệu rồi đưa cho LLM sinh câu trả lời |
| Ingestion | Đưa tài liệu vào hệ thống và xử lý thành chunk |
| Parser | Thành phần đọc nội dung từ file |
| OCR | Nhận dạng chữ từ ảnh hoặc PDF scan |
| Chunk | Đoạn nhỏ được cắt ra từ tài liệu |
| Chunk size | Kích thước mục tiêu của mỗi chunk |
| Chunk overlap | Phần lặp lại giữa các chunk liền nhau |
| Metadata | Thông tin mô tả chunk như nguồn, trang, section |
| Embedding | Vector số đại diện cho ý nghĩa văn bản |
| Vector index | Chỉ mục tìm kiếm theo vector |
| BM25 | Thuật toán tìm kiếm theo từ khóa |
| Dense retrieval | Truy xuất theo embedding/ngữ nghĩa |
| Sparse retrieval | Truy xuất theo từ khóa |
| Hybrid retrieval | Kết hợp dense và sparse retrieval |
| RRF | Kỹ thuật hợp nhất nhiều danh sách xếp hạng |
| Reranker | Mô hình sắp xếp lại kết quả retrieval |
| Cross-Encoder | Mô hình chấm điểm trực tiếp cặp query-document |
| Context | Các đoạn tài liệu được đưa vào LLM |
| Prompt | Chỉ dẫn gửi cho LLM |
| LLM | Mô hình ngôn ngữ lớn |
| Qwen | Họ mô hình LLM thường dùng cho local/demo, không bắt buộc |
| Ollama | Runtime local phổ biến để chạy LLM, có thể thay bằng runtime khác |
| Temperature | Tham số điều khiển độ ngẫu nhiên của output |
| Max tokens | Giới hạn độ dài output |
| Grounding | Kiểm tra câu trả lời có dựa trên tài liệu không |
| Hallucination | LLM bịa thông tin không có cơ sở |
| Citation | Nguồn hoặc bằng chứng cho câu trả lời |
| Abstain | Từ chối trả lời khi thiếu bằng chứng |
| Query rewrite | Viết lại câu hỏi để retrieval tốt hơn |
| Critique | Tự đánh giá câu trả lời hoặc bằng chứng |
| Refine | Chỉnh câu trả lời sau critique |
| Retry | Thử lại retrieval hoặc generation |
| Fallback | Cơ chế dự phòng khi một thành phần lỗi |
| LRU cache | Cache loại bỏ mục ít dùng gần đây nhất |
| Golden dataset | Bộ câu hỏi/đáp án chuẩn để đánh giá |
| Hit Rate | Tỷ lệ retrieval có lấy được kết quả đúng trong top-k |
| MRR | Metric đo vị trí kết quả đúng đầu tiên |
| nDCG | Metric đánh giá chất lượng xếp hạng |
| Latency | Độ trễ xử lý request |
| p95 | Mức latency mà 95% request nhanh hơn |

---

## 18. Cách trình bày ngắn gọn khi thuyết trình

Có thể trình bày hệ thống theo mạch sau:

1. **Vấn đề**: LLM có thể không biết tài liệu riêng và dễ hallucinate.
2. **Giải pháp**: RAG truy xuất tài liệu liên quan trước, rồi mới cho LLM trả lời.
3. **Ingestion**: hệ thống đọc PDF/DOCX/TXT/MD, xử lý text, bảng, OCR và chia chunk.
4. **Indexing**: mỗi chunk được tạo embedding và lưu vào vector index, đồng thời tạo BM25 index.
5. **Retrieval**: khi có câu hỏi, hệ thống tìm chunk bằng cả vector search và BM25.
6. **Hybrid + Rerank**: kết quả được hợp nhất và sắp xếp lại bằng reranker.
7. **Generation**: backend gọi LLM qua API OpenAI-compatible (ví dụ Qwen qua Ollama/vLLM/SGLang) để tạo câu trả lời dựa trên context.
8. **Grounding/Self-RAG**: hệ thống kiểm tra câu trả lời có đủ bằng chứng không, có thể refine hoặc từ chối.
9. **Evaluation**: dùng Hit Rate, MRR, nDCG và latency để đo chất lượng.
10. **Kết luận**: hệ thống này không chỉ hỏi đáp tài liệu, mà còn có cơ chế kiểm chứng và đánh giá để giảm hallucination.

---

## 19. Tóm tắt một câu

Hệ thống trong repo là một pipeline Self-RAG: tài liệu được ingestion, chunking, embedding và indexing; khi người dùng hỏi, hệ thống truy xuất bằng hybrid retrieval, rerank context, gọi LLM qua API OpenAI-compatible (không khóa cứng model/runtime), rồi kiểm tra grounding để hạn chế hallucination và hỗ trợ các mode standard, advanced, compare.
