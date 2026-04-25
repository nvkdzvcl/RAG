/**
 * Vietnamese translations for the application
 */

export const translations = {
  // App title
  // appName: "Self-RAG",
  appName: "SmartDocAI",

  // Modes
  modes: {
    standard: "Chuẩn",
    advanced: "Nâng cao",
    compare: "So sánh",
  },

  // Mode descriptions
  modeDescriptions: {
    standard: "RAG cơ bản - Nhanh và hiệu quả",
    advanced: "Self-RAG - Đáng tin cậy hơn với khả năng tự đánh giá",
    compare: "So sánh cả hai chế độ song song",
  },

  // Model selector
  model: {
    title: "Mô hình Qwen",
    active: "Model đang dùng",
    sharedAcrossModes: "Áp dụng chung cho Chuẩn, Nâng cao và So sánh",
  },

  // Sidebar
  sidebar: {
    newChat: "Cuộc trò chuyện mới",
    recentChats: "Lịch sử gần đây",
    noRecentChats: "Chưa có lịch sử",
  },

  // Document upload
  upload: {
    title: "Tải lên tài liệu",
    button: "Chọn tài liệu",
    uploading: "Đang tải lên...",
    processing: "Đang xử lý...",
    ready: "Sẵn sàng",
    failed: "Thất bại",
    dragDrop: "Kéo thả tài liệu vào đây hoặc nhấn để chọn",
    supportedFormats: "Hỗ trợ: PDF, DOCX, TXT, MD",
    activeDocument: "Tài liệu đang dùng",
    noDocuments: "Chưa có tài liệu nào",
    uploadFirst: "Vui lòng tải lên tài liệu trước khi đặt câu hỏi",
  },

  // Chat composer
  chat: {
    placeholder: "Đặt câu hỏi về tài liệu của bạn...",
    send: "Gửi",
    sending: "Đang gửi...",
    disabled: "Vui lòng tải lên và xử lý tài liệu trước",
    submitHint: "Enter để gửi • Shift+Enter để xuống dòng",
  },

  // Answer panel
  answer: {
    title: "Câu trả lời",
    loading: "Đang tạo câu trả lời...",
    error: "Lỗi",
    noAnswer: "Chưa có câu trả lời. Hãy đặt câu hỏi để bắt đầu.",
    confidence: "Độ tin cậy",
    latency: "Thời gian",
    status: "Trạng thái",
    stopReason: "Lý do dừng",
    loopCount: "Số vòng lặp",
    languageMismatch: "Có thể sai ngôn ngữ",
    noCitationWarning: "Không có trích dẫn",
    hallucinationWarning: "Có dấu hiệu suy diễn ngoài tài liệu",
    llmFallbackWarning: "Đã dùng fallback LLM",
  },

  // Citations
  citations: {
    title: "Trích dẫn",
    noCitations: "Không có trích dẫn",
    source: "Nguồn",
    page: "Trang",
    section: "Phần",
    chunk: "Đoạn",
  },

  // Sources panel
  sources: {
    title: "Nguồn tài liệu",
    noSources: "Chưa có nguồn tài liệu",
    runQuery: "Chạy câu hỏi để xem nguồn tài liệu được truy xuất",
    retrieved: "đã truy xuất",
    rerank: "điểm xếp hạng",
  },

  // Workflow trace
  trace: {
    title: "Luồng xử lý",
    noTrace: "Chưa có luồng xử lý",
    runQuery: "Chạy câu hỏi để xem các bước xử lý",
    steps: {
      retrievalGate: "Cổng truy xuất",
      queryRewrite: "Viết lại câu hỏi",
      retrieval: "Truy xuất",
      rerank: "Xếp hạng lại",
      critique: "Đánh giá",
      finalDecision: "Quyết định cuối",
      loop: "Vòng lặp",
    },
    status: {
      success: "Thành công",
      warning: "Cảnh báo",
      error: "Lỗi",
      info: "Thông tin",
    },
  },

  // Metrics
  metrics: {
    confidence: "Độ tin cậy",
    grounded: "Độ bám tài liệu",
    latency: "Thời gian",
    sources: "Nguồn",
    loops: "Vòng lặp",
    notAvailable: "Chưa có",
  },

  // Compare mode
  compare: {
    title: "So sánh",
    standard: "Chuẩn",
    advanced: "Nâng cao",
    comparison: "Kết quả so sánh",
    confidenceDelta: "Chênh lệch độ tin cậy",
    latencyDelta: "Chênh lệch thời gian",
    citationDelta: "Chênh lệch trích dẫn",
    note: "Ghi chú",
  },

  // Common
  common: {
    loading: "Đang tải...",
    error: "Lỗi",
    success: "Thành công",
    cancel: "Hủy",
    confirm: "Xác nhận",
    close: "Đóng",
    save: "Lưu",
    delete: "Xóa",
    edit: "Sửa",
    view: "Xem",
    download: "Tải xuống",
    upload: "Tải lên",
    search: "Tìm kiếm",
    filter: "Lọc",
    sort: "Sắp xếp",
    refresh: "Làm mới",
    back: "Quay lại",
    next: "Tiếp theo",
    previous: "Trước",
    submit: "Gửi",
    reset: "Đặt lại",
  },
} as const;

export type TranslationKey = keyof typeof translations;
