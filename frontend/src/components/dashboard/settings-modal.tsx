import { useEffect, useMemo, useState } from "react";
import { X } from "lucide-react";

import type { ApiChunkingMode, ApiRetrievalMode } from "@/api/types";
import { AlertDialog } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export type SettingsChangePayload = {
  chunking: {
    mode: ApiChunkingMode;
    chunkSize: number;
    chunkOverlap: number;
  };
  retrieval: {
    mode: ApiRetrievalMode;
    topK: number;
  };
};

type SettingsModalProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  chunkSize: number;
  chunkOverlap: number;
  retrievalMode: ApiRetrievalMode;
  topK: number;
  rerankTopN: number;
  contextTopK: number;
  uploadedDocumentsCount: number;
  onSettingsChange: (payload: SettingsChangePayload) => void | Promise<void>;
};

type ChunkStrategy = {
  key: ApiChunkingMode;
  label: string;
  description: string;
  chunkSize: number | null;
  chunkOverlap: number | null;
};

type RetrievalStrategy = {
  key: ApiRetrievalMode;
  label: string;
  description: string;
  topK: number | null;
};

const CHUNK_STRATEGIES: ChunkStrategy[] = [
  {
    key: "small",
    label: "Small",
    description: "500 / 50 • truy xuất chi tiết hơn",
    chunkSize: 500,
    chunkOverlap: 50,
  },
  {
    key: "medium",
    label: "Medium",
    description: "1000 / 100 • cân bằng cho đa số tài liệu",
    chunkSize: 1000,
    chunkOverlap: 100,
  },
  {
    key: "large",
    label: "Large",
    description: "1500 / 200 • ngữ cảnh rộng hơn, ít chunk hơn",
    chunkSize: 1500,
    chunkOverlap: 200,
  },
  {
    key: "custom",
    label: "Custom",
    description: "Tự cấu hình chunk size/overlap",
    chunkSize: null,
    chunkOverlap: null,
  },
];

const RETRIEVAL_STRATEGIES: RetrievalStrategy[] = [
  {
    key: "low",
    label: "Low",
    description: "top_k=3 • nhanh, ít ngữ cảnh",
    topK: 3,
  },
  {
    key: "balanced",
    label: "Balanced",
    description: "top_k=5 • cân bằng mặc định",
    topK: 5,
  },
  {
    key: "high",
    label: "High",
    description: "top_k=8 • nhiều ngữ cảnh hơn",
    topK: 8,
  },
  {
    key: "custom",
    label: "Custom",
    description: "Tự cấu hình top_k",
    topK: null,
  },
];

export function SettingsModal({
  open,
  onOpenChange,
  chunkSize,
  chunkOverlap,
  retrievalMode,
  topK,
  rerankTopN,
  contextTopK,
  uploadedDocumentsCount,
  onSettingsChange,
}: SettingsModalProps) {
  const initialChunkStrategyKey = useMemo<ApiChunkingMode>(() => {
    const matched = CHUNK_STRATEGIES.find(
      (strategy) =>
        strategy.key !== "custom" &&
        strategy.chunkSize === chunkSize &&
        strategy.chunkOverlap === chunkOverlap,
    );
    return matched?.key ?? "custom";
  }, [chunkOverlap, chunkSize]);

  const initialRetrievalMode = useMemo<ApiRetrievalMode>(() => {
    if (retrievalMode === "custom") {
      return "custom";
    }
    const matched = RETRIEVAL_STRATEGIES.find(
      (strategy) => strategy.key !== "custom" && strategy.topK === topK,
    );
    return matched?.key ?? retrievalMode;
  }, [retrievalMode, topK]);

  const [selectedChunkStrategyKey, setSelectedChunkStrategyKey] = useState<ApiChunkingMode>(
    initialChunkStrategyKey,
  );
  const [selectedRetrievalKey, setSelectedRetrievalKey] = useState<ApiRetrievalMode>(
    initialRetrievalMode,
  );
  const [localChunkSize, setLocalChunkSize] = useState(chunkSize);
  const [localChunkOverlap, setLocalChunkOverlap] = useState(chunkOverlap);
  const [localTopK, setLocalTopK] = useState(topK);
  const [isApplying, setIsApplying] = useState(false);
  const [showApplyConfirm, setShowApplyConfirm] = useState(false);

  const isCustomChunkMode = selectedChunkStrategyKey === "custom";
  const isCustomRetrievalMode = selectedRetrievalKey === "custom";

  const chunkValidationError = useMemo(() => {
    if (!isCustomChunkMode) {
      return null;
    }
    if (!Number.isFinite(localChunkSize) || localChunkSize < 100 || localChunkSize > 4000) {
      return "chunk_size phải nằm trong khoảng 100 đến 4000.";
    }
    if (!Number.isFinite(localChunkOverlap) || localChunkOverlap < 0 || localChunkOverlap > 1000) {
      return "chunk_overlap phải nằm trong khoảng 0 đến 1000.";
    }
    if (localChunkOverlap >= localChunkSize) {
      return "chunk_overlap phải nhỏ hơn chunk_size.";
    }
    return null;
  }, [isCustomChunkMode, localChunkOverlap, localChunkSize]);

  const retrievalValidationError = useMemo(() => {
    if (!isCustomRetrievalMode) {
      return null;
    }
    if (!Number.isFinite(localTopK) || !Number.isInteger(localTopK)) {
      return "top_k phải là số nguyên.";
    }
    if (localTopK < 1 || localTopK > 20) {
      return "top_k phải nằm trong khoảng 1 đến 20.";
    }
    return null;
  }, [isCustomRetrievalMode, localTopK]);

  const chunkChanged = localChunkSize !== chunkSize || localChunkOverlap !== chunkOverlap;
  const retrievalChanged = selectedRetrievalKey !== retrievalMode || localTopK !== topK;
  const hasChanges = chunkChanged || retrievalChanged;
  const hasValidationError = chunkValidationError !== null || retrievalValidationError !== null;
  const showPerformanceWarning = localChunkSize >= 1500 || localTopK >= 8;

  useEffect(() => {
    if (!open) {
      return;
    }
    setSelectedChunkStrategyKey(initialChunkStrategyKey);
    setSelectedRetrievalKey(initialRetrievalMode);
    setLocalChunkSize(chunkSize);
    setLocalChunkOverlap(chunkOverlap);
    setLocalTopK(topK);
    setShowApplyConfirm(false);
  }, [open, initialChunkStrategyKey, initialRetrievalMode, chunkSize, chunkOverlap, topK]);

  if (!open) return null;

  const applySettings = async () => {
    setIsApplying(true);
    try {
      await onSettingsChange({
        chunking: {
          mode: selectedChunkStrategyKey,
          chunkSize: localChunkSize,
          chunkOverlap: localChunkOverlap,
        },
        retrieval: {
          mode: selectedRetrievalKey,
          topK: localTopK,
        },
      });
      setShowApplyConfirm(false);
      onOpenChange(false);
    } finally {
      setIsApplying(false);
    }
  };

  const handleApply = () => {
    if (hasValidationError || isApplying) {
      return;
    }
    if (!hasChanges) {
      onOpenChange(false);
      return;
    }
    if (uploadedDocumentsCount > 0) {
      setShowApplyConfirm(true);
      return;
    }
    void applySettings();
  };

  const handleCancel = () => {
    if (isApplying) {
      return;
    }
    setSelectedChunkStrategyKey(initialChunkStrategyKey);
    setSelectedRetrievalKey(initialRetrievalMode);
    setLocalChunkSize(chunkSize);
    setLocalChunkOverlap(chunkOverlap);
    setLocalTopK(topK);
    setShowApplyConfirm(false);
    onOpenChange(false);
  };

  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-2 sm:p-4">
        <Card className="flex max-h-[90vh] w-full max-w-3xl flex-col border-slate-200 shadow-lg">
          <CardHeader className="shrink-0 border-b border-slate-100 pb-4">
            <div className="flex flex-row items-center justify-between">
              <CardTitle className="text-lg">Cài đặt hệ thống</CardTitle>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => onOpenChange(false)}
                disabled={isApplying}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>

          <CardContent className="flex min-h-0 flex-1 flex-col p-0">
            <div className="min-h-0 flex-1 space-y-6 overflow-y-auto px-4 pb-4 pt-4 sm:px-6 sm:pr-5">
              <div className="space-y-4">
                <h3 className="text-base font-semibold text-slate-900">Cài đặt Chunking</h3>

                <div className="grid gap-2">
                  {CHUNK_STRATEGIES.map((strategy) => {
                    const active = strategy.key === selectedChunkStrategyKey;
                    return (
                      <button
                        key={strategy.key}
                        type="button"
                        onClick={() => {
                          setSelectedChunkStrategyKey(strategy.key);
                          if (strategy.chunkSize !== null && strategy.chunkOverlap !== null) {
                            setLocalChunkSize(strategy.chunkSize);
                            setLocalChunkOverlap(strategy.chunkOverlap);
                          }
                        }}
                        className={`rounded-xl border px-3 py-3 text-left transition ${
                          active
                            ? "border-blue-300 bg-blue-50 text-slate-900"
                            : "border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50"
                        }`}
                      >
                        <p className="text-sm font-semibold">{strategy.label}</p>
                        <p className="text-xs text-slate-500">{strategy.description}</p>
                      </button>
                    );
                  })}
                </div>

                {isCustomChunkMode ? (
                  <div className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
                    <div className="grid gap-1">
                      <label htmlFor="custom-chunk-size" className="text-xs font-medium text-slate-700">
                        chunk_size (100 - 4000)
                      </label>
                      <input
                        id="custom-chunk-size"
                        type="number"
                        min={100}
                        max={4000}
                        value={localChunkSize}
                        onChange={(event) => {
                          const next = Number(event.target.value);
                          setLocalChunkSize(Number.isFinite(next) ? Math.round(next) : 0);
                        }}
                        className="h-10 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-800 outline-none focus:border-blue-400"
                      />
                    </div>
                    <div className="grid gap-1">
                      <label htmlFor="custom-chunk-overlap" className="text-xs font-medium text-slate-700">
                        chunk_overlap (0 - 1000)
                      </label>
                      <input
                        id="custom-chunk-overlap"
                        type="number"
                        min={0}
                        max={1000}
                        value={localChunkOverlap}
                        onChange={(event) => {
                          const next = Number(event.target.value);
                          setLocalChunkOverlap(Number.isFinite(next) ? Math.round(next) : 0);
                        }}
                        className="h-10 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-800 outline-none focus:border-blue-400"
                      />
                    </div>
                    {chunkValidationError ? (
                      <p className="text-xs font-medium text-rose-600">{chunkValidationError}</p>
                    ) : null}
                  </div>
                ) : null}

                <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                  <p className="text-xs text-slate-600">
                    Cấu hình hiện tại (áp dụng cho truy vấn mới): chunk size <strong>{chunkSize}</strong>,
                    overlap <strong>{chunkOverlap}</strong>.
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-base font-semibold text-slate-900">Cài đặt Truy xuất (Retrieval)</h3>

                <div className="grid gap-2">
                  {RETRIEVAL_STRATEGIES.map((strategy) => {
                    const active = strategy.key === selectedRetrievalKey;
                    return (
                      <button
                        key={strategy.key}
                        type="button"
                        onClick={() => {
                          setSelectedRetrievalKey(strategy.key);
                          if (typeof strategy.topK === "number") {
                            setLocalTopK(strategy.topK);
                          }
                        }}
                        className={`rounded-xl border px-3 py-3 text-left transition ${
                          active
                            ? "border-blue-300 bg-blue-50 text-slate-900"
                            : "border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50"
                        }`}
                      >
                        <p className="text-sm font-semibold">{strategy.label}</p>
                        <p className="text-xs text-slate-500">{strategy.description}</p>
                      </button>
                    );
                  })}
                </div>

                {isCustomRetrievalMode ? (
                  <div className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
                    <div className="grid gap-1">
                      <label htmlFor="custom-top-k" className="text-xs font-medium text-slate-700">
                        top_k (1 - 20)
                      </label>
                      <input
                        id="custom-top-k"
                        type="number"
                        min={1}
                        max={20}
                        step={1}
                        value={localTopK}
                        onChange={(event) => {
                          const next = Number(event.target.value);
                          setLocalTopK(Number.isFinite(next) ? next : 0);
                        }}
                        className="h-10 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-800 outline-none focus:border-blue-400"
                      />
                    </div>
                    {retrievalValidationError ? (
                      <p className="text-xs font-medium text-rose-600">{retrievalValidationError}</p>
                    ) : null}
                  </div>
                ) : null}

                <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                  <p className="text-xs text-slate-600">
                    Pipeline hiện tại: Retrieve top_k=<strong>{topK}</strong> → Rerank giữ{" "}
                    <strong>{rerankTopN}</strong> → Context chọn <strong>{contextTopK}</strong>
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    Top-k càng lớn giúp tăng khả năng tìm thấy thông tin, nhưng có thể làm chậm và tăng nhiễu.
                  </p>
                </div>
              </div>

              {showPerformanceWarning ? (
                <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
                  <p className="text-xs text-amber-800">
                    Cấu hình lớn có thể làm chậm quá trình re-index hoặc truy vấn.
                  </p>
                </div>
              ) : null}
            </div>

            <div className="shrink-0 border-t border-slate-200 bg-white px-4 py-4 sm:px-6">
              <p className="mb-3 text-xs text-slate-500">
                Khi áp dụng, hệ thống có thể re-index toàn bộ tài liệu đã tải nếu cấu hình thay đổi.
              </p>
              <div className="flex justify-end gap-3">
                <Button type="button" variant="outline" onClick={handleCancel} disabled={isApplying}>
                  Hủy
                </Button>
                <Button
                  type="button"
                  onClick={handleApply}
                  disabled={isApplying || hasValidationError}
                  className="bg-primary text-white hover:bg-primary/90"
                >
                  {isApplying ? "Đang áp dụng..." : "Áp dụng"}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <AlertDialog
        open={showApplyConfirm}
        onOpenChange={setShowApplyConfirm}
        title="Áp dụng cấu hình mới?"
        description="Hệ thống sẽ re-index toàn bộ tài liệu đã tải. Quá trình này có thể mất thời gian."
        onConfirm={applySettings}
        confirmText={isApplying ? "Đang áp dụng..." : "Xác nhận áp dụng"}
        cancelText="Hủy"
        confirmDisabled={isApplying}
        cancelDisabled={isApplying}
      />
    </>
  );
}
