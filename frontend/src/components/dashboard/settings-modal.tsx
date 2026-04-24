import { useState } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type SettingsModalProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  chunkSize: number;
  chunkOverlap: number;
  topK: number;
  onSettingsChange: (chunkSize: number, chunkOverlap: number, topK: number) => void;
};

const CHUNK_SIZE_OPTIONS = [500, 1000, 1500, 2000];
const CHUNK_OVERLAP_OPTIONS = [50, 100, 200];
const TOP_K_OPTIONS = [3, 5, 10, 15, 20];

export function SettingsModal({
  open,
  onOpenChange,
  chunkSize,
  chunkOverlap,
  topK,
  onSettingsChange,
}: SettingsModalProps) {
  const [localChunkSize, setLocalChunkSize] = useState(chunkSize);
  const [localChunkOverlap, setLocalChunkOverlap] = useState(chunkOverlap);
  const [localTopK, setLocalTopK] = useState(topK);

  if (!open) return null;

  const handleApply = () => {
    onSettingsChange(localChunkSize, localChunkOverlap, localTopK);
    onOpenChange(false);
  };

  const handleCancel = () => {
    setLocalChunkSize(chunkSize);
    setLocalChunkOverlap(chunkOverlap);
    setLocalTopK(topK);
    onOpenChange(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <Card className="w-full max-w-2xl border-slate-200 shadow-lg">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-lg">Cài đặt hệ thống</CardTitle>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => onOpenChange(false)}
          >
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Chunk Settings Section */}
          <div className="space-y-4">
            <h3 className="text-base font-semibold text-slate-900">Cài đặt Chunking</h3>
            
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-slate-700">
                  Chunk Size (ký tự)
                </label>
                <div className="mt-2">
                  <input
                    type="number"
                    min="100"
                    max="5000"
                    step="50"
                    value={localChunkSize}
                    onChange={(e) => setLocalChunkSize(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                    placeholder="Nhập kích thước chunk (100-5000)"
                  />
                </div>
                <p className="mt-1 text-xs text-slate-500">
                  Kích thước mỗi đoạn văn bản. Giá trị lớn hơn = ngữ cảnh nhiều hơn nhưng ít chính xác hơn.
                </p>
              </div>

              <div>
                <label className="text-sm font-medium text-slate-700">
                  Chunk Overlap (ký tự)
                </label>
                <div className="mt-2">
                  <input
                    type="number"
                    min="0"
                    max="500"
                    step="10"
                    value={localChunkOverlap}
                    onChange={(e) => setLocalChunkOverlap(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                    placeholder="Nhập chunk overlap (0-500)"
                  />
                </div>
                <p className="mt-1 text-xs text-slate-500">
                  Số ký tự chồng lấp giữa các đoạn. Giúp duy trì ngữ cảnh liên tục.
                </p>
              </div>

              <div>
                <label className="text-sm font-medium text-slate-700">
                  Top K (số lượng documents)
                </label>
                <div className="mt-2">
                  <input
                    type="number"
                    min="1"
                    max="50"
                    step="1"
                    value={localTopK}
                    onChange={(e) => setLocalTopK(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                    placeholder="Nhập số lượng documents (1-50)"
                  />
                </div>
                <p className="mt-1 text-xs text-slate-500">
                  Số lượng documents được truy xuất từ vector store. Nhiều hơn = ngữ cảnh phong phú hơn nhưng chậm hơn.
                </p>
              </div>
            </div>

            <div className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2">
              <p className="text-xs text-slate-600">
                <strong>Lưu ý:</strong> Thay đổi cài đặt này sẽ áp dụng cho các tài liệu được tải lên sau này.
                Tài liệu đã tải cần được tải lại để áp dụng cài đặt mới.
              </p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex justify-end gap-3 border-t border-slate-200 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={handleCancel}
            >
              Hủy
            </Button>
            <Button
              type="button"
              onClick={handleApply}
              className="bg-primary hover:bg-primary/90 text-white"
            >
              Áp dụng
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}