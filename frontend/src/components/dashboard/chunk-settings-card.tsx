import { useState } from "react";
import { Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type ChunkSettingsCardProps = {
  chunkSize: number;
  chunkOverlap: number;
  onSettingsChange: (chunkSize: number, chunkOverlap: number) => void;
};

const CHUNK_SIZE_OPTIONS = [500, 1000, 1500, 2000];
const CHUNK_OVERLAP_OPTIONS = [50, 100, 200];

export function ChunkSettingsCard({
  chunkSize,
  chunkOverlap,
  onSettingsChange,
}: ChunkSettingsCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [localChunkSize, setLocalChunkSize] = useState(chunkSize);
  const [localChunkOverlap, setLocalChunkOverlap] = useState(chunkOverlap);

  const handleApply = () => {
    onSettingsChange(localChunkSize, localChunkOverlap);
    setIsExpanded(false);
  };

  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Cài đặt Chunking</CardTitle>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      {isExpanded && (
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Chunk Size (ký tự)
            </label>
            <div className="grid grid-cols-4 gap-2">
              {CHUNK_SIZE_OPTIONS.map((size) => (
                <button
                  key={size}
                  type="button"
                  onClick={() => setLocalChunkSize(size)}
                  className={`rounded-lg border px-3 py-2 text-sm transition ${
                    localChunkSize === size
                      ? "border-primary bg-blue-50 text-slate-900"
                      : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  {size}
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-500">
              Kích thước mỗi đoạn văn bản. Giá trị lớn hơn = ngữ cảnh nhiều hơn nhưng ít chính xác hơn.
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Chunk Overlap (ký tự)
            </label>
            <div className="grid grid-cols-3 gap-2">
              {CHUNK_OVERLAP_OPTIONS.map((overlap) => (
                <button
                  key={overlap}
                  type="button"
                  onClick={() => setLocalChunkOverlap(overlap)}
                  className={`rounded-lg border px-3 py-2 text-sm transition ${
                    localChunkOverlap === overlap
                      ? "border-primary bg-blue-50 text-slate-900"
                      : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  {overlap}
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-500">
              Số ký tự chồng lấp giữa các đoạn. Giúp duy trì ngữ cảnh liên tục.
            </p>
          </div>

          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                setLocalChunkSize(chunkSize);
                setLocalChunkOverlap(chunkOverlap);
                setIsExpanded(false);
              }}
            >
              Hủy
            </Button>
            <Button
              type="button"
              size="sm"
              onClick={handleApply}
              className="bg-primary hover:bg-primary/90 text-white"
            >
              Áp dụng
            </Button>
          </div>

          <div className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2">
            <p className="text-xs text-slate-600">
              <strong>Lưu ý:</strong> Thay đổi cài đặt này sẽ áp dụng cho các tài liệu được tải lên sau này.
              Tài liệu đã tải cần được tải lại để áp dụng cài đặt mới.
            </p>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
