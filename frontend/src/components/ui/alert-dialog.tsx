import * as React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type AlertDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  onConfirm: () => void | boolean | Promise<void | boolean>;
  confirmText?: string;
  cancelText?: string;
  variant?: "default" | "destructive";
  confirmDisabled?: boolean;
  cancelDisabled?: boolean;
};

export function AlertDialog({
  open,
  onOpenChange,
  title,
  description,
  onConfirm,
  confirmText = "Xác nhận",
  cancelText = "Hủy",
  variant = "default",
  confirmDisabled = false,
  cancelDisabled = false,
}: AlertDialogProps) {
  if (!open) return null;

  const handleConfirm = async () => {
    if (confirmDisabled) {
      return;
    }
    try {
      const result = await onConfirm();
      if (result === false) {
        return;
      }
      onOpenChange(false);
    } catch {
      // Keep dialog open when confirm action fails.
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <Card className="w-full max-w-md border-slate-200 shadow-lg">
        <CardHeader>
          <CardTitle className="text-lg">{title}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-slate-600">{description}</p>
          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              disabled={cancelDisabled}
              onClick={() => onOpenChange(false)}
            >
              {cancelText}
            </Button>
            <Button
              type="button"
              disabled={confirmDisabled}
              onClick={handleConfirm}
              className={
                variant === "destructive"
                  ? "bg-rose-600 hover:bg-rose-700 text-white"
                  : "bg-primary hover:bg-primary/90 text-white"
              }
            >
              {confirmText}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
