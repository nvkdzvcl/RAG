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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-3 backdrop-blur-sm dark:bg-black/70">
      <Card className="w-full max-w-md border-border bg-card shadow-soft">
        <CardHeader>
          <CardTitle className="text-lg">{title}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm leading-6 text-muted-foreground">{description}</p>
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
                  ? "bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  : "bg-primary text-primary-foreground hover:bg-primary/90"
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
