import { FileText } from "lucide-react";

/**
 * Supported file types for the FileTypeIcon component.
 * "generic" is used as a fallback for unknown or unrecognized file types.
 */
export type FileType = "pdf" | "docx" | "txt" | "md" | "generic";

type FileTypeIconProps = {
  /** Original file name, used to detect type by extension. */
  fileName?: string;
  /** MIME type, used as fallback when extension is missing or ambiguous. */
  mimeType?: string;
  /** Explicit file type override. Takes lowest priority after extension and MIME. */
  fileType?: FileType;
  /** Additional Tailwind classes for the outer wrapper. */
  className?: string;
};

// ─── Type detection ──────────────────────────────────────────────────────────

const EXTENSION_MAP: Record<string, FileType> = {
  ".pdf": "pdf",
  ".docx": "docx",
  ".txt": "txt",
  ".md": "md",
  ".markdown": "md",
};

const MIME_MAP: Record<string, FileType> = {
  "application/pdf": "pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
  "text/plain": "txt",
  "text/markdown": "md",
  "text/x-markdown": "md",
};

function detectFileType(
  fileName?: string,
  mimeType?: string,
  fileType?: FileType,
): FileType {
  // 1. Extension-based detection (highest priority)
  if (fileName) {
    const lower = fileName.toLowerCase();
    for (const ext of Object.keys(EXTENSION_MAP)) {
      if (lower.endsWith(ext)) {
        return EXTENSION_MAP[ext];
      }
    }
  }

  // 2. MIME-based detection
  if (mimeType) {
    const mapped = MIME_MAP[mimeType.toLowerCase()];
    if (mapped) {
      return mapped;
    }
  }

  // 3. Explicit override
  if (fileType) {
    return fileType;
  }

  return "generic";
}

// ─── Visual config per type ──────────────────────────────────────────────────

type FileTypeVisual = {
  label: string;
  ariaLabel: string;
  /** Tailwind classes for background, border, and text tint. */
  containerClass: string;
  /** Tailwind class for the label text color (slightly more saturated). */
  labelClass: string;
};

const VISUALS: Record<FileType, FileTypeVisual> = {
  pdf: {
    label: "PDF",
    ariaLabel: "PDF file",
    containerClass:
      "border-red-300/60 bg-red-50 text-red-600 dark:border-red-500/30 dark:bg-red-950/40 dark:text-red-400",
    labelClass: "text-red-700 dark:text-red-300",
  },
  docx: {
    label: "DOCX",
    ariaLabel: "DOCX file",
    containerClass:
      "border-blue-300/60 bg-blue-50 text-blue-600 dark:border-blue-500/30 dark:bg-blue-950/40 dark:text-blue-400",
    labelClass: "text-blue-700 dark:text-blue-300",
  },
  txt: {
    label: "TXT",
    ariaLabel: "TXT file",
    containerClass:
      "border-slate-300/60 bg-slate-50 text-slate-500 dark:border-slate-500/30 dark:bg-slate-800/40 dark:text-slate-400",
    labelClass: "text-slate-600 dark:text-slate-300",
  },
  md: {
    label: "MD",
    ariaLabel: "Markdown file",
    containerClass:
      "border-violet-300/60 bg-violet-50 text-violet-600 dark:border-violet-500/30 dark:bg-violet-950/40 dark:text-violet-400",
    labelClass: "text-violet-700 dark:text-violet-300",
  },
  generic: {
    label: "FILE",
    ariaLabel: "File",
    containerClass:
      "border-border bg-muted/50 text-muted-foreground",
    labelClass: "text-muted-foreground",
  },
};

// ─── Component ───────────────────────────────────────────────────────────────

/**
 * A compact, polished file-type icon badge.
 *
 * Renders a small card (≈36×36px) with a lucide FileText icon and a short
 * label (PDF, DOCX, TXT, MD, FILE) tinted to match the file type.
 *
 * Detects type in order: extension → MIME → explicit prop → generic.
 */
export function FileTypeIcon({
  fileName,
  mimeType,
  fileType,
  className = "",
}: FileTypeIconProps) {
  const resolved = detectFileType(fileName, mimeType, fileType);
  const visual = VISUALS[resolved];

  return (
    <div
      className={`inline-flex h-9 w-9 shrink-0 flex-col items-center justify-center gap-0.5 rounded-lg border ${visual.containerClass} ${className}`}
      role="img"
      aria-label={visual.ariaLabel}
      title={visual.ariaLabel}
    >
      <FileText className="h-3.5 w-3.5" strokeWidth={2} />
      <span
        className={`text-[8px] font-bold uppercase leading-none tracking-wide ${visual.labelClass}`}
      >
        {visual.label}
      </span>
    </div>
  );
}

/** Re-export the detection utility so consumers can use it independently. */
export { detectFileType };
