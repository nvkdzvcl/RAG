import { FileCode, FileText } from "lucide-react";
import type { LucideIcon } from "lucide-react";

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
  /** Lucide icon component to render inside the tile. */
  icon: LucideIcon;
  /**
   * Inline CSS background (gradient). Used via `style` so Tailwind doesn't
   * need to know every possible gradient value.
   */
  bgStyle: React.CSSProperties;
  /**
   * Tailwind border class – a translucent tint of the tile colour so the
   * edge is visible without being heavy.
   */
  borderClass: string;
};

const VISUALS: Record<FileType, FileTypeVisual> = {
  pdf: {
    label: "PDF",
    ariaLabel: "PDF file",
    icon: FileText,
    bgStyle: { background: "linear-gradient(135deg, #dc2626 0%, #991b1b 100%)" },
    borderClass: "border-red-400/30 dark:border-red-300/20",
  },
  docx: {
    label: "DOCX",
    ariaLabel: "DOCX file",
    icon: FileText,
    bgStyle: { background: "linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%)" },
    borderClass: "border-blue-400/30 dark:border-blue-300/20",
  },
  txt: {
    label: "TXT",
    ariaLabel: "TXT file",
    icon: FileText,
    bgStyle: { background: "linear-gradient(135deg, #64748b 0%, #334155 100%)" },
    borderClass: "border-slate-400/30 dark:border-slate-300/20",
  },
  md: {
    label: "MD",
    ariaLabel: "Markdown file",
    icon: FileCode,
    bgStyle: { background: "linear-gradient(135deg, #7c3aed 0%, #4c1d95 100%)" },
    borderClass: "border-violet-400/30 dark:border-violet-300/20",
  },
  generic: {
    label: "FILE",
    ariaLabel: "File",
    icon: FileText,
    bgStyle: { background: "linear-gradient(135deg, #94a3b8 0%, #475569 100%)" },
    borderClass: "border-slate-400/25 dark:border-slate-300/15",
  },
};

// ─── Component ───────────────────────────────────────────────────────────────

/**
 * A compact, polished file-type tile icon.
 *
 * Renders a filled gradient square (≈36×36 px) with a white lucide icon
 * and a tiny white label (PDF, DOCX, TXT, MD, FILE).  The tile uses a
 * subtle shadow and translucent border so it looks great in both light
 * and dark modes.
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
  const Icon = visual.icon;

  return (
    <div
      className={`inline-flex h-9 w-9 shrink-0 flex-col items-center justify-center gap-0.5 rounded-xl border shadow-sm ${visual.borderClass} ${className}`}
      style={visual.bgStyle}
      role="img"
      aria-label={visual.ariaLabel}
      title={visual.ariaLabel}
    >
      <Icon className="h-3.5 w-3.5 text-white/90" strokeWidth={2.2} />
      <span className="text-[7px] font-extrabold uppercase leading-none tracking-wider text-white/80">
        {visual.label}
      </span>
    </div>
  );
}

/** Re-export the detection utility so consumers can use it independently. */
export { detectFileType };
