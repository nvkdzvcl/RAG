import type { Citation, ModeResult, QueryResult, SingleMode } from "@/types/chat";

export type CitationPanelItem = {
  panelId: string;
  branchKey: SingleMode;
  branchLabel: string;
  citation: Citation;
  index: number;
};

export function citationPanelId(branchKey: SingleMode, citation: Citation, index: number): string {
  return `${branchKey}-${citation.id}-${index}`;
}

function itemsForMode(result: ModeResult, branchLabel: string): CitationPanelItem[] {
  return result.citations.map((citation, index) => ({
    panelId: citationPanelId(result.mode, citation, index),
    branchKey: result.mode,
    branchLabel,
    citation,
    index,
  }));
}

export function flattenCitationItems(result: QueryResult | null): CitationPanelItem[] {
  if (!result) {
    return [];
  }
  if (result.mode === "compare") {
    return [
      ...itemsForMode(result.standard, "Standard"),
      ...itemsForMode(result.advanced, "Advanced"),
    ];
  }
  return itemsForMode(result, result.mode === "advanced" ? "Advanced" : "Standard");
}

export function citationContext(citation: Citation): string {
  return citation.content || citation.text || citation.snippet || "";
}

export function citationDocumentName(citation: Citation): string {
  return citation.fileName || citation.title || citation.source || "Unknown document";
}

export function citationScore(citation: Citation): number | null {
  return citation.rerankScore ?? citation.score ?? null;
}
