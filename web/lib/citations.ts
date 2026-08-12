import type { Citation } from "./sse";

export type NumberedCitation = {
  citation: Citation;
  /** 1-based chip number, in arrival order after dedupe. */
  index: number;
};

/** Identity of a cited passage: same document, same page, same box. */
export function citationKey(citation: Citation): string {
  const { l, t, r, b } = citation.bbox;
  return JSON.stringify([citation.document_id, citation.page, l, t, r, b]);
}

/** Resolve a citation to the real PDF route in same-origin and split deployments. */
export function documentPdfUrl(documentId: string, baseUrl?: string): string {
  const normalizedBaseUrl = baseUrl?.replace(/\/$/, "") ?? "";
  return `${normalizedBaseUrl}/api/documents/${encodeURIComponent(documentId)}/pdf`;
}

/**
 * Dedupe citations on document_id + page + bbox and number them [1]..[N]
 * in arrival order.
 */
export function numberCitations(citations: Citation[]): NumberedCitation[] {
  const seen = new Set<string>();
  const numbered: NumberedCitation[] = [];

  for (const citation of citations) {
    const key = citationKey(citation);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    numbered.push({ citation, index: numbered.length + 1 });
  }

  return numbered;
}

/** Make generated `[n]` markers actionable without changing their visible text. */
export function linkCitationMarkers(text: string): string {
  return text.replace(/\[(\d+)]/g, "[[$1]](#citation-$1)");
}

/** Keep citations from older stored answers reachable when their text has no markers. */
export function fallbackCitations(
  text: string,
  citations: Citation[],
): NumberedCitation[] {
  const numbered = numberCitations(citations);
  const availableMarkers = new Set(numbered.map(({ index }) => String(index)));
  const hasResolvableMarker = [...text.matchAll(/\[(\d+)]/g)].some(
    (match) => match[1] !== undefined && availableMarkers.has(match[1]),
  );
  return hasResolvableMarker ? [] : numbered;
}

export type CssRect = {
  left: number;
  top: number;
  width: number;
  height: number;
};

/**
 * Convert a Docling bbox (origin bottom-left, PDF points) into CSS pixels
 * inside a pdf.js page canvas (origin top-left) rendered at `scale`.
 */
export function bboxToCssRect(
  bbox: Citation["bbox"],
  pageHeightPoints: number,
  scale: number,
): CssRect {
  return {
    left: bbox.l * scale,
    top: (pageHeightPoints - bbox.t) * scale,
    width: (bbox.r - bbox.l) * scale,
    height: (bbox.t - bbox.b) * scale,
  };
}

/** "015-hickson-2019-senolytics" -> "Hickson 2019 senolytics". */
export function prettifySlug(documentId: string): string {
  const words = documentId.split("-");
  const withoutIndex = /^\d+$/.test(words[0] ?? "") ? words.slice(1) : words;
  const joined = withoutIndex.join(" ");
  return joined.charAt(0).toUpperCase() + joined.slice(1);
}
