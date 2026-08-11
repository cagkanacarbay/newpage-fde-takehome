import type { Citation } from "./sse";

export type NumberedCitation = {
  citation: Citation;
  /** 1-based chip number, in arrival order after dedupe. */
  index: number;
};

/** Identity of a cited passage: same document, same page, same box. */
export function citationKey(citation: Citation): string {
  const { l, t, r, b } = citation.bbox;
  return `${citation.document_id}${citation.page}${l},${t},${r},${b}`;
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
