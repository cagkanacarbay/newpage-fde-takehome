import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  bboxToCssRect,
  citationKey,
  documentPdfUrl,
  fallbackCitations,
  linkCitationMarkers,
  numberCitations,
  prettifySlug,
} from "../lib/citations";
import type { Citation } from "../lib/sse";

function makeCitation(overrides: Partial<Citation> = {}): Citation {
  return {
    document_id: "015-hickson-2019-senolytics",
    page: 2,
    heading_path: ["1. Introduction"],
    bbox: { l: 10, t: 700, r: 200, b: 680 },
    snippet: "Dasatinib plus quercetin.",
    ...overrides,
  };
}

describe("numberCitations", () => {
  it("numbers citations from 1 in arrival order", () => {
    const numbered = numberCitations([
      makeCitation({ page: 2 }),
      makeCitation({ page: 5 }),
      makeCitation({ page: 1 }),
    ]);
    assert.deepEqual(
      numbered.map((item) => item.index),
      [1, 2, 3],
    );
    assert.deepEqual(
      numbered.map((item) => item.citation.page),
      [2, 5, 1],
    );
  });

  it("dedupes on document_id + page + bbox, keeping the first occurrence", () => {
    const duplicate = makeCitation({ snippet: "Second copy of the same passage." });
    const numbered = numberCitations([makeCitation(), duplicate, makeCitation({ page: 3 })]);
    assert.equal(numbered.length, 2);
    assert.equal(numbered[0].citation.snippet, "Dasatinib plus quercetin.");
    assert.equal(numbered[1].index, 2);
  });

  it("treats same page with different bbox as distinct citations", () => {
    const numbered = numberCitations([
      makeCitation(),
      makeCitation({ bbox: { l: 10, t: 300, r: 200, b: 280 } }),
    ]);
    assert.equal(numbered.length, 2);
  });

  it("treats same bbox on different documents as distinct", () => {
    const numbered = numberCitations([
      makeCitation(),
      makeCitation({ document_id: "016-justice-2019-trial" }),
    ]);
    assert.equal(numbered.length, 2);
  });

  it("keeps locations distinct when document ID and page boundaries overlap", () => {
    const numbered = numberCitations([
      makeCitation({ document_id: "001-x", page: 23 }),
      makeCitation({ document_id: "001-x2", page: 3 }),
    ]);

    assert.equal(numbered.length, 2);
  });
});

describe("citationKey", () => {
  it("is stable for identical citations", () => {
    assert.equal(citationKey(makeCitation()), citationKey(makeCitation()));
  });
});

describe("documentPdfUrl", () => {
  it("loads the cited document from the same-origin API when no base URL is set", () => {
    assert.equal(
      documentPdfUrl("015-hickson-2019-senolytics", undefined),
      "/api/documents/015-hickson-2019-senolytics/pdf",
    );
  });
});

describe("linkCitationMarkers", () => {
  it("turns answer markers into links that preserve their citation number", () => {
    assert.equal(
      linkCitationMarkers("One finding [1]. A conflict [2][3]."),
      "One finding [[1]](#citation-1). A conflict [[2]](#citation-2)[[3]](#citation-3).",
    );
  });
});

describe("fallbackCitations", () => {
  it("keeps markerless stored citations reachable", () => {
    const citations = [makeCitation(), makeCitation({ page: 3 })];

    assert.deepEqual(fallbackCitations("A legacy saved answer.", citations), [
      { citation: citations[0], index: 1 },
      { citation: citations[1], index: 2 },
    ]);
  });

  it("does not duplicate controls when inline markers are present", () => {
    assert.deepEqual(fallbackCitations("A verified answer [1].", [makeCitation()]), []);
  });

  it("keeps controls when bracketed numbers do not match a citation", () => {
    const citations = [makeCitation(), makeCitation({ page: 3 })];

    assert.deepEqual(fallbackCitations("The 2024 result [2024].", citations), [
      { citation: citations[0], index: 1 },
      { citation: citations[1], index: 2 },
    ]);
  });
});

describe("bboxToCssRect", () => {
  it("flips the bottom-left origin to top-left using the page height", () => {
    const rect = bboxToCssRect({ l: 10, t: 700, r: 210, b: 660 }, 792, 1);
    assert.deepEqual(rect, { left: 10, top: 92, width: 200, height: 40 });
  });

  it("multiplies by the render scale", () => {
    const rect = bboxToCssRect({ l: 10, t: 700, r: 210, b: 660 }, 792, 1.5);
    assert.deepEqual(rect, { left: 15, top: 138, width: 300, height: 60 });
  });
});

describe("prettifySlug", () => {
  it("drops the numeric index and capitalizes", () => {
    assert.equal(
      prettifySlug("015-hickson-2019-senolytics-dasatinib-quercetin-first-in-human"),
      "Hickson 2019 senolytics dasatinib quercetin first in human",
    );
  });

  it("keeps ids without a numeric index", () => {
    assert.equal(prettifySlug("some-document"), "Some document");
  });
});
