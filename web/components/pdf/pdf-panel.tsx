"use client";

import { ChevronLeft, ChevronRight, X, ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";

import { Button } from "@/components/ui/button";
import { bboxToCssRect, documentPdfUrl, prettifySlug } from "@/lib/citations";
import type { Citation } from "@/lib/sse";

// Static export: the worker is served from the app itself, no CDN.
pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 2.5;
const ZOOM_STEP = 0.2;

function clampPage(page: number, numPages: number | null): number {
  if (numPages === null) {
    return Math.max(1, page);
  }
  return Math.min(Math.max(1, page), numPages);
}

type PdfPanelProps = {
  documentId: string;
  citation: Citation;
  onClose: () => void;
};

export default function PdfPanel({ documentId, citation, onClose }: PdfPanelProps) {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "");
  const fileUrl = documentPdfUrl(documentId, baseUrl);

  const [title, setTitle] = useState(() => prettifySlug(documentId));
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState(citation.page);
  const [pageInput, setPageInput] = useState(String(citation.page));
  const [zoom, setZoom] = useState(1);
  const [pageSizePoints, setPageSizePoints] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [containerWidth, setContainerWidth] = useState<number | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const highlightRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Fit-to-width: the page fills the panel at zoom 1; zoom buttons multiply.
  useEffect(() => {
    const container = scrollRef.current;
    if (!container) {
      return;
    }
    const observer = new ResizeObserver(() => {
      setContainerWidth(container.clientWidth - 32);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch(`${baseUrl ?? ""}/api/documents/${encodeURIComponent(documentId)}`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data: { title?: string } | null) => {
        if (!cancelled && data?.title) {
          setTitle(data.title);
        }
      })
      .catch(() => {
        // Slug fallback is already in place.
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, documentId]);

  const goToPage = (page: number) => {
    const next = clampPage(page, numPages);
    setPageNumber(next);
    setPageInput(String(next));
  };

  const commitPageInput = () => {
    const parsed = Number.parseInt(pageInput, 10);
    if (Number.isNaN(parsed)) {
      setPageInput(String(pageNumber));
      return;
    }
    goToPage(parsed);
  };

  const zoomBy = (delta: number) => {
    setZoom((current) =>
      Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number((current + delta).toFixed(2)))),
    );
  };

  const fitScale =
    pageSizePoints !== null && containerWidth !== null && containerWidth > 0
      ? containerWidth / pageSizePoints.width
      : 1;
  const scale = fitScale * zoom;

  const showHighlight = pageNumber === citation.page && pageSizePoints !== null;
  const rect =
    showHighlight && pageSizePoints !== null
      ? bboxToCssRect(citation.bbox, pageSizePoints.height, scale)
      : null;

  return (
    <div className="flex h-full min-h-0 flex-col bg-surface">
      <header className="flex items-center gap-2 px-4 py-3">
        <h2 className="min-w-0 flex-1 truncate text-sm font-semibold text-ink" title={title}>
          {title}
        </h2>

        <div className="flex items-center gap-1 text-sm text-muted">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Previous page"
            disabled={pageNumber <= 1}
            onClick={() => goToPage(pageNumber - 1)}
            className="size-8"
          >
            <ChevronLeft className="size-4" />
          </Button>
          <span className="flex items-center gap-1 whitespace-nowrap text-xs">
            <label className="sr-only" htmlFor="pdf-page-input">
              Page
            </label>
            <input
              id="pdf-page-input"
              value={pageInput}
              onChange={(event) => setPageInput(event.target.value)}
              onBlur={commitPageInput}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  commitPageInput();
                }
              }}
              inputMode="numeric"
              className="w-10 rounded-lg bg-surface-2 px-1.5 py-1 text-center text-xs text-ink outline-none focus-visible:ring-2 focus-visible:ring-teal"
            />
            of {numPages ?? "…"}
          </span>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Next page"
            disabled={numPages !== null && pageNumber >= numPages}
            onClick={() => goToPage(pageNumber + 1)}
            className="size-8"
          >
            <ChevronRight className="size-4" />
          </Button>
        </div>

        <div className="flex items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Zoom out"
            disabled={zoom <= MIN_ZOOM}
            onClick={() => zoomBy(-ZOOM_STEP)}
            className="size-8"
          >
            <ZoomOut className="size-4" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Zoom in"
            disabled={zoom >= MAX_ZOOM}
            onClick={() => zoomBy(ZOOM_STEP)}
            className="size-8"
          >
            <ZoomIn className="size-4" />
          </Button>
        </div>

        <Button
          type="button"
          variant="ghost"
          size="icon"
          aria-label="Close document"
          onClick={onClose}
          className="size-8"
        >
          <X className="size-4" />
        </Button>
      </header>

      <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto bg-surface-2 px-4 py-6">
        {loadError ? (
          <p role="alert" className="mx-auto w-fit rounded-2xl bg-orange-bg px-4 py-3 text-sm text-orange-ink">
            {loadError}
          </p>
        ) : (
          <div className="relative mx-auto w-fit">
            <Document
              file={fileUrl}
              onLoadSuccess={({ numPages: total }) => {
                setNumPages(total);
                goToPage(citation.page);
              }}
              onLoadError={(error) => setLoadError(`Could not load the PDF: ${error.message}`)}
              loading={
                <p className="px-4 py-8 text-center text-sm text-faint">Loading document…</p>
              }
            >
              <Page
                pageNumber={pageNumber}
                scale={scale}
                renderTextLayer={false}
                renderAnnotationLayer={false}
                onLoadSuccess={(page) => {
                  setPageSizePoints({
                    width: page.originalWidth,
                    height: page.originalHeight,
                  });
                  if (page.pageNumber === citation.page) {
                    requestAnimationFrame(() => {
                      highlightRef.current?.scrollIntoView({ block: "center" });
                    });
                  }
                }}
              />
            </Document>
            {rect ? (
              <div
                key={`${citation.page}-${scale}`}
                ref={highlightRef}
                className="citation-highlight"
                style={{
                  left: rect.left,
                  top: rect.top,
                  width: rect.width,
                  height: rect.height,
                }}
              />
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
