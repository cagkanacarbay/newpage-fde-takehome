"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Assistant text rendered as markdown on the design-system typography. */
export function Markdown({ text }: { text: string }) {
  return (
    <div className="text-sm leading-6 text-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0">{children}</p>,
          h1: ({ children }) => (
            <h1 className="mb-2 mt-4 text-lg font-semibold text-ink first:mt-0">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="mb-2 mt-4 text-base font-semibold text-ink first:mt-0">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="mb-1 mt-3 text-sm font-semibold text-ink first:mt-0">{children}</h3>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-ink">{children}</strong>
          ),
          a: ({ children, href }) => (
            <a
              href={href}
              className="font-medium text-teal underline decoration-teal-mid underline-offset-2 hover:text-orange"
            >
              {children}
            </a>
          ),
          ul: ({ children }) => (
            <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>
          ),
          blockquote: ({ children }) => (
            <blockquote className="my-2 rounded-r-xl bg-surface-2 px-4 py-2 text-muted">
              {children}
            </blockquote>
          ),
          code: ({ children, className }) =>
            className ? (
              <code className="font-mono text-[0.85em]">{children}</code>
            ) : (
              <code className="rounded-md bg-surface-2 px-1.5 py-0.5 font-mono text-[0.85em] text-ink">
                {children}
              </code>
            ),
          pre: ({ children }) => (
            <pre className="my-2 overflow-x-auto rounded-xl bg-surface-2 p-4 text-[0.85em] text-ink">
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <div className="my-3 overflow-x-auto rounded-xl bg-surface-2 p-1">
              <table className="w-full border-collapse text-left text-[0.9em]">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="px-3 py-2 font-semibold text-ink">{children}</th>
          ),
          td: ({ children }) => <td className="px-3 py-2 align-top">{children}</td>,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
