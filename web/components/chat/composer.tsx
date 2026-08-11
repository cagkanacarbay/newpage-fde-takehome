"use client";

import { SendHorizontal } from "lucide-react";
import { useEffect, useRef } from "react";

import { Button } from "@/components/ui/button";

const LINE_HEIGHT_PX = 24;
const MAX_LINES = 8;

type ComposerProps = {
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
  onSend: () => void;
};

/** Autogrowing composer: Enter sends, Shift+Enter inserts a newline. */
export function Composer({ value, disabled, onChange, onSend }: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_LINES * LINE_HEIGHT_PX + 24)}px`;
  }, [value]);

  return (
    <div className="bg-bg px-4 pb-4 pt-2 sm:px-6">
      <div className="mx-auto flex w-full max-w-3xl items-end gap-2 rounded-3xl bg-surface p-2">
        <label className="sr-only" htmlFor="composer">
          Ask a research question
        </label>
        <textarea
          id="composer"
          ref={textareaRef}
          rows={1}
          value={value}
          disabled={disabled}
          placeholder={disabled ? "Waiting for the answer…" : "Ask about longevity research"}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSend();
            }
          }}
          className="max-h-56 flex-1 resize-none self-center rounded-2xl bg-transparent px-3 py-2 text-sm leading-6 text-ink outline-none placeholder:text-faint disabled:cursor-not-allowed"
        />
        <Button
          type="button"
          size="icon"
          aria-label="Send message"
          disabled={disabled || value.trim() === ""}
          onClick={onSend}
          className="rounded-full"
        >
          <SendHorizontal className="size-4" />
        </Button>
      </div>
    </div>
  );
}
