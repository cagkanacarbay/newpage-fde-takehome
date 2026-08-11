"use client";

const SUGGESTED_QUESTIONS = [
  "What did the first-in-human trials of dasatinib + quercetin find?",
  "Does rapamycin extend lifespan more reliably than metformin?",
  "What are the hallmarks of aging, and which are druggable?",
];

export function EmptyState({ onAsk }: { onAsk: (question: string) => void }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-8 px-6 text-center">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">
          What do you want to know about aging?
        </h1>
        <p className="mt-2 text-sm text-muted">
          Answers come from the longevity research corpus, with citations you can open
          at the exact passage.
        </p>
      </div>
      <ul className="flex w-full max-w-xl flex-col gap-2">
        {SUGGESTED_QUESTIONS.map((question) => (
          <li key={question}>
            <button
              type="button"
              onClick={() => onAsk(question)}
              className="w-full rounded-2xl bg-surface px-5 py-3.5 text-left text-sm text-body transition-colors hover:bg-teal-bg hover:text-teal-ink"
            >
              {question}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
