# ruff Rules

These are the standards a new Python project inherits from `ruff-defaults.toml` in this
folder. Read this before changing any lint threshold.

## The lint standard

Ruff (0.15.17) enforces the active `select` in `ruff-defaults.toml`: a stack-agnostic core
(`E, W, F, I, B, UP, SIM, C4, PIE, RUF, PLE`) plus strong defaults
(`RET, PTH, ARG, PLC0415, TID, PERF, PLW`) and a cherry-picked set of high-value rules from
two otherwise-noisy families. The reasoning for each cherry-pick is summarised below.

### Exception hygiene (tryceratops, cherry-picked)

On: `TRY002` (no bare `Exception`), `TRY004` (TypeError for type checks), `TRY301` (don't
`raise` inside the `try` you catch), `TRY400` (`logging.exception`, not `logging.error`),
`TRY401` (don't repeat the exception object in the log message). Off: `TRY003` and `TRY300` —
opinions that pay off only with a real custom-exception hierarchy.

### Small refactors (Pylint, cherry-picked)

On: `PLR1714` (`x == a or x == b` → `x in (a, b)`), `PLR5501` (`else:`+`if` → `elif`),
`PLR1722` (`sys.exit()` over `exit()`). Off: `PLR2004` (magic-value comparison) — useful idea,
too noisy on ordinary code.

## Complexity guardrails — the dial

Four Pylint metrics cap how large a single function may get. Each is a numeric **dial** set in
`[tool.ruff.lint.pylint]`; the values are Ruff's defaults, made explicit so they are tunable:

| Rule | What it counts | Dial | Default |
|------|----------------|------|---------|
| `PLR0913` | arguments in a function signature | `max-args` | 5 |
| `PLR0912` | branches (`if`/`for`/`while`/`except`…) in a function | `max-branches` | 12 |
| `PLR0915` | statements in a function | `max-statements` | 50 |
| `PLR0911` | `return` statements in a function | `max-returns` | 6 |

These are guardrails, not correctness rules. A function that trips one is usually doing too
much and is a candidate to split — that is the intended first response.

### Policy for changing the dial

**The thresholds change only with human approval.** Do not raise a limit to make a finding go
away, and do not silence it with a blanket `# noqa`.

When the agent hits one of these limits:

1. **First, try to lower the complexity** — extract a helper, collapse branches, group
   arguments into a small dataclass/params object. Most hits are a real signal to refactor.
2. **If the function is legitimately at this complexity** (and refactoring would make it
   worse, not better), **flag the human.** Say which rule, which function, the current count vs.
   the limit, and why you think the limit (not the code) is the problem.
3. **The human decides** whether to raise the dial in `[tool.ruff.lint.pylint]`, refactor, or
   add a narrow per-line `# noqa: PLR0913 — <reason>` for one justified exception.

If the agent keeps running into the **same** limit across many functions, that is itself the
signal to raise: tell the human "we hit `max-args` on N functions this week, here are they, the
limit may be too low for this codebase" — and let them adjust it deliberately. The point of the
dial is that the threshold is a conscious, reviewed choice, not something that drifts upward one
`# noqa` at a time.

## Where the config lives

- Template: `ruff-defaults.toml` (this folder).
- In a project: either `[tool.ruff.lint]` + `[tool.ruff.lint.pylint]` in `pyproject.toml`,
  or a standalone `ruff.toml` at the repo root.
