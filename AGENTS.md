# AGENTS.md

A fullstack conversational AI assistant built on retrieval-augmented generation, for a
hypothetical customer, Live Long R&D: researchers in longevity science who need an
assistant focused on their domain, one that can parse their field's research reliably
and accurately and surface it with traceable sources.

## The pieces of the puzzle

The assignment requires an explicit choice, with reasoning, for each of these. This
table tracks status; the reasoning is written up in the README, not here.

| Piece | Status |
|---|---|
| Orchestration framework | Decided — LlamaIndex |
| Parsing | Decided — Docling |
| Chunking | Open |
| Embedding model | Open |
| LLM selection | Open |
| Vector database | Open |
| Retrieval approach | Open |
| Prompt engineering | Open |
| Context management | Decided |
| Guardrails | Decided |
| Quality controls | Open |
| Observability | Open |


| Deferred | Created when |
|---|---|
| remote, first push | Cha creates the GitHub repo |
| `pyproject.toml`, `.python-version` | the stack is scaffolded |
| `no-mistakes init`, `treehouse init` | before the first feature branch |
| CI workflow | the first PR needs it |
| `Dockerfile`, `.dockerignore` | packaging becomes real work |

Do not create any of these ahead of their trigger.

## Rule: README ownership

The human updates the README unless otherwise stated.

## How you develop: TDD loop

Red → green, one test at a time, then refactor — repeat. Use the `tdd` skill every time
an application feature is built or an application bug is fixed. Documentation-only
changes do not require application tests. Write a brief before each test; after green,
review the test for weak assertions, internal mocks, and missing counter-examples. The
method lives in the skill — this file does not restate it.

Work in vertical slices: each slice goes end to end through the stack and is
demonstrable on its own, rather than building a whole layer at a time.

## Testing

Use `@pytest.mark.e2e` end-to-end tests for application changes that cross a process or
I/O boundary, and always to reproduce an application bug before fixing it.
Documentation-only changes do not require pytest or end-to-end tests.
Conventions, markers, mocking policy, and the reproduction protocol: `rules/testing.md`.


## Environment

- Python `3.12`, pinned in `.python-version` (created with the stack scaffold).
- Package manager: `uv`. Never call `pip` or bare `python3` for project work, and never
  hand-edit `.venv`. Run things as `uv run <cmd>`.
- Lint/format: `ruff`. Type check: `mypy`, strict. Test runner: `pytest`. The rule set,
  the reasoning behind it, and the complexity dials all live in `rules/ruff-rules.md` —
  this file does not restate them.
- Secrets live in `.env`, gitignored once `.gitignore` exists. Never read, write, or
  print it. When a secret needs setting, say so and let the user set it.

## Project layout

Keep this map current as the repo grows, so it stays a reliable index of where things live.

- `tests/` — top-level, **mirrors the package**: `src/<pkg>/sub/foo.py` →
  `tests/sub/test_foo.py`. No `__init__.py`.
- `docs/` — the HTML documentation set (see below).
- `rules/` — durable rules referenced from this file.
- `.agents/skills/` — vendored skills, symlinked into `.claude/skills/` and `.codex/skills/`.

## Worktrees and PRs

**Every feature gets its own git worktree.** Multiple pieces of work run in parallel
without sharing a checkout and without stepping on each other. Treehouse maintains the
pool of reusable, pre-warmed worktrees:

- `treehouse init` — write the default `treehouse.toml` (one-time, after `git init`)
- `treehouse get` — acquire a pre-warmed worktree and drop into its subshell
- `treehouse status` — show the status of all worktrees in the pool
- `treehouse return` — terminate lingering processes and hand a worktree back
- `treehouse prune` / `treehouse destroy` — clear stale worktrees / remove them from the pool

**Every change ships as a pull request that Cha reviews.** Nothing lands on the main
branch directly — not a one-line fix, not a docs tweak. The PR is the review surface.

## The gate: no-mistakes

Use no-mistakes only for application changes in `src/`, `tests/`, `web/`, or `scripts/`.
Never use it otherwise.

- `no-mistakes init` — install the gate in this repo (one-time; **deferred until git exists**)
- `git push no-mistakes` — push through the gate instead of `origin`
- `no-mistakes` — open the TUI to review findings (`-y` to accept defaults headless)
- `no-mistakes axi` — drive the gate from an autonomous agent
- `no-mistakes status` / `no-mistakes runs` — state of this repo / recent pipeline runs


## Documentation

`docs/index.html` is the entry point and is always current: what the system is, its
architecture, and a link to every sub-document. Each part of the system gets its own
HTML sub-document in `docs/`. Pages load shared assets from `docs/_assets/docs.css` and
`docs/_assets/docs.js`. Inline CSS, inline JavaScript, and `style=` attributes are
defects. Pages must render directly from the local filesystem with no CDN. A sub-doc
not linked from the index is a defect.

The documentation map is:

- `docs/index.html` — entry point for the full documentation set.
- `docs/research/<topic>.html` — complete measured evidence for one topic.
- `docs/decisions.html` — current technical choices, with one section per choice.
- `docs/_assets/` — shared CSS, JavaScript, templates, and documentation checks.

Store measured evidence in `docs/research/<topic>.html`.
Store all current technical choices in `docs/decisions.html`.
Give each choice its own section in that page.
Keep a decision summary brief and link its reasoning to the related research page.
Use expandable details for required configuration, known limits, and conditions that would reopen the choice.
Link both research pages and `docs/decisions.html` from `docs/index.html`.
Research tickets must create their primary artifact directly in `docs/research/` as HTML.
Do not leave a Markdown research file as the only readable result.
Link the HTML page from `docs/index.html` before resolving the ticket.

Start new docs from `docs/_assets/template.html`. Run
`python3 docs/_assets/sync-design-system.py` after a docs change. The automatic docs
asset gate runs `docs/_assets/check-doc-assets.py --check` before an agent finishes.


## References

- `rules/ruff-rules.md` — the ruff rule families enabled and why. Complexity dials
  (`PLR09xx`) change only with human approval; refactor first.
- `rules/testing.md` — testing conventions, e2e policy, bug-reproduction protocol.
- `rules/verification.md` — how to verify rendered UI before reporting done.
