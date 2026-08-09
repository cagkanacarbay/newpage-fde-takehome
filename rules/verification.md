# Frontend verification

> **Scope: actual UI elements only.** Everything in this file applies to the product's
> rendered UI. HTML documentation under `docs/` is exempt — it is never screenshotted,
> never judged by a model, and is excluded from the UI verify hooks. Documentation is
> checked by reading it.

How AI agents verify that generated UI renders correctly and interactions work —
which tool to reach for, and when.

Agents that generate frontend code must verify the result before reporting done. The
rule is simple: **shot-scraper** for any read-only check, **agent-browser** when the
task requires clicking or navigating, **Playwright** to catch runtime JS errors.

## 1. The three tools

| Tool | What it does that the others cannot | Install |
|---|---|---|
| **shot-scraper** | One-liner CLI screenshots and JS extraction. `--selector` crops to one element. `javascript` subcommand returns DOM data as text with zero image tokens. | `uv tool install shot-scraper && shot-scraper install` |
| **Playwright (Python)** | Console and page-error hooks — the only way to catch runtime JS errors during load. Also network interception, PDF export. | `uv add playwright && playwright install` |
| **agent-browser** | Stateful session: open → click → navigate → verify. Accessibility-tree snapshots give page structure as text (~200 tokens) without a screenshot. Use for any interaction check. | `npm i -g agent-browser && agent-browser install` |

## 2. The rule

> **After building or editing any UI element, take a screenshot and review it before
> reporting done.** Do not claim UI works without visual confirmation. The test suite
> checks logic; only a screenshot confirms rendering.

Three decision points:

1. **Static element or layout changed** → shot-scraper screenshot of the affected element.
2. **Text content or structure changed** → shot-scraper javascript check (cheaper than a screenshot).
3. **Interaction changed** (click, form submit, navigation, toggle) → agent-browser
   session that performs the interaction and re-snapshots the result.

JS console errors matter. Add a Playwright error check whenever a JS-heavy page is
touched. A page that looks correct but throws silent errors will fail in production.

## 3. The Stop-hook gate (automatic)

A Stop hook enforces this rule on both Claude Code and Codex. When the agent tries to
finish a turn, the gate refuses to let it stop if the session changed UI without ever
looking at it. It fires only when **all three** conditions hold:

1. A UI file (`.html .css .js .jsx .tsx .vue .svelte`) outside `docs/` was changed this
   session. Paths under `docs/` are filtered out before anything else runs.
2. The change altered the DOM/visual skeleton — tags, nesting, classes, `style`, or
   `id` — not just the text between tags. A pure copy edit is skipped. Any `.css`/`.js`
   change always counts. The classifier is `.claude/hooks/ui_structural_diff.py`.
3. No `shot-scraper`, `agent-browser`, or `playwright` command actually ran this session.

When it blocks, it names the changed files and a target path and points back here for
the how. To clear it, run the cheapest check below that covers what changed, then
finish. If the change really was text-only and the classifier over-fired, running any
check still clears it; the gate nudges at most once per session before giving way.

Hooks: `.claude/hooks/ui-verify-gate.sh` (exit 2) and `.codex/hooks/ui-verify-gate.sh`
(JSON `decision:block`), wired in `.claude/settings.json` and `.codex/config.toml`. The
Claude gate scopes to files edited this session via the transcript; the Codex gate uses
the git working tree, so it is slightly broader. Neither fires on non-UI sessions.

## 4. Commands

The examples use a served page (`http://localhost:<port>/<path>`). For a static HTML
file opened directly, pass shot-scraper the **plain file path** — a `file://` URL is
rewritten to `http://file//…` and fails to resolve.

### Content check — zero image tokens

Confirm text, headings, section count, or absence of error elements. Returns JSON; no
screenshot taken.

```bash
shot-scraper javascript http://localhost:<port>/<path> "
JSON.stringify({
  title: document.title,
  h1: document.querySelector('h1')?.textContent?.trim(),
  sections: Array.from(document.querySelectorAll('h2')).map(e => e.textContent.trim()),
  error_elements: document.querySelectorAll('.error,[class*=error]').length
})"
```

### Element screenshot — targeted, cheapest visual check

Screenshot only the affected component.

```bash
shot-scraper http://localhost:<port>/<path> --selector ".my-component" --padding 10 -o /tmp/check.png
```

### Full-page screenshot — layout review

Use 800px width to keep token cost down. Reserve 1280px for when wide-layout behaviour
is specifically what changed.

```bash
shot-scraper http://localhost:<port>/<path> --width 800 -o /tmp/full.png
```

### JS error capture — Playwright

Run after any change to JS-heavy pages. Returns a text list; no image tokens.

```bash
python3 - <<'EOF'
from playwright.sync_api import sync_playwright
errors = []
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page()
    pg.on("console", lambda m: errors.append(f"[{m.type}] {m.text}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
    pg.goto("http://localhost:<port>/<path>", wait_until="networkidle")
    b.close()
print(f"JS errors: {len(errors)}")
for e in errors:
    print(" ", e)
EOF
```

### Interaction check — agent-browser

Use when a click, form submit, modal, or navigation is part of what changed. Always
call `wait --load networkidle` before screenshotting — without it the screenshot comes
out black.

```bash
agent-browser open http://localhost:<port>/<path>
agent-browser wait --load networkidle
agent-browser snapshot -i -c                  # verify structure before interacting (~200 tokens)
agent-browser click @e3                       # ref from snapshot
agent-browser wait --load networkidle
agent-browser snapshot -i -c                  # verify result
agent-browser screenshot /tmp/after-click.png # visual confirmation if needed
agent-browser close --all
```

## 5. What to look for when reviewing a screenshot

- Layout is not broken: no overflowing text, no missing sections, no collapsed containers.
- Typography renders: fonts loaded, weights correct, no fallback sans-serif where a
  custom font is expected.
- Colors match the design: no missing CSS variables, no dark/light mode bleed.
- Interactive elements are visible: buttons, inputs, and links present and not hidden.
- No visible error text: no stack traces, "undefined", "NaN", or placeholder copy left in.

## 6. Token cost reference

Image tokens are charged by pixel count, not file size. JPEG compression does not
reduce token cost.

| Method | Approx tokens |
|---|---|
| shot-scraper javascript (text output) | ~150 |
| agent-browser snapshot -c (text output) | ~200–400 |
| shot-scraper --selector (single element) | ~80–200 |
| Screenshot at 800×600 | ~640 |
| Screenshot at 1280×800 | ~800 |
| Full-page screenshot | ~1,200 |
