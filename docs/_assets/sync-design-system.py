#!/usr/bin/env python3
"""Reference the shared design system from every doc in docs/.

Docs load their CSS and JavaScript from docs/_assets/. HTML pages must not carry
inline CSS or JavaScript. This command replaces the old generated blocks with
external asset references.

    python3 docs/_assets/sync-design-system.py           # write
    python3 docs/_assets/sync-design-system.py --check    # verify, exit 1 if stale

Every doc must contain these markers, in <head>:

    <!-- ds:css:start -->
    <link rel="stylesheet" href="_assets/docs.css">
    <!-- ds:css:end -->

and before </body>:

    <!-- ds:js:start -->
    <script defer src="_assets/docs.js"></script>
    <!-- ds:js:end -->

A doc without either marker is a defect: it means the page is off-system.
Start new docs from docs/_assets/template.html.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent
DOCS = ASSETS.parent

CSS_BLOCK = re.compile(r"(?P<open><!-- ds:css:start -->).*?(?P<close><!-- ds:css:end -->)", re.S)
JS_BLOCK = re.compile(r"(?P<open><!-- ds:js:start -->).*?(?P<close><!-- ds:js:end -->)", re.S)


def relative_asset(path: Path, asset: str) -> str:
    return Path(os.path.relpath(ASSETS / asset, path.parent)).as_posix()


def render_css(path: Path) -> str:
    return (
        "<!-- ds:css:start -->\n"
        f'<link rel="stylesheet" href="{relative_asset(path, "docs.css")}">\n'
        "<!-- ds:css:end -->"
    )


def render_js(path: Path) -> str:
    return (
        "<!-- ds:js:start -->\n"
        f'<script defer src="{relative_asset(path, "docs.js")}"></script>\n'
        "<!-- ds:js:end -->"
    )


def sync(path: Path) -> tuple[str, str]:
    """Return (new_html, status) for one doc."""
    html = path.read_text(encoding="utf-8")

    if not CSS_BLOCK.search(html) or not JS_BLOCK.search(html):
        return html, "MISSING MARKERS"

    out = CSS_BLOCK.sub(lambda _: render_css(path), html, count=1)
    out = JS_BLOCK.sub(lambda _: render_js(path), out, count=1)

    return out, ("unchanged" if out == html else "updated")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if any doc is stale or unmarked",
    )
    args = ap.parse_args()

    targets = sorted(p for p in DOCS.rglob("*.html") if ASSETS not in p.parents)
    targets += sorted(ASSETS.glob("template.html"))
    if not targets:
        print("no docs found", file=sys.stderr)
        return 1

    stale = False
    for path in targets:
        new, status = sync(path)
        rel = path.relative_to(DOCS.parent)
        if status == "MISSING MARKERS":
            print(f"  !! {rel}: no design-system markers — page is off-system")
            stale = True
            continue
        if status == "updated":
            stale = True
            if not args.check:
                path.write_text(new, encoding="utf-8")
        print(f"  {'stale' if args.check and status == 'updated' else status:>9}  {rel}")

    if args.check and stale:
        print("\ndesign system out of sync — run: python3 docs/_assets/sync-design-system.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
