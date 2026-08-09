#!/usr/bin/env python3
"""Validate that docs use the shared external CSS and JavaScript assets."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


DOCS = Path(__file__).resolve().parents[1]
ASSETS = DOCS / "_assets"
INLINE_STYLE = re.compile(r"<style\b|\sstyle\s*=", re.IGNORECASE)
INLINE_SCRIPT = re.compile(r"<script\b(?![^>]*\bsrc\s*=)", re.IGNORECASE)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def asset_reference(page: Path, asset: str) -> str:
    return Path(os.path.relpath(ASSETS / asset, page.parent)).as_posix()


def errors_for(page: Path) -> list[str]:
    html = page.read_text(encoding="utf-8")
    content = HTML_COMMENT.sub("", html)
    css = asset_reference(page, "docs.css")
    js = asset_reference(page, "docs.js")
    errors: list[str] = []

    if INLINE_STYLE.search(content):
        errors.append("contains inline CSS")
    if INLINE_SCRIPT.search(content):
        errors.append("contains inline JavaScript")
    if f'<link rel="stylesheet" href="{css}">' not in html:
        errors.append(f"does not link {css}")
    if f'<script defer src="{js}"></script>' not in html:
        errors.append(f"does not load {js}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate docs without writing")
    parser.parse_args()
    pages = sorted(DOCS.rglob("*.html"))
    failures = [(page, errors_for(page)) for page in pages]
    failures = [(page, errors) for page, errors in failures if errors]
    if not failures:
        print(f"checked {len(pages)} docs pages: external assets only")
        return 0
    for page, errors in failures:
        print(f"{page.relative_to(DOCS)}: {', '.join(errors)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
