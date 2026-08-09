#!/usr/bin/env python3
"""Classify an HTML edit as structural/visual vs text-only.

Reads the OLD file content on stdin and takes the NEW file path as argv[1].
Prints "structural" when the visual/DOM skeleton differs (tags, nesting order,
attribute *names*, and the values of class/style/id), else "text-only".

Empty stdin (a new file with no HEAD version) or an unparseable document is
treated as "structural" so the gate errs toward asking for a look. Editing the
text *between* existing tags leaves the skeleton untouched and reads as
"text-only" — that is the copy-edit case the UI verify gate intentionally skips.

Shared by the Claude and Codex UI verify Stop hooks. Stdlib only, no deps.
"""

import sys
from html.parser import HTMLParser

# Attribute values that drive layout/appearance/targeting. A change to any of
# these counts as visual; other attribute *values* (href, alt, data-*) do not.
VISUAL_ATTRS = ("class", "style", "id")


class Skeleton(HTMLParser):
    """Collect a structural token stream, ignoring text, comments, whitespace."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tokens = []

    def handle_starttag(self, tag, attrs):
        self._emit("start", tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._emit("startend", tag, attrs)

    def handle_endtag(self, tag):
        self.tokens.append(("end", tag))

    def _emit(self, kind, tag, attrs):
        d = dict(attrs)
        names = tuple(sorted(k for k, _ in attrs))
        visual = tuple((a, d.get(a)) for a in VISUAL_ATTRS if a in d)
        self.tokens.append((kind, tag, names, visual))


def skeleton(content):
    p = Skeleton()
    try:
        p.feed(content)
    except Exception:
        return None  # unparseable -> caller treats as structural
    return p.tokens


def main():
    old = sys.stdin.read()
    if not old.strip():
        print("structural")
        return
    try:
        with open(sys.argv[1], encoding="utf-8", errors="replace") as fh:
            new = fh.read()
    except OSError:
        print("structural")
        return
    so, sn = skeleton(old), skeleton(new)
    print("structural" if (so is None or sn is None or so != sn) else "text-only")


if __name__ == "__main__":
    main()
