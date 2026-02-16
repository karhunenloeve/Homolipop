from __future__ import annotations

import re


_TITLE_RE = re.compile(r"^(?P<title>[^\n]+)\n(?P<underline>[-=~^\"`']+)\n", re.MULTILINE)


def _fix_titles(text: str) -> str:
    def repl(m: re.Match) -> str:
        title = m.group("title").rstrip()
        underline = m.group("underline")[0] * len(title)
        return f"{title}\n{underline}\n"

    return _TITLE_RE.sub(repl, text)


def setup(app):
    def process_docstring(app, what, name, obj, options, lines):
        txt = "\n".join(lines)
        txt = _fix_titles(txt)
        lines[:] = txt.splitlines()

    app.connect("autodoc-process-docstring", process_docstring)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}