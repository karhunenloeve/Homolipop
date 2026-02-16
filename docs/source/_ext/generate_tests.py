# docs/source/_ext/generate_tests.py
from __future__ import annotations

from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rst_title(title: str) -> str:
    return "\n".join([title, "=" * len(title), ""])


def _generate_auto_tests(app) -> None:
    docs = Path(app.confdir).resolve()  # docs/source
    root = docs.parents[1]  # project root
    tests_dir = root / "tests"
    out_dir = docs / "auto_tests"

    if not tests_dir.is_dir():
        raise RuntimeError(f"tests directory not found: {tests_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    index_lines: list[str] = []
    index_lines += _rst_title("Tests").splitlines()
    index_lines += [
        ".. toctree::",
        "   :maxdepth: 1",
        "   :titlesonly:",
        "",
    ]

    patterns = ("test*.py", "*_test.py")
    files: list[Path] = []
    for pat in patterns:
        files += sorted(tests_dir.glob(pat))

    seen: set[Path] = set()
    for py in files:
        if py in seen:
            continue
        seen.add(py)

        if py.name.startswith("_") or py.name == "conftest.py":
            continue

        stem = py.stem

        rst = []
        rst += _rst_title(stem).splitlines()
        rst += [
            f".. literalinclude:: ../../tests/{py.name}",
            "   :language: python",
            "   :linenos:",
            "",
        ]

        _write_text(out_dir / f"{stem}.rst", "\n".join(rst).rstrip() + "\n")
        index_lines.append(f"   {stem}")

    _write_text(out_dir / "index.rst", "\n".join(index_lines).rstrip() + "\n")


def setup(app):
    app.connect("builder-inited", _generate_auto_tests)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}