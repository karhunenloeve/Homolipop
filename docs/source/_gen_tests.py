from __future__ import annotations

from pathlib import Path

DOCS = Path(__file__).resolve().parent  # docs/source
ROOT = DOCS.parents[2]  # project root
TESTS = ROOT / "tests"

OUT = DOCS / "auto_tests"
OUT.mkdir(parents=True, exist_ok=True)

index = [
    "Tests",
    "=====",
    "",
    ".. toctree::",
    "   :maxdepth: 1",
    "",
]

for py in sorted(TESTS.glob("test*.py")):
    if py.name.startswith("_"):
        continue
    stem = py.stem
    (OUT / f"{stem}.rst").write_text(
        "\n".join(
            [
                stem,
                "=" * len(stem),
                "",
                f".. literalinclude:: ../../../tests/{py.name}",
                "   :language: python",
                "   :linenos:",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    index.append(f"   {stem}")

(OUT / "index.rst").write_text("\n".join(index) + "\n", encoding="utf-8")