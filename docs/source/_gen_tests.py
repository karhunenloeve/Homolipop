from __future__ import annotations

from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parents[1]
TESTS = ROOT / "tests"
OUT = DOCS / "auto_tests"
OUT.mkdir(parents=True, exist_ok=True)

index_lines = [
    "Tests",
    "=====",
    "",
    ".. toctree::",
    "   :maxdepth: 1",
    "",
]

# Generated files live in docs/source/auto_tests/, so go up three levels to project root.
LITERAL_PREFIX = "../../../tests"

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
                f".. literalinclude:: {LITERAL_PREFIX}/{py.name}",
                "   :language: python",
                "   :linenos:",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    index_lines.append(f"   {stem}")

(OUT / "index.rst").write_text("\n".join(index_lines) + "\n", encoding="utf-8")