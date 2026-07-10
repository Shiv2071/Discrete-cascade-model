"""Convert LaTeX-style math delimiters in markdown for reliable pandoc PDF."""
import re
import sys
from pathlib import Path


def process_segment(text: str) -> str:
    # Display math \[ ... \] -> $$ ... $$
    def repl_display(m):
        return "$$\n" + m.group(1).strip() + "\n$$"

    text = re.sub(r"\\\[\s*([\s\S]*?)\s*\\\]", repl_display, text)

    # Inline \( ... \) -> $ ... $ (repeat for nested occurrences)
    while True:
        new = re.sub(r"\\\(([\s\S]*?)\\\)", lambda m: "$" + m.group(1) + "$", text, count=1)
        if new == text:
            break
        text = new
    return text


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    raw = src.read_text(encoding="utf-8")
    parts = re.split(r"(```[\s\S]*?```)", raw)
    out = []
    for i, part in enumerate(parts):
        if part.startswith("```"):
            out.append(part)
        else:
            out.append(process_segment(part))
    dst.write_text("".join(out), encoding="utf-8")
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
