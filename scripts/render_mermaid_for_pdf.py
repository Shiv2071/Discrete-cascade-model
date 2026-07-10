"""
Replace ```mermaid ... ``` blocks with embedded PNGs for Pandoc/LaTeX PDF.
Requires: Node + npx (@mermaid-js/mermaid-cli).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figures" / "generated_pdf"
MERMAID_RE = re.compile(r"```mermaid\n([\s\S]*?)```", re.MULTILINE)


def run_mmdc(mmd_path: Path, png_path: Path) -> None:
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        raise RuntimeError("npx not found on PATH (install Node.js)")
    args = [
        npx,
        "--yes",
        "@mermaid-js/mermaid-cli",
        "-i",
        str(mmd_path),
        "-o",
        str(png_path),
        "-b",
        "white",
        "-w",
        "1100",
    ]
    r = subprocess.run(
        args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180000,
        shell=False,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr or "")
        sys.stderr.write(r.stdout or "")
        raise RuntimeError(f"mermaid-cli failed ({r.returncode}) for {mmd_path}")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: render_mermaid_for_pdf.py <input.md> <output.md>", file=sys.stderr)
        sys.exit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    text = src.read_text(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        body = m.group(1).strip()
        if not body:
            return m.group(0)
        count += 1
        mmd = OUT_DIR / f"_src_{count}.mmd"
        png = OUT_DIR / f"diagram_{count}.png"
        mmd.write_text(body + "\n", encoding="utf-8")
        print(f"Rendering diagram {count} -> {png.name} ...")
        run_mmdc(mmd, png)
        rel = f"figures/generated_pdf/diagram_{count}.png"
        return f"\n\n![Diagram {count}]({rel})\n\n"

    new_text = MERMAID_RE.sub(repl, text)
    dst.write_text(new_text, encoding="utf-8")
    print(f"Wrote {dst} ({count} Mermaid diagram(s) -> PNG).")


if __name__ == "__main__":
    main()
