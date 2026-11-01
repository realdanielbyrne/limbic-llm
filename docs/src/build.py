"""Build docs/architecture.html from the Mermaid sources in docs/src/.

Each diagram is rendered to SVG with mermaid-cli (``mmdc``) using placeholder
colours, which are then rewritten to CSS variables so the inline SVG follows
the page's light/dark theme. No Mermaid runtime is needed to view the page.

Usage:  python docs/src/build.py [--fragment OUT]
  --fragment OUT   also write the page body without the HTML wrapper
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent
DOCS = SRC.parent

# placeholder index (last byte of #0a0bNN) -> CSS variable
PALETTE = {
    0x00: "--d-label", 0x01: "--d-node", 0x02: "--d-node-line", 0x03: "--d-text", 0x04: "--d-edge",
    0x05: "--d-io", 0x06: "--d-io-line", 0x07: "--d-io-text",
    0x08: "--d-store", 0x09: "--d-store-line", 0x0A: "--d-store-text",
    0x0B: "--d-train", 0x0C: "--d-train-line", 0x0D: "--d-train-text",
    0x0E: "--d-pick", 0x0F: "--d-cluster", 0x10: "--d-cluster-line", 0x11: "--d-label",
}


def render(name: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{name}.svg"
        subprocess.run(["mmdc", "-i", str(SRC / f"{name}.mmd"), "-o", str(out),
                        "-c", str(SRC / "mermaid.json"), "-b", "transparent"],
                       check=True, capture_output=True)
        svg = out.read_text()

    var = lambda n: f"var({PALETTE[n]})" if n in PALETTE else None
    svg = re.sub(r"#0a0b([0-9a-fA-F]{2})\b", lambda m: var(int(m.group(1), 16)) or m.group(0), svg)
    svg = re.sub(r"rgba?\(10,\s*11,\s*(\d+)(?:,\s*[\d.]+)?\)",
                 lambda m: var(int(m.group(1))) or m.group(0), svg)
    svg = re.sub(r"hsl\(66,[^)]*\)", "var(--d-node-line)", svg)       # derived from node border
    svg = svg.replace("#f5f4ff", "var(--d-edge)")                       # arrowhead path
    svg = re.sub(r'"?DejaVu Sans"?', "var(--font-ui)", svg)
    svg = svg.replace("my-svg", f"dia-{name}")                          # unique ids per diagram
    width = re.search(r'viewBox="[\d.\-]+ [\d.\-]+ ([\d.]+)', svg).group(1)
    svg = re.sub(r'<svg([^>]*?) width="100%"', r"<svg\1", svg, count=1)
    svg = re.sub(r'style="max-width: [\d.]+px;[^"]*"', f'style="--w: {float(width):.0f}px"', svg, count=1)
    return svg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", type=Path)
    args = ap.parse_args()
    page = (SRC / "architecture.template.html").read_text()
    page = page.replace("{{OVERVIEW_SVG}}", render("overview")).replace("{{BLOCK_SVG}}", render("block"))
    head, body = page.split("<!-- body -->", 1)
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            f"{head.strip()}\n</head>\n<body>\n{body.strip()}\n</body>\n</html>\n")
    (DOCS / "architecture.html").write_text(full)
    if args.fragment:
        args.fragment.write_text(page)
    print(f"wrote {DOCS / 'architecture.html'} ({len(full) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
