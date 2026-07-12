#!/usr/bin/env python3
"""Export a Markdown figure book as a single self-contained HTML file."""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
from pathlib import Path


IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CODE_RE = re.compile(r"`([^`]+)`")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path("benchmark/workspace/final_paper_ready_figure_book_20260629/report.md"),
        help="Markdown report with local image links.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("benchmark/workspace/final_paper_ready_single_document_20260630"),
        help="Output directory for the standalone document.",
    )
    parser.add_argument(
        "--output-name",
        default="scale_pdc_vs_external_baselines_full_report.html",
        help="Standalone HTML file name.",
    )
    return parser.parse_args()


def inline_markdown(text: str) -> str:
    escaped = html.escape(text)

    def code_sub(match: re.Match[str]) -> str:
        return f"<code>{match.group(1)}</code>"

    escaped = CODE_RE.sub(code_sub, escaped)

    def link_sub(match: re.Match[str]) -> str:
        label = match.group(1)
        target = html.escape(match.group(2), quote=True)
        return f'<a href="{target}">{label}</a>'

    escaped = LINK_RE.sub(link_sub, escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def image_data_uri(report_dir: Path, rel_path: str) -> tuple[str | None, str | None]:
    path = (report_dir / rel_path).resolve()
    if not path.exists():
        return None, f"Missing image: {rel_path}"
    mime, _ = mimetypes.guess_type(path.name)
    if path.suffix.lower() == ".svg":
        mime = "image/svg+xml"
    if not mime:
        mime = "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}", None


def render_table(lines: list[str]) -> str:
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""

    head, body = rows[0], rows[1:]
    html_lines = ["<table>", "<thead><tr>"]
    html_lines.extend(f"<th>{inline_markdown(cell)}</th>" for cell in head)
    html_lines.append("</tr></thead>")
    if body:
        html_lines.append("<tbody>")
        for row in body:
            html_lines.append("<tr>")
            html_lines.extend(f"<td>{inline_markdown(cell)}</td>" for cell in row)
            html_lines.append("</tr>")
        html_lines.append("</tbody>")
    html_lines.append("</table>")
    return "\n".join(html_lines)


def markdown_to_html(markdown: str, report_dir: Path) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    i = 0
    in_ul = False

    def close_ul() -> None:
        nonlocal in_ul
        if in_ul:
            output.append("</ul>")
            in_ul = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            close_ul()
            i += 1
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            close_ul()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            output.append(render_table(table_lines))
            continue

        image_match = IMAGE_RE.match(stripped)
        if image_match:
            close_ul()
            alt, rel_path = image_match.groups()
            data_uri, error = image_data_uri(report_dir, rel_path)
            if error:
                output.append(f'<p class="missing">{html.escape(error)}</p>')
            else:
                output.append(
                    '<figure>'
                    f'<img src="{data_uri}" alt="{html.escape(alt, quote=True)}">'
                    f'<figcaption>{html.escape(alt)}</figcaption>'
                    '</figure>'
                )
            i += 1
            continue

        if stripped.startswith("#"):
            close_ul()
            hashes = len(stripped) - len(stripped.lstrip("#"))
            level = min(max(hashes, 1), 4)
            title = stripped[hashes:].strip()
            output.append(f"<h{level}>{inline_markdown(title)}</h{level}>")
            i += 1
            continue

        if stripped.startswith("- "):
            if not in_ul:
                output.append("<ul>")
                in_ul = True
            output.append(f"<li>{inline_markdown(stripped[2:].strip())}</li>")
            i += 1
            continue

        close_ul()
        output.append(f"<p>{inline_markdown(stripped)}</p>")
        i += 1

    close_ul()
    return "\n".join(output)


def write_html(body: str, out_path: Path, source_report: Path) -> None:
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SCALE+PDC vs External Baselines: Full Report</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #1f2933;
      --muted: #52616b;
      --border: #d9e2ec;
      --bg: #ffffff;
      --soft: #f5f7fa;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    main {{
      width: min(1180px, calc(100vw - 48px));
      margin: 32px auto 72px;
    }}
    h1, h2, h3, h4 {{
      line-height: 1.2;
      margin: 2rem 0 0.75rem;
    }}
    h1 {{
      font-size: 2.2rem;
      border-bottom: 2px solid var(--accent);
      padding-bottom: 0.7rem;
    }}
    h2 {{
      font-size: 1.55rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.35rem;
    }}
    h3 {{
      font-size: 1.18rem;
    }}
    p, li {{
      font-size: 0.98rem;
    }}
    code {{
      background: var(--soft);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 0.08rem 0.28rem;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.92em;
    }}
    a {{
      color: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0 1.4rem;
      font-size: 0.92rem;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 0.5rem 0.6rem;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: var(--soft);
      font-weight: 650;
    }}
    figure {{
      margin: 1.4rem 0 2rem;
      padding: 0;
      break-inside: avoid;
    }}
    img {{
      display: block;
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border);
      background: white;
    }}
    figcaption {{
      color: var(--muted);
      font-size: 0.86rem;
      margin-top: 0.45rem;
    }}
    .source {{
      color: var(--muted);
      font-size: 0.88rem;
      margin-top: -0.3rem;
    }}
    .missing {{
      color: #b42318;
      font-weight: 600;
    }}
    @media print {{
      main {{
        width: auto;
        margin: 16mm;
      }}
      a {{
        color: inherit;
      }}
      img {{
        max-height: 230mm;
      }}
    }}
  </style>
</head>
<body>
<main>
<p class="source">Standalone export generated from {html.escape(str(source_report))}. All figures are embedded in this single HTML file.</p>
{body}
</main>
</body>
</html>
"""
    out_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_report = args.source_report.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    markdown = source_report.read_text(encoding="utf-8")
    body = markdown_to_html(markdown, source_report.parent)
    html_path = outdir / args.output_name
    write_html(body, html_path, source_report)
    print(html_path)


if __name__ == "__main__":
    main()
