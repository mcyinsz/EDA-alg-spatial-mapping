"""Generate report.html from report.md for browser-based PDF export."""
import markdown
import re
from pathlib import Path

ROOT = Path(__file__).parent

md_path = ROOT / "report.md"
html_path = ROOT / "report.html"

with open(md_path, encoding="utf-8") as f:
    md_text = f.read()

# Convert markdown to HTML (tables, fenced code)
html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

full_html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page { size: A4; margin: 2cm; }
body { font-family: "Microsoft YaHei", "SimSun", serif; font-size: 11pt; line-height: 1.8; color: #222; }
h1 { font-size: 18pt; text-align: center; margin-top: 40px; }
h2 { font-size: 15pt; border-bottom: 2px solid #333; padding-bottom: 6px; margin-top: 30px; }
h3 { font-size: 13pt; margin-top: 20px; }
h4 { font-size: 12pt; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }
th, td { border: 1px solid #999; padding: 5px 8px; text-align: center; }
th { background: #e8e8e8; font-weight: bold; }
td { text-align: right; }
td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; margin: 8px 0; }
blockquote { border-left: 4px solid #aaa; padding-left: 12px; color: #555; margin: 10px 0; background: #fafafa; }
code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 10pt; }
em { font-style: italic; }
strong { font-weight: bold; }
hr { border: none; border-top: 1px solid #ccc; margin: 20px 0; }
</style>
</head>
<body>
""" + html_body + """
</body>
</html>"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"HTML generated: {html_path}")
print("Open in browser, then Ctrl+P to save as PDF.")
