"""docs/REPORT_template.md + docs/bayes_section.md -> docs/REPORT.md -> HTML (pandoc, MathML) -> PDF (headless Chromium)."""
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright
D = Path(__file__).resolve().parents[1] / "docs"
t = (D / "REPORT_template.md").read_text()
b = (D / "bayes_section.md").read_text() if (D / "bayes_section.md").exists() else ""
(D / "REPORT.md").write_text(t.replace("@@BAYES@@", b))
css = """body{font-family:Arial,Helvetica,sans-serif;font-size:10pt;line-height:1.38;color:#1a1a1a;max-width:none}
h1{font-size:13.5pt;margin:18pt 0 5pt;border-bottom:1px solid #bbb;padding-bottom:2pt;page-break-after:avoid}
header h1.title{font-size:18pt;border:none;margin-top:0} .author,.date{color:#555;margin:0}
figure{margin:8pt 0;page-break-inside:avoid} img{max-width:100%} figcaption{font-size:8.5pt;color:#444}
table{border-collapse:collapse;font-size:9pt;margin:6pt 0} td,th{border:1px solid #ccc;padding:2pt 6pt} p,li{orphans:3;widows:3}
code{font-size:9pt;background:#f3f3f3;padding:0 2pt}"""
(D / "report.css").write_text(css)
subprocess.run(["pandoc", str(D / "REPORT.md"), "-s", "--mathml", "--resource-path", str(D), "-c", "report.css", "-o", str(D / "REPORT.html"), "--metadata", "pagetitle=apexlab"], check=True)
with sync_playwright() as p:
    br = p.chromium.launch(executable_path="/opt/pw-browsers/chromium") if Path("/opt/pw-browsers/chromium").exists() else p.chromium.launch()
    pg = br.new_page(); pg.goto((D / "REPORT.html").as_uri()); pg.wait_for_timeout(800)
    pg.pdf(path=str(D / "REPORT.pdf"), format="A4", margin=dict(top="18mm", bottom="18mm", left="18mm", right="18mm"), print_background=True, display_header_footer=True,
           header_template="<span></span>", footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#666"><span class="pageNumber"></span> / <span class="totalPages"></span></div>')
    br.close()
