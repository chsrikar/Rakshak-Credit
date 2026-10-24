from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf"
PUBLIC = ROOT / "frontend" / "public" / "demo-documents"
OUT.mkdir(parents=True, exist_ok=True)
PUBLIC.mkdir(parents=True, exist_ok=True)

INK = colors.HexColor("#27342b")
MUTED = colors.HexColor("#68746b")
ACCENT = colors.HexColor("#486b50")
PALE = colors.HexColor("#edf2e8")


def make_pdf(filename, title, subtitle, rows, note):
    path = OUT / filename
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    c.setFillColor(PALE)
    c.roundRect(42, height - 180, width - 84, 108, 8, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(58, height - 98, "RAKSHAK CREDIT | FICTIONAL DEMO FIXTURE")
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 21)
    c.drawString(58, height - 128, title)
    c.setFont("Helvetica", 10)
    c.setFillColor(MUTED)
    c.drawString(58, height - 148, subtitle)

    y = height - 218
    c.setFillColor(colors.HexColor("#fff4d6"))
    c.roundRect(48, y - 50, width - 96, 46, 5, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#8a3d16"))
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, y - 23, "NOT A REAL DOCUMENT - INVALID FOR KYC OR FINANCIAL USE")
    y -= 84
    for label, value in rows:
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(58, y, label.upper())
        c.setFillColor(INK)
        c.setFont("Helvetica", 12)
        c.drawString(58, y - 17, value)
        y -= 48
    y -= 8
    c.setStrokeColor(colors.HexColor("#d8dfd4"))
    c.line(58, y, width - 58, y)
    y -= 24
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(58, y, "DEMO CONTENT")
    y -= 19
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 10)
    for line in note:
        c.drawString(58, y, line)
        y -= 16

    c.saveState()
    c.translate(width / 2, 145)
    c.rotate(25)
    c.setFillAlpha(0.16)
    c.setFillColor(colors.HexColor("#b31b1b"))
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(0, 0, "SYNTHETIC DEMO ONLY")
    c.restoreState()
    c.setFillAlpha(1)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(48, 32, "Entirely invented sample data. No government, bank, or tax authority issued this page.")
    c.save()
    (PUBLIC / filename).write_bytes(path.read_bytes())


make_pdf(
    "synthetic-demo-pan-identity.pdf",
    "Demo identity reference",
    "Invented profile card; intentionally not formatted as an official PAN card",
    [("Name", "Asha Example (fictional)"), ("Reference", "INVALID-DEMO-PAN-0000"), ("Issue status", "Not issued by any authority")],
    ["Use this file only to exercise the upload control.", "The placeholder is deliberately invalid and is not a PAN number."])
make_pdf(
    "synthetic-demo-bank-statement.pdf",
    "Fictional account activity",
    "Sample Bank is an invented name; no account or institution exists",
    [("Customer", "Asha Example (fictional)"), ("Account reference", "000000000000 (invalid placeholder)"), ("Period", "01-30 September 2026")],
    ["03 Sep 2026   DEMO MARKET CREDIT                    INR 12,000", "12 Sep 2026   DEMO SUPPLIER PAYMENT                 INR  4,250", "24 Sep 2026   DEMO MARKET CREDIT                    INR  9,800", "Balances and transactions are fabricated for interface testing."])
make_pdf(
    "synthetic-demo-tax-summary.pdf",
    "Illustrative income summary",
    "Plain demo worksheet; not an income tax return, filing, or tax certificate",
    [("Taxpayer", "Asha Example (fictional)"), ("Reference", "NOT-FILED-DEMO-0000"), ("Tax year", "Illustrative only - no filing")],
    ["Illustrative annual receipts: INR 240,000 (invented)", "Illustrative expenses: INR 132,000 (invented)", "No tax was calculated, filed, assessed, or paid."])
print("Created 3 fictional sample PDFs in output/pdf and frontend/public/demo-documents")
