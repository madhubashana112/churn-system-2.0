"""Paginated, printable reports built from the same stored run as the dashboard."""
from collections import Counter
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak

from churn_platform.application.use_cases.summarize_analysis import dashboard_charts
from churn_platform.presentation.pdf_charts import DashboardChart, legend_rows

def analysis_pdf(run, outcomes, tenant_name, tier, search):
    font_dir = Path(reportlab.__file__).parent / "fonts"
    if "ReportText" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("ReportText", str(font_dir / "Vera.ttf")))
        pdfmetrics.registerFont(TTFont("ReportBold", str(font_dir / "VeraBd.ttf")))
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=44, leftMargin=44,
                           topMargin=55, bottomMargin=48, title="Churn AI - Analysis report")
    styles = {
        "title": ParagraphStyle("title", fontName="ReportBold", fontSize=25, leading=31, spaceAfter=16, textColor=colors.HexColor("#242458")),
        "heading": ParagraphStyle("heading", fontName="ReportBold", fontSize=12, leading=17, spaceBefore=15, spaceAfter=6, keepWithNext=True),
        "body": ParagraphStyle("body", fontName="ReportText", fontSize=9, leading=14, spaceAfter=6, splitLongWords=True),
        "muted": ParagraphStyle("muted", fontName="ReportText", fontSize=8, leading=12, textColor=colors.HexColor("#596477"), spaceAfter=6),
    }
    def paragraph(value, kind="body"):
        # Uploaded text is always literal, never ReportLab markup or a URL fetch.
        text = "".join(c for c in str(value or "-") if ord(c) >= 32 or c in "\n\t")
        return Paragraph(escape(text).replace("\n", "<br/>"), styles[kind])

    story = [paragraph("Customer retention report", "title"), paragraph(tenant_name, "heading"),
             paragraph(f"{run.sector} | Analyzed {run.created_at:%d %b %Y, %H:%M UTC}", "muted"),
             paragraph("Engine: " + ("System model - deterministic local scoring" if run.offline_mode else "Hosted AI model")),
             paragraph(f"Filters: risk tier {tier}; customer ID contains {search!r}" if search else f"Filters: risk tier {tier}; all customer IDs", "muted")]
    counts = Counter(o.prediction.risk_tier for o in outcomes)
    summary = Table([["Included", "Critical", "High", "Medium", "Low"],
                     [str(len(outcomes)), *[str(counts[k]) for k in ("CRITICAL", "HIGH", "MEDIUM", "LOW")]]], colWidths=[101]*5)
    summary.setStyle(TableStyle([("FONTNAME", (0,0),(-1,-1),"ReportText"), ("FONTSIZE",(0,0),(-1,-1),9),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#ecebff")), ("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#242458")),
        ("TOPPADDING",(0,0),(-1,-1),10), ("BOTTOMPADDING",(0,0),(-1,-1),10)]))
    story += [Spacer(1,10), summary, Spacer(1,10), paragraph(f"{len(outcomes)} of {len(run.outcomes)} scored customers included. {run.entities_uploaded} entities uploaded.", "muted")]
    for warning in run.warnings:
        story.append(paragraph("Analysis note: " + warning))
    story.append(paragraph("Scores indicate risk, not confirmed cancellations. Recommendations should be reviewed before contacting customers.", "muted"))
    if not outcomes:
        story.append(paragraph("No customers match the selected filters.", "heading"))
    if outcomes:
        filtered_run = run.model_copy(update={"outcomes": list(outcomes)})
        charts = dashboard_charts(filtered_run)
        ordered = ["tier_mix", "probability_histogram"] + [key for key in charts if key not in {"tier_mix", "probability_histogram"}]
        for key in ordered:
            chart = charts[key]
            story += [PageBreak(), paragraph(chart.title, "heading"),
                      paragraph(chart.subtitle, "muted"),
                      paragraph(f"Selected population: {len(outcomes)} customers. Same filters as this report.", "muted"),
                      DashboardChart(chart)]
            rows = legend_rows(chart)
            if rows:
                legend = Table([[paragraph(label), paragraph(value) if value else ""] for label, value, _ in rows], colWidths=[405,100])
                commands = [("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),12)]
                for index, (_, _, color) in enumerate(rows):
                    commands.append(("LINEBEFORE",(0,index),(0,index),5,color))
                legend.setStyle(TableStyle(commands))
                story += [Spacer(1,12), legend]
            if chart.kind == "bar":
                # Exact values also make charts accessible and printable in grayscale.
                headers = [paragraph("Category")] + [paragraph(d.label) for d in chart.datasets]
                rows = [[paragraph(f"{i+1}. {label}")] + [paragraph(f"{d.values[i]:g}" if i < len(d.values) else "0") for d in chart.datasets] for i,label in enumerate(chart.labels)]
                table = Table([headers]+rows, colWidths=[205]+[300/max(1,len(chart.datasets))]*len(chart.datasets), repeatRows=1)
                table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#ecebff")),("LINEBELOW",(0,0),(-1,-1),.3,colors.HexColor("#e2e5eb"))]))
                story += [Spacer(1,12),table]
        story += [PageBreak(), paragraph("Customer details", "heading")]
    for outcome in outcomes:
        p, action = outcome.prediction, outcome.playbook
        story.append(KeepTogether([paragraph(f"{p.entity_id} | {p.risk_tier} | {p.churn_probability:.1%}", "heading"),
                  paragraph("Risk drivers: " + ("; ".join(p.primary_drivers or []) or p.root_cause or "Not supplied")),
                  paragraph(f"Recommended action: {action.action_type} | Channel: {action.channel}"),
                  paragraph(action.action_payload)]))
    def footer(canvas, document):
        canvas.setFont("ReportText", 8)
        canvas.setFillColor(colors.HexColor("#596477"))
        canvas.drawString(44, 27, "CHURN AI  /  Customer retention")
        canvas.drawRightString(A4[0]-44, 27, f"Page {document.page}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
