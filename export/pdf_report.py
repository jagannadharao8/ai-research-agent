from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
    HRFlowable
)
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4


def generate_pdf_report(
    filename,
    query,
    answer,
    score,
    confidence,
    risk,
    sources
):

    pdf_doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=60,
        bottomMargin=40
    )

    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor("#1f4e79"),
        spaceAfter=20
    )

    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#333333"),
        spaceBefore=15,
        spaceAfter=8
    )

    normal_style = styles["Normal"]

    # ------------------------
    # TITLE
    # ------------------------
    elements.append(Paragraph("Autonomous AI Research Report", title_style))
    elements.append(Paragraph("Created by Jagannadharao", normal_style))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(HRFlowable(width="100%", thickness=1))
    elements.append(Spacer(1, 0.3 * inch))

    # ------------------------
    # QUERY
    # ------------------------
    elements.append(Paragraph("Research Question", section_style))
    elements.append(Paragraph(query, normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # ------------------------
    # RESPONSE
    # ------------------------
    elements.append(Paragraph("AI Generated Response", section_style))
    elements.append(Paragraph(answer.replace("\n", "<br/>"), normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # ------------------------
    # METRICS
    # ------------------------
    elements.append(Paragraph("Evaluation Metrics", section_style))

    metrics = [
        f"Hallucination Score: {score:.2f}%",
        f"Confidence Level: {confidence:.2f}%",
        f"Risk Classification: {risk}"
    ]

    elements.append(
        ListFlowable(
            [ListItem(Paragraph(m, normal_style)) for m in metrics],
            bulletType='bullet'
        )
    )

    elements.append(Spacer(1, 0.2 * inch))

    # ------------------------
    # SOURCES
    # ------------------------
    if sources:
        elements.append(Paragraph("Sources", section_style))

        source_list = []

        for source_doc in sources:   # <-- renamed variable
            citation = source_doc.get("citation", "")
            title = source_doc.get("title", "Untitled")
            url = source_doc.get("url", "")
            source_text = f"[{citation}] {title} - {url}"

            source_list.append(
                ListItem(Paragraph(source_text, normal_style))
            )

        elements.append(ListFlowable(source_list, bulletType='bullet'))

    # FINAL BUILD
    pdf_doc.build(elements)