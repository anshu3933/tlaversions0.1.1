from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
from typing import Dict, Any

def create_lesson_plan_pdf(plan_data: Dict[str, Any]) -> io.BytesIO:
    """Create a formatted PDF from lesson plan data."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph(f"Lesson Plan - {plan_data['timeframe'].title()}", title_style))
    story.append(Spacer(1, 12))

    # Sections
    for section, content in plan_data.items():
        if section not in ['timeframe', 'timestamp', 'source_iep', 'quality_score']:
            # Section header
            story.append(Paragraph(section.replace('_', ' ').title(), styles['Heading2']))
            story.append(Spacer(1, 6))
            
            # Section content
            if isinstance(content, list):
                for item in content:
                    story.append(Paragraph(f"• {item}", styles['Normal']))
            else:
                story.append(Paragraph(str(content), styles['Normal']))
            story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)
    return buffer 