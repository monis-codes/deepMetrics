import os
import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie

def generate_report(metrics: Dict[str, Any], model_name: str, output_path: str) -> str:
    """
    Generate a PDF report with model benchmarking results
    
    Args:
        metrics: Dictionary of model metrics
        model_name: Name of the model file
        output_path: Path where to save the PDF report
        
    Returns:
        Path to the generated PDF
    """
    # Create the PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Container for the elements to add to the PDF
    elements = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Add title
    elements.append(Paragraph(f"AI Model Benchmark Report", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Model: {model_name}", subtitle_style))
    elements.append(Spacer(1, 12))
    
    # Add timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated: {timestamp}", normal_style))
    elements.append(Spacer(1, 24))
    
    # Add model information section
    elements.append(Paragraph("Model Information", subtitle_style))
    elements.append(Spacer(1, 12))
    
    # Create a table for model info
    model_info = [
        ["Property", "Value"],
        ["File Format", metrics.get("format", "Unknown")],
        ["File Size", f"{metrics.get('file_size_mb', 0)} MB"],
        ["Inference Time", f"{metrics.get('inference_time_ms', 0)} ms per sample"],
        ["Memory Usage", f"{metrics.get('memory_usage_mb', 0)} MB"]
    ]
    
    # Add model info table
    model_table = Table(model_info, colWidths=[200, 200])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(model_table)
    elements.append(Spacer(1, 24))
    
    # Check if model is classification or regression based on metrics
    is_classification = "accuracy" in metrics
    
    if is_classification:
        # Classification metrics
        elements.append(Paragraph("Classification Performance Metrics", subtitle_style))
        elements.append(Spacer(1, 12))
        
        # Classification metrics table
        class_metrics = [
            ["Metric", "Value"],
            ["Accuracy", f"{metrics.get('accuracy', 0):.4f}"],
            ["Precision", f"{metrics.get('precision', 0):.4f}"],
            ["Recall", f"{metrics.get('recall', 0):.4f}"],
            ["F1 Score", f"{metrics.get('f1_score', 0):.4f}"]
        ]
        
        # Add ROC-AUC if available
        if metrics.get("roc_auc") is not None:
            class_metrics.append(["ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}"])
        
        # Create table
        metrics_table = Table(class_metrics, colWidths=[200, 200])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('BACKGROUND', (0, 1), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(metrics_table)
        
        # Add a bar chart for classification metrics
        elements.append(Spacer(1, 24))
        elements.append(Paragraph("Performance Visualization", subtitle_style))
        elements.append(Spacer(1, 12))
        
        # Create bar chart for metrics
        drawing = Drawing(400, 200)
        data = [
            [metrics.get('accuracy', 0), 
             metrics.get('precision', 0), 
             metrics.get('recall', 0), 
             metrics.get('f1_score', 0)]
        ]
        
        bc = VerticalBarChart()
        bc.x = 50
        bc.y = 50
        bc.height = 125
        bc.width = 300
        bc.data = data
        bc.strokeColor = colors.black
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = 1
        bc.valueAxis.valueStep = 0.1
        bc.categoryAxis.labels.boxAnchor = 'ne'
        bc.categoryAxis.labels.dx = 8
        bc.categoryAxis.labels.dy = -2
        bc.categoryAxis.labels.angle = 30
        bc.categoryAxis.categoryNames = ['Accuracy', 'Precision', 'Recall', 'F1']
        
        drawing.add(bc)
        elements.append(drawing)
        
    else:
        # Regression metrics
        elements.append(Paragraph("Regression Performance Metrics", subtitle_style))
        elements.append(Spacer(1, 12))
        
        # Regression metrics table
        reg_metrics = [
            ["Metric", "Value"],
            ["Mean Squared Error", f"{metrics.get('mean_squared_error', 0):.4f}"],
            ["Root Mean Squared Error", f"{metrics.get('root_mean_squared_error', 0):.4f}"],
            ["Mean Absolute Error", f"{metrics.get('mean_absolute_error', 0):.4f}"],
            ["R² Score", f"{metrics.get('r2_score', 0):.4f}"]
        ]
        
        # Create table
        metrics_table = Table(reg_metrics, colWidths=[200, 200])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('BACKGROUND', (0, 1), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(metrics_table)
        
        # Add a bar chart for regression error metrics
        elements.append(Spacer(1, 24))
        elements.append(Paragraph("Error Metrics Visualization", subtitle_style))
        elements.append(Spacer(1, 12))
        
        # Create bar chart for error metrics
        drawing = Drawing(400, 200)
        data = [
            [metrics.get('mean_squared_error', 0), 
             metrics.get('root_mean_squared_error', 0), 
             metrics.get('mean_absolute_error', 0)]
        ]
        
        bc = VerticalBarChart()
        bc.x = 50
        bc.y = 50
        bc.height = 125
        bc.width = 300
        bc.data = data
        bc.strokeColor = colors.black
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueStep = max(data[0]) / 5
        bc.categoryAxis.labels.boxAnchor = 'ne'
        bc.categoryAxis.labels.dx = 8
        bc.categoryAxis.labels.dy = -2
        bc.categoryAxis.labels.angle = 30
        bc.categoryAxis.categoryNames = ['MSE', 'RMSE', 'MAE']
        
        drawing.add(bc)
        elements.append(drawing)
    
    # Add conclusion section
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("Conclusion", subtitle_style))
    elements.append(Spacer(1, 12))
    
    # Generate conclusion text based on metrics
    if is_classification:
        conclusion_text = f"The model '{model_name}' demonstrates "
        accuracy = metrics.get('accuracy', 0)
        if accuracy > 0.9:
            conclusion_text += "excellent accuracy"
        elif accuracy > 0.7:
            conclusion_text += "good accuracy"
        else:
            conclusion_text += "moderate accuracy"
            
        conclusion_text += f" at {metrics.get('inference_time_ms', 0)} ms per inference."
        conclusion_text += f" Memory usage is {metrics.get('memory_usage_mb', 0)} MB."
    else:
        conclusion_text = f"The regression model '{model_name}' shows "
        r2 = metrics.get('r2_score', 0)
        if r2 > 0.9:
            conclusion_text += "excellent fit"
        elif r2 > 0.7:
            conclusion_text += "good fit"
        else:
            conclusion_text += "moderate fit"
            
        conclusion_text += f" with inference time of {metrics.get('inference_time_ms', 0)} ms per sample."
        conclusion_text += f" Memory usage is {metrics.get('memory_usage_mb', 0)} MB."
    
    elements.append(Paragraph(conclusion_text, normal_style))
    
    # Add disclaimer
    elements.append(Spacer(1, 36))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=normal_style,
        fontSize=8,
        textColor=colors.grey
    )
    disclaimer_text = "Disclaimer: This benchmark was run with synthetic data and should be considered approximate. For production deployment, real-world data specific to your use case should be used for evaluation."
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build the PDF
    doc.build(elements)
    
    return output_path 