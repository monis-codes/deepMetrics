import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import Dict
import config
from utils.model_loader import load_model
from utils.benchmark import benchmark_model
from utils.pdf_generator import generate_report

router = APIRouter()

# Store uploaded model paths by session id
model_sessions = {}


@router.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    """
    Upload an AI model file (.pt, .h5, .onnx, .tflite, .pkl)
    """
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in config.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {config.SUPPORTED_FORMATS}"
        )
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    # Save the uploaded file
    file_path = os.path.join(config.UPLOAD_DIR, f"{session_id}{file_ext}")
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Store model path for this session
    model_sessions[session_id] = {
        "model_path": file_path,
        "model_name": file.filename,
        "report_path": None
    }
    
    return {"session_id": session_id, "message": "Model uploaded successfully"}


@router.post("/benchmark/{session_id}")
async def run_benchmark(session_id: str, background_tasks: BackgroundTasks):
    """
    Run benchmark on the uploaded model
    """
    if session_id not in model_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    model_path = model_sessions[session_id]["model_path"]
    
    try:
        # Load the model
        model = load_model(model_path)
        
        # Run benchmarking
        metrics = benchmark_model(model, model_path)
        
        # Generate report asynchronously
        report_path = os.path.join(config.REPORT_DIR, f"{session_id}_report.pdf")
        background_tasks.add_task(
            generate_and_save_report, 
            metrics=metrics, 
            model_name=model_sessions[session_id]["model_name"],
            report_path=report_path,
            session_id=session_id
        )
        
        return {
            "message": "Benchmark completed. Report is being generated.",
            "metrics": metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.get("/report/{session_id}")
async def get_report(session_id: str):
    """
    Get the benchmark report for a session
    """
    if session_id not in model_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if model_sessions[session_id]["report_path"] is None:
        raise HTTPException(status_code=404, detail="Report not generated yet")
    
    report_path = model_sessions[session_id]["report_path"]
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=f"{model_sessions[session_id]['model_name']}_benchmark.pdf"
    )


def generate_and_save_report(metrics: Dict, model_name: str, report_path: str, session_id: str):
    """Helper function to generate and save report"""
    generate_report(metrics, model_name, report_path)
    model_sessions[session_id]["report_path"] = report_path 