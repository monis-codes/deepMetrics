import os
import pytest
from fastapi.testclient import TestClient
from main import app
import shutil
import tempfile
from pathlib import Path

# Create a test client
client = TestClient(app)

# Create temporary directories for testing
@pytest.fixture(scope="module")
def temp_dirs():
    temp_base = tempfile.mkdtemp()
    uploads_dir = os.path.join(temp_base, "uploads")
    reports_dir = os.path.join(temp_base, "reports")
    data_dir = os.path.join(temp_base, "data")

    for dir_path in [uploads_dir, reports_dir, data_dir]:
        os.makedirs(dir_path, exist_ok=True)

    import config
    config.UPLOAD_DIR = uploads_dir
    config.REPORT_DIR = reports_dir
    config.DATA_DIR = data_dir

    yield {
        "base": temp_base,
        "uploads": uploads_dir,
        "reports": reports_dir,
        "data": data_dir
    }

    shutil.rmtree(temp_base)

@pytest.fixture
def dummy_model_file():
    temp_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    temp_file.close()
    with open(temp_file.name, 'wb') as f:
        f.write(b'\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x05dummy\x94\x8c\x05model\x94s.')
    yield temp_file.name
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)

def test_upload_endpoint(temp_dirs, dummy_model_file):
    with open(dummy_model_file, 'rb') as f:
        response = client.post(
            "/upload",
            files={"file": ("dummy_model.pkl", f, "application/octet-stream")}
        )

    assert response.status_code == 200
    assert "session_id" in response.json()
    assert response.json()["message"] == "Model uploaded successfully"

    session_id = response.json()["session_id"]
    uploaded_file = os.path.join(temp_dirs["uploads"], f"{session_id}.pkl")
    assert os.path.exists(uploaded_file)

def test_upload_invalid_format(temp_dirs):
    temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
    temp_file.close()

    with open(temp_file.name, 'w') as f:
        f.write("This is not a model file")

    with open(temp_file.name, 'rb') as f:
        response = client.post(
            "/upload",
            files={"file": ("not_a_model.txt", f, "text/plain")}
        )

    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

    os.unlink(temp_file.name)

def test_benchmark_endpoint(temp_dirs, dummy_model_file):
    with open(dummy_model_file, 'rb') as f:
        upload_response = client.post(
            "/upload",
            files={"file": ("dummy_model.pkl", f, "application/octet-stream")}
        )

    session_id = upload_response.json()["session_id"]

    import sys
    from unittest.mock import patch

    mock_model = {"mock": "model"}
    mock_metrics = {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1_score": 0.94,
        "roc_auc": 0.98,
        "inference_time_ms": 15.5,
        "memory_usage_mb": 120.5,
        "file_size_mb": 10.2,
        "format": ".pkl",
        "filename": "dummy_model.pkl"
    }

    with patch('utils.model_loader.load_model', return_value=mock_model), \
         patch('utils.benchmark.benchmark_model', return_value=mock_metrics):

        benchmark_response = client.post(f"/benchmark/{session_id}")

        assert benchmark_response.status_code == 200
        assert benchmark_response.json()["message"] == "Benchmark completed. Report is being generated."
        assert benchmark_response.json()["metrics"] == mock_metrics

def test_benchmark_invalid_session(temp_dirs):
    response = client.post("/benchmark/invalid_session_id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"

def test_report_endpoint(temp_dirs, dummy_model_file):
    with open(dummy_model_file, 'rb') as f:
        upload_response = client.post(
            "/upload",
            files={"file": ("dummy_model.pkl", f, "application/octet-stream")}
        )

    session_id = upload_response.json()["session_id"]

    mock_model = {"mock": "model"}
    mock_metrics = {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1_score": 0.94,
        "roc_auc": 0.98,
        "inference_time_ms": 15.5,
        "memory_usage_mb": 120.5,
        "file_size_mb": 10.2,
        "format": ".pkl",
        "filename": "dummy_model.pkl"
    }

    report_path = os.path.join(temp_dirs["reports"], f"{session_id}_report.pdf")
    with open(report_path, 'wb') as f:
        f.write(b'dummy PDF content')

    from routes import model_sessions
    model_sessions[session_id]["report_path"] = report_path

    response = client.get(f"/report/{session_id}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert b'dummy PDF content' == response.content

def test_upload_real_model_file(temp_dirs):
    """Test uploading a real local model file (manual integration test)"""
    real_model_path = "/home/yourname/Documents/my_model.pkl"  # Replace with your actual path

    if not os.path.exists(real_model_path):
        pytest.skip("Real model file not found. Skipping manual test.")

    with open(real_model_path, 'rb') as f:
        response = client.post(
            "/upload",
            files={"file": ("my_model.pkl", f, "application/octet-stream")}
        )

    assert response.status_code == 200
    assert "session_id" in response.json()
    assert response.json()["message"] == "Model uploaded successfully"
