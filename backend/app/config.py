import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
REPORT_DIR = os.environ.get("REPORT_DIR", os.path.join(BASE_DIR, "reports"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))

# Ensure all directories exist
def create_directories():
    """Create necessary directories if they don't exist."""
    for directory in [UPLOAD_DIR, REPORT_DIR, DATA_DIR]:
        os.makedirs(directory, exist_ok=True)
        
# Max upload size (10MB)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

# Supported model formats
SUPPORTED_FORMATS = [".pt", ".h5", ".onnx", ".tflite", ".pkl"] 