# AI Model Benchmarking Tool

A lightweight FastAPI web backend for benchmarking AI models. The system allows users to upload pre-trained models, runs inference on them using dummy datasets, and generates a PDF report with performance metrics.

## Features

- Support for multiple model formats (.pt, .h5, .onnx, .tflite, .pkl)
- Performance metrics calculation (accuracy, inference time, memory usage)
- PDF report generation with visualizations
- Simple REST API for model upload and benchmarking

## Setup

1. Clone the repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
uvicorn main:app --reload
```

The application will be available at `http://localhost:8000`.

## API Endpoints

- `POST /upload`: Upload a model file
- `POST /benchmark/{session_id}`: Benchmark the uploaded model
- `GET /report/{session_id}`: Get the generated PDF report

## Example Usage

```python
import requests

# Upload model
with open('model.pt', 'rb') as f:
    response = requests.post('http://localhost:8000/upload', files={'file': f})
    
session_id = response.json()['session_id']

# Run benchmark
benchmark_response = requests.post(f'http://localhost:8000/benchmark/{session_id}')
metrics = benchmark_response.json()['metrics']

# Get report
report_response = requests.get(
    f'http://localhost:8000/report/{session_id}',
    stream=True
)

# Save report
with open('benchmark_report.pdf', 'wb') as f:
    for chunk in report_response.iter_content(chunk_size=8192):
        f.write(chunk)
```

## Project Structure

- `main.py`: FastAPI application entry point
- `config.py`: Configuration settings
- `routes.py`: API route handlers
- `utils/model_loader.py`: Model loading utilities
- `utils/benchmark.py`: Benchmarking functionality
- `utils/pdf_generator.py`: PDF report generation
- `tests/`: Unit tests

## Supported Model Formats

- PyTorch models (.pt)
- TensorFlow/Keras models (.h5)
- ONNX models (.onnx)
- TensorFlow Lite models (.tflite)
- Scikit-learn models (.pkl)

## License

MIT
