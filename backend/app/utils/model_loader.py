import os
import importlib.util
from typing import Any, Dict

def load_model(model_path: str) -> Any:
    """
    Load a machine learning model based on its file extension
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model object
        
    Raises:
        ValueError: If the model format is not supported or loading fails
    """
    file_ext = os.path.splitext(model_path)[1].lower()
    
    try:
        if file_ext == ".pt":
            return _load_pytorch_model(model_path)
        elif file_ext == ".h5":
            return _load_keras_model(model_path)
        elif file_ext == ".onnx":
            return _load_onnx_model(model_path)
        elif file_ext == ".tflite":
            return _load_tflite_model(model_path)
        elif file_ext == ".pkl":
            return _load_sklearn_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {file_ext}")
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")

def _load_pytorch_model(model_path: str) -> Any:
    """Load a PyTorch model"""
    try:
        import torch
        model = torch.load(model_path, map_location=torch.device('cpu'))
        if hasattr(model, 'eval'):
            model.eval()
        return model
    except ImportError:
        raise ImportError("PyTorch is not installed. Please install it to load .pt models.")

def _load_keras_model(model_path: str) -> Any:
    """Load a Keras model"""
    try:
        from tensorflow import keras
        return keras.models.load_model(model_path)
    except ImportError:
        raise ImportError("TensorFlow is not installed. Please install it to load .h5 models.")

def _load_onnx_model(model_path: str) -> Any:
    """Load an ONNX model"""
    try:
        import onnxruntime as ort
        return ort.InferenceSession(model_path)
    except ImportError:
        raise ImportError("ONNX Runtime is not installed. Please install it to load .onnx models.")

def _load_tflite_model(model_path: str) -> Any:
    """Load a TensorFlow Lite model"""
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except ImportError:
        raise ImportError("TensorFlow is not installed. Please install it to load .tflite models.")

def _load_sklearn_model(model_path: str) -> Any:
    """Load a scikit-learn model"""
    try:
        import pickle
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except ImportError:
        raise ImportError("scikit-learn is not installed. Please install it to load .pkl models.")

def get_model_info(model: Any, model_path: str) -> Dict:
    """
    Get basic information about the model
    
    Args:
        model: The loaded model object
        model_path: Path to the model file
        
    Returns:
        Dictionary with model information
    """
    file_ext = os.path.splitext(model_path)[1].lower()
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    
    model_info = {
        "file_size_mb": round(file_size, 2),
        "format": file_ext,
        "filename": os.path.basename(model_path)
    }
    
    return model_info 