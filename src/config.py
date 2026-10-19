from pathlib import Path 

# project paths

PROJECT_ROOT = Path(__name__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "simple_cnn_legacy.onnx"

# Model Input Configuration

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
INPUT_CHANNELS = 3

# ONNX Run Time Configuration 

PROVIDER = "CPUExecutionProvider"

# WebCam Configuration

CAMERA_INDEX = 0 
WINDOW_NAME = "Real-Time-CV-Inference-System"


