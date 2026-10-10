from src.config import MODEL_PATH

print(f"Model Path: {MODEL_PATH}")
print(f"Model Exists: {MODEL_PATH.exists()}")

assert MODEL_PATH.exists(), "ONNX Model file not found"

print("Configuration test passed.")