import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/yolo11n.onnx"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.40

def preprocess(frame):
    original_h, original_w = frame.shape[:2]

    # Simple resize for this first experiment
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image)

    return image, original_w, original_h

def inspect_model():

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )


    print("Inputs:")
    for item in session.get_inputs():
        print(
            f"Name: {item.name}, "
            f"Shape: {item.shape}, "
            f"Type: {item.type}"
        )


    print("\nOutputs:")
    for item in session.get_outputs():
        print(
            f"Name: {item.name}, "
            f"Shape: {item.shape}, "
            f"Type: {item.type}"
        )

    print("\nProviders:", session.get_providers())








