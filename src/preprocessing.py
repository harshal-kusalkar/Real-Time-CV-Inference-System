import cv2
import numpy as np 

from src.config import (
    INPUT_CHANNELS, 
    INPUT_HEIGHT, 
    INPUT_WIDTH
)

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV BGR frame to model-ready NCHW float32 input.

    """

    if frame is None:
        raise ValueError("Inout frame is empty.")

    # BGR -> RGB  convert frame.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))

    # uint8 [0, 255] -> float32 [0, 1]
    image = image.astype(np.float32) / 255.0

    # HWC -> CHW
    image = np.transpose(image, (2,0,1))

    # CHW -> NCHW 
    image = np.expand_dims(image, axis=0)

    return np.ascontiguousarray(image)
