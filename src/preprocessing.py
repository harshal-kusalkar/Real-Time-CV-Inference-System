import cv2
import numpy as np 

from src.config import (
    INPUT_CHANNELS, 
    INPUT_HEIGHT, 
    INPUT_WIDTH
)

def preprocess(frame):
    original_h, original_w = frame.shape[:2]

    # Simple resize for this first experiment
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image)

    return image, original_w, original_h
