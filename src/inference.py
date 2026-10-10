from pathlib import Path 

import numpy as np 
import onnxruntime as ort 

from src.config import (
    MODEL_PATH, 
    PROVIDER, 
    INPUT_HEIGHT, 
    INPUT_WIDTH
)


class InferenceEngine:

    def __init__(self, model_path: Path = MODEL_PATH) -> None:

        if not model_path.exists():
            raise FileNotFoundError(model_path)
        
        self.session = ort.InferenceSession(
            str(model_path), 
            providers=[PROVIDER],
        )

        self.input_name = self.session.get_inputs()[0].name 
        self.output_name = self.session.get_outputs()[0].name 

        self.input_info = self.session.get_inputs()[0]

        self.input_shape = self.input_info.shape 
        self.input_dtype = self.input_info.type

        print(f"Loaded model: {model_path}")
        print(f"Input: {self.input_name}, shape={self.input_shape}")
        print(f"Output: {self.output_name}")

    def predict(self, model_input: np.ndarray) -> np.ndarray:

        if model_input.dtype != np.float32:
            raise ValueError("Model input must have dtype float32.")

        if model_input.ndim != 4:
            raise ValueError("Model input must have NCHW dimensions.")

        if model_input.shape[1:] != (3, INPUT_HEIGHT, INPUT_WIDTH):
            raise ValueError(
                f"Unexpected input shape: {model_input.shape}"
            ) 

        output = self.session.run(
            [self.output_name], 
            {self.input_name: model_input}, 
        )

        return output[0]


        