from pathlib import Path 

import onnxruntime as ort 
import numpy as np 

class YOLOInference:

    def __init__(self, model_path: Path) -> None:

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        session = ort.InferenceSession(
            str(model_path), 
            providers=["CPUExecutionProvider"],
        )

        self.input_name = session.get_inputs()[0].name 
        self.output_name = session.get_outputs()[0].name 

        self.input_info = session.get_inputs()[0]

        self.input_shape = self.input_info.shape 
        self.input_dtype = self.input_info.type 

        print(f"Loaded Model: {model_path}")
        print(f"Input: {self.input_name}, shape: {self.input_shape}")
        print(f"Output: {self.output_name}")

    def predict(self, input_data):
        pass

if __name__ == "__main__":

    model_path = Path("models/yolo11n.onnx")
    engine = YOLOInference(model_path)

    



        