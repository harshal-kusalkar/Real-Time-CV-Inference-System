import time
import numpy as np 
import cv2 

import onnxruntime as ort

import json
from src.preprocessing import preprocess
from src.postprocessing import postprocess


MODEL_PATH = "models/yolo11n.onnx"
IMAGE_SIZE = 640

# load labels.json

try:

    with open("labels.json", "r") as f:
        labels = json.load(f)

    print(f"Labels loaded successfully. \n size: {len(labels)}")
except:
    raise FileNotFoundError()
     
    
def main():

    session = ort.InferenceSession(
        MODEL_PATH, 
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name 
    output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
 
    try:
        while True: 

            ret, frame = cap.read()

            if not ret:
                print("Could not read frame")

            # print ret and shape of frame cap.
            # print(ret)
            # print(frame.shape)

            # preprocessing 
            model_input, frame_w, frame_h = preprocess(frame)  # return image, frame_h, frame_w
            # print("After preprocess")
            # print(model_input)
            # print(model_input.shape)
            # print(model_input.dtype)

            # Inference 

            output = session.run( 
                [output_name], 
                {input_name: model_input},
            )

            pred = output[0]
            # print(pred.shape)

            # Post-Process
            boxes, confidences, class_ids = postprocess(pred, (frame_w, frame_h))

            for idx in range(len(boxes)):
              x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
              confidence = confidences[idx]
              label = labels[str(class_ids[idx])]

              cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
              cv2.putText(
                  frame, 
                  f"Confidence: {confidence}", 
                  (x1, y1-10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 
                  0.6, 
                  (0, 255, 0),
                  2
              )

              cv2.putText(
                  frame, 
                  f"Label: {label}", 
                  (x1, y1-30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 
                  0.6, 
                  (0,255,0),
                  2
              )

              cv2.imshow("YOLO WEBCAM", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

