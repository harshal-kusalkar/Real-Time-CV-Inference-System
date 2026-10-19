from pathlib import Path
from ultralytics import YOLO 

model = YOLO("models/yolo11n.pt")

print("Number of classes: ", len(model.names))
print(model.names)
print("Class ID 0: ", model.names[0])

# model.export( 
#     format="onnx", 
#     imgsz=640, 
#     batch=1, 
#     dynamic=False, 
# )

print("YOLO model export completed")

