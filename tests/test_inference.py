import numpy as np

from src.inference import InferenceEngine
from src.preprocessing import preprocess_frame

engine = InferenceEngine()

# Simulated webcam frame
frame = np.random.randint(
    0, 256,
    size=(480, 640, 3),
    dtype=np.uint8,
)

model_input = preprocess_frame(frame)
prediction = engine.predict(model_input)

print("Prediction shape:", prediction.shape)
print("Prediction dtype:", prediction.dtype)
print("Predicted class:", int(np.argmax(prediction, axis=1)[0]))

assert prediction.shape == (1, 10)
assert np.isfinite(prediction).all()

print("Inference test passed!")