import numpy as np 

from src.preprocessing import preprocess_frame

frame = np.random.randint(
    0, 255, 
    size=(480, 640, 3), 
    dtype='uint8', 
)

model_input = preprocess_frame(frame)

print("Input shape:", frame.shape)
print("Input dtype:", frame.dtype)

print("Model input shape:", model_input.shape)
print("Model input dtype:", model_input.dtype)
print("Minimum:", model_input.min())
print("Maximum:", model_input.max())
print("C-contiguous:", model_input.flags["C_CONTIGUOUS"])

assert model_input.shape == (1, 3, 224, 224)
assert model_input.dtype == np.float32
assert 0.0 <= model_input.min() <= model_input.max() <= 1.0
assert model_input.flags["C_CONTIGUOUS"]

print("Preprocessing test passed!")