1. What we're building

A webcam inference application with a modular pipeline:

1. Frame Capture

Read frames from the webcam

2. Preprocessing

BGR → RGB → resize → normalize → NCHW

3. ONNX Inference Engine

Load once, run repeatedly

4. Postprocessing & Profiling

Extract predictions, measure latency and FPS

5. Display

Show the frame and performance metrics

