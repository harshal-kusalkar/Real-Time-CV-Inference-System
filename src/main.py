import cv2 
import numpy as np 

from src.config import WINDOW_NAME, CAMERA_INDEX
from src.preprocessing import preprocess_frame
from src.inference import InferenceEngine
from src.profiling import InferenceProfiler


def main() -> None: 

    engine = InferenceEngine()
    profiler = InferenceProfiler(window_size=100)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:

            ret, frame = cap.read()

            if not ret: 
                print("Failed to read webcam frame")
                break

            # Pre-processing
            model_input = preprocess_frame(frame)

            # predict 

            start_time = profiler.start()
            predition = engine.predict(model_input)
            inference_latency_ms = profiler.stop(start_time)

            predicted_class = int(np.argmax(predition, axis=1)[0])

            # Display performance metrics
            cv2.putText(
                frame,
                f"Class: {predicted_class}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Inference: {inference_latency_ms:.2f} ms",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Avg FPS: {profiler.fps:.1f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )


            cv2.imshow(WINDOW_NAME, frame)

            # press q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break 

    finally:

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam resource released.")

if __name__ == "__main__":
    main()





