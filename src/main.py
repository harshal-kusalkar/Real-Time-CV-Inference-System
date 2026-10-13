import time

import cv2
import numpy as np

from src.config import WINDOW_NAME, CAMERA_INDEX
from src.preprocessing import preprocess_frame
from src.inference import InferenceEngine


def main() -> None:
    engine = InferenceEngine()
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    window_size = 100
    counter = 0
    SHOW_DISPLAY = False

    stage_totals = {
        "capture": 0.0,
        "preprocess": 0.0,
        "inference": 0.0,
        "postprocess": 0.0,
        "display": 0.0,
        "pipeline": 0.0,
    }

    def elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000

    try:
        while True:
            # Measure the entire iteration
            pipeline_start = time.perf_counter()

            # 1. Capture
            start = time.perf_counter()
            ret, frame = cap.read()
            capture_ms = elapsed_ms(start)

            if not ret:
                print("Failed to read webcam frame.")
                break

            # 2. Preprocessing
            start = time.perf_counter()
            model_input = preprocess_frame(frame)
            preprocess_ms = elapsed_ms(start)

            # 3. Inference
            start = time.perf_counter()
            prediction = engine.predict(model_input)
            inference_ms = elapsed_ms(start)

            # 4. Postprocessing
            start = time.perf_counter()
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            postprocess_ms = elapsed_ms(start)

            # 5. Display and event handling
            start = time.perf_counter()

            if SHOW_DISPLAY:
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
                    f"Inference: {inference_ms:.2f} ms",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
            else:
              key = -1

            display_ms = elapsed_ms(start)

            # End-to-end time for this measured iteration
            pipeline_ms = elapsed_ms(pipeline_start)

            stage_totals["capture"] += capture_ms
            stage_totals["preprocess"] += preprocess_ms
            stage_totals["inference"] += inference_ms
            stage_totals["postprocess"] += postprocess_ms
            stage_totals["display"] += display_ms
            stage_totals["pipeline"] += pipeline_ms

            counter += 1

            if counter % window_size == 0:
                print(f"\n--- Last {window_size} frames ---")

                for stage in (
                    "capture",
                    "preprocess",
                    "inference",
                    "postprocess",
                    "display",
                    "pipeline",
                ):
                    avg_ms = stage_totals[stage] / window_size
                    print(f"{stage:12s}: {avg_ms:.3f} ms/frame")

                avg_pipeline_ms = (
                    stage_totals["pipeline"] / window_size
                )
                pipeline_fps = 1000.0 / avg_pipeline_ms

                print(f"Pipeline FPS: {pipeline_fps:.2f}")

                # Start a fresh measurement window
                for stage in stage_totals:
                    stage_totals[stage] = 0.0

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam resources released.")


if __name__ == "__main__":
    main()