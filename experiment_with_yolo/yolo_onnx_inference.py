import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/yolo11n.onnx"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.40

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def preprocess(frame):
    original_h, original_w = frame.shape[:2]

    # Simple resize for this first experiment
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image)

    return image, original_w, original_h


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            model_input, original_w, original_h = preprocess(frame)

            output = session.run(
                [output_name],
                {input_name: model_input},
            )


            prediction = output[0]

            pred = prediction[0]

            pred = pred.T

            # print("Transpose shape: ", pred.shape) 

            # box coordinate
            boxes = pred[:, :4]

            # remaining 80 values are class scores
            class_scores = pred[:, 4:]

            # Best class and its score 
            class_ids = np.argmax(class_scores, axis=1)
            confidences = np.max(class_scores, axis=1)

            # Find the strongest candidates
            best_idx = np.argmax(confidences)

            # print("Shape: ",pred.shape)
            # print("Dtype: ",pred.dtype)
            # print("First Candidate: ", pred[0, :, 0])


            # ============ Filter candidates whose threshold value > THRESHOLD
            mask = confidences > CONFIDENCE_THRESHOLD

            filtered_boxes = boxes[mask]
            filtered_scores = confidences[mask]
            filtered_class_ids = class_ids[mask]

            # print("Total candidates:", len(confidences))
            # print("Candidates above threshold:", len(filtered_scores))
            # print("Top 5 scores:", np.sort(filtered_scores)[-5:][::-1])

            # for i in np.argsort(filtered_scores)[::-1]:
                #     print(
                #     f"Class: {filtered_class_ids[i]}, "
                #     f"Confidence: {filtered_scores[i]:.4f}, "
                #     f"Box: {filtered_boxes[i]}"
                # )

            # print("Best candidate index:", best_idx)
            # print("Bounding box:", boxes[best_idx])
            # print("Class ID:", class_ids[best_idx])
            # print("Confidence:", confidences[best_idx])


            # Convert xywh center format to top-left format for OpenCV NMS.
            xywh = []

            for xc, yc, w, h in filtered_boxes:
                x = float(xc - w / 2)
                y = float(yc - h / 2)
                xywh.append([x, y, float(w), float(h)])

            indices = cv2.dnn.NMSBoxes(
                bboxes=xywh,
                scores=filtered_scores.tolist(),
                score_threshold=CONFIDENCE_THRESHOLD,
                nms_threshold=0.65,
            )

            # Draw final detections.
            height, width = frame.shape[:2]
            x_scale = original_w / IMAGE_SIZE
            y_scale = original_h / IMAGE_SIZE

            for idx in np.array(indices).flatten():
                xc, yc, w, h = filtered_boxes[idx]

                x1 = int((xc - w / 2) * x_scale)
                y1 = int((yc - h / 2) * y_scale)
                x2 = int((xc + w / 2) * x_scale)
                y2 = int((yc + h / 2) * y_scale)

                # Clip coordinates to original frame.
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                confidence = float(filtered_scores[idx])
                class_id = int(filtered_class_ids[idx])

                # COCO class 0 = person.
                label = f"{filtered_class_ids}: {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("YOLO", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()