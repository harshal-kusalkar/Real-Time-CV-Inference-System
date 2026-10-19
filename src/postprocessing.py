import numpy as np
import cv2

def postprocess(prediction: np.ndarray, orig_img_shape, input_size=(640, 640)):

    # Remove batch dim and Transpose
    pred = prediction[0].T 

    # print(pred.shape)

    # Extract Bounding boxes and class scores
    boxes = pred[:, :4]
    scores = pred[:, 4:]

    # print(boxes[:2])
    # print(scores[:2])

    # Get winnign class ids and max confidence score
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # print(class_ids[:1])
    # print(confidence[:1])

    # filter classes below confidence threshold
    mask = confidences > 0.45 # return a list of True and False
    # print(mask)
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        pass
        # return [], [], []
    
    # Convert centre format (cx, cy, w, h) to (x1, y1, w, h)
    x_centre, y_centre, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_centre - (width/2)
    y1 = y_centre - (height/2)

    boxes_nms = np.vstack((x1, y1, width, height)).T.tolist()
    confidences_nms = confidences.tolist()

    # Apply Non-Maximum-Suppression (NMS) to clear overlapping boxe
    indices = cv2.dnn.NMSBoxes(boxes_nms, confidences_nms, 0.45, 0.65) # last tow index conf_threshold  iou_threshol

    # print(indices)
    # print(type(indices))

    final_boxes = []
    final_conf = []
    final_class_ids = []

    if len(indices) > 0:
        indices = np.array(indices).flatten()

        # print(indices)
        # print(indices.dtype)

        # Scale bounding box to back to original image aspect ratio

        orig_width, orig_height = orig_img_shape[:2]
        scale_x = orig_width / input_size[0]
        scale_y = orig_height / input_size[1]

        for i in indices:
            bx, by, bw, bh = boxes_nms[i]

            # Map back to [x1, y1, x2, y2]
            x1_orig = int(bx * scale_x)
            y1_orig = int(by * scale_y)
            x2_orig = int((bx + bw)*scale_x)
            y2_orig = int((by+bh)*scale_y)

            final_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
            final_conf.append(confidences_nms[i])
            final_class_ids.append(int(class_ids[i]))

    return final_boxes, final_conf, final_class_ids