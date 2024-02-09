# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import cv2.dnn
import numpy as np

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml
from draw_boxes import draw_bounding_box
from PIL import Image

CLASSES = yaml_load(check_yaml("coco128.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def detect(input_array, onnx_model="model-runs\\detect\\train\\weights\\yolov8n.onnx"):
    """
    Function:
    Load ONNX model, perform inference, draw bounding boxes, and display the output image.

    Input:
        onnx_model (str): Path to the ONNX model.
        input_array (np.ndarray): Path to the input image.

    Output:
        detections: Array of detections (list) with info such as class_id, class_name, confidence, etc.
    """
    # Load the ONNX model
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # Read the input image
    original_image = input_array
    [height, width, _] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    # Display the image with bounding boxes
    '''
    For testing purposes to show the bounding boxes on the stereographic projected image
    '''
    # cv2.imshow("image with detections", original_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model-runs\\detect\\train\\weights\\yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", help="Path to input image.")
    parser.add_argument("--output", help="Path to output image.", default=None)
    args = parser.parse_args()

    original_image: np.ndarray = cv2.imread(args.img)
    print(original_image.shape)
    detect(original_image, args.model)
    if args.output:
        image_height = original_image.shape[0]
        image_width = original_image.shape[1]        

        output_image = Image.fromarray(original_image.reshape((image_height, image_width, 3)).astype('uint8'), 'RGB')
        
        cv2.imwrite(args.output)