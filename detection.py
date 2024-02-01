'''
Object Detection on Panorama pictures
Usage:
    $ python detection.py <panorama_picture> <output_picture>

    panorama_picture(str):  the panorama pic file
    output_picture(str):    the result picture
'''
import sys
import cv2
import numpy as np
from yolov8_model_run import detect
from stereo import panorama_to_stereo_multiprojections, stereo_bounding_boxes_to_panorama
from tqdm import tqdm


def main():
    '''
    Function:
        Take in a set of equirectangular panoramas (360 images) and apply object detection.
        Split panorama into 4 images based on stereographic projection.
        Run Yolov8 model finetuned with coco128 on each image to generate bounding boxes.
        Draw bounding boxes back on panoramas.

        Based on "Object Detection in Equirectangular Panorama".

    Inputs:
        System Arguments:
            (1) input panorama image
            (2) output file path to write panorama image with object detection bounding boxes
    '''
    # my_net = Yolo()

    # Set variable values
    input_panorama_path = sys.argv[1]
    stereographic_image_size = (640, 640)
    FOV = (180, 180)
    output_file_path = sys.argv[2]

    # Get frames along with (yaw, pitch) rotation value for the 4 stereographic projections for input panorama
    frames = panorama_to_stereo_multiprojections(input_panorama_path, stereographic_image_size, FOV)

    # Get bounding boxes for each frame
    frames_detections_with_meta = []
    for frame in frames:
        # detections contains all YOLOv8 'detections' within the current frame
        detections = detect(frame['image'])

        # Add meta data about the yaw and pitch rotations of the frame to derive the image
        detections_with_meta = (detections, frame['yaw'], frame['pitch'])
        # Append the frame detections with meta data to the list of frames
        frames_detections_with_meta.append(detections_with_meta)

    # Format as an np array
    frames_detections_with_meta_np = np.array(frames_detections_with_meta, dtype=np.dtype([('image_detections', np.ndarray), ('yaw', int), ('pitch', int)]))
    
    # Add the bounding boxes from the stereographic projection frames to the original panorama and return the annotated np.ndarray
    output_panorama_np = stereo_bounding_boxes_to_panorama(frames_detections_with_meta_np, input_panorama_path, stereographic_image_size, FOV)

    # Store the panorama image with bounding boxes
    cv2.imwrite(sys.argv[2], output_panorama_np)

if __name__ == '__main__':
    main()
