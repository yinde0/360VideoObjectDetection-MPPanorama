'''
Object Detection on Panorama pictures and videos
Usage:
    $ python detection.py --img <input_file> --output <output_file>
    $ python detection.py --video <input_file> --output <output_file>

    input_file (str):  the input panorama image or video
    output_file (str): the output panorama image or video with bounding boxes
'''

import argparse
import queue

import sys
import cv2
import numpy as np
from yolov8_model_run import detect
from stereo import panorama_to_stereo_multiprojections, stereo_bounding_boxes_to_panorama
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed



def video_detection(input_video_path, stereographic_image_size, FOV, output_file_path, thread_count):
    '''
    Function:
        Take in a set of equirectangular panoramas (360-degree video) and apply object detection.
        Split each panorama frame into 4 images based on stereographic projection.
        Run Yolov8 model finetuned with coco128 on each image to generate bounding boxes.
        Draw bounding boxes back on panoramas.

        Based on "Object Detection in Equirectangular Panorama".

    Inputs:
        System Arguments:
            (1) input 360-degree video file path
            (2) output file path to write 360-degree video with object bounding boxes
    '''

    try:
        video_reader = cv2.VideoCapture(input_video_path)
    except:
        print("Failed to read input video path.")

    # Ensure that video opened successfully
    if not video_reader.isOpened():
        print("Error: Could not open video.")
        exit()

    annotated_panoramas = {}

    fps = video_reader.get(cv2.CAP_PROP_FPS)
    total_num_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Frame Rate: ", fps)
    print("Total number of frames: ", total_num_frames)


    def process_frame(frame_count, pano_array, stereographic_image_size, FOV):
        print("Processing frame: ", frame_count)
        
        # Get frames along with (yaw, pitch) rotation value for the 4 stereographic projections for input panorama
        frames = panorama_to_stereo_multiprojections(pano_array, stereographic_image_size, FOV)

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
        output_panorama_np = stereo_bounding_boxes_to_panorama(frames_detections_with_meta_np, pano_array, stereographic_image_size, FOV)
        annotated_panoramas[frame_count] = output_panorama_np

        # Successful return
        return 0


    # if only one thread, run it on the current thread without using the multithread manager
    if thread_count == 1:
        for frame_count in tqdm(range(int(total_num_frames))):
            ret, pano_array = video_reader.read() # pano_array written in BGR format
            if ret is None:
                print("Finished reading all frames before expected")
                break
            result_code = process_frame(frame_count, pano_array, stereographic_image_size, FOV)
            if result_code != 0:
                break
    
    # Otherwise, run it with the multithread manager
    elif thread_count > 1:
        # Create the multithread manager and specify the number of threads in use
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = queue.Queue()
            for frame_count in tqdm(range(int(total_num_frames))):
                ret, pano_array = video_reader.read() # pano_array written in BGR format
                if ret is None:
                    print("Finished reading all frames before expected")
                    
                # Submit the task to the executor
                future = executor.submit(process_frame, frame_count, pano_array, stereographic_image_size, FOV)
                futures.put(future)
                while futures.qsize() > thread_count * 2:
                    future = futures.get()
                    result = future.result()  # This line will block until the future is done
                    print(f"Task returned: {result}")

            # handle task results (possible errors) after all have been queued
            for future in as_completed(futures):
                result = future.result()  # This line will block until the future is done
                print(f"Task returned: {result}")
    

    # Release video reader object
    video_reader.release()

    # Store the panorama image with bounding boxes
    if output_file_path:
        
        # Defining codec and creating video_writer object
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(output_file_path, fourcc, fps, (annotated_panoramas.shape[1], annotated_panoramas.shape[0]))

        # Write each frame in annotated_panoramas to the video file
        for output_image in annotated_panoramas:
            video_writer.write(output_image)

        # Close the video_writer object when finished
        video_writer.release()
        print("The annotated 360 video file has been written successfully.")


    

def image_detection(input_panorama_path, stereographic_image_size, FOV, output_file_path):
    '''
    Function:
        Take in an equirectangular panorama (360 image) and apply object detection.
        Split panorama into 4 images based on stereographic projection.
        Run Yolov8 model finetuned with coco128 on each image to generate bounding boxes.
        Draw bounding boxes back on panoramas.

        Based on "Object Detection in Equirectangular Panorama".

    Inputs:
        System Arguments:
            (1) input panorama image file path
            (2) output file path to write panorama image with object bounding boxes
    '''

    try:
        pano_array = cv2.imread(input_panorama_path) # Written in BGR format
    except:
        print("Failed to read input panorama path.")

    # Ensure the image was loaded successfully
    if pano_array is None:
        raise IOError("The image could not be opened or is empty.")
    


    # Get frames along with (yaw, pitch) rotation value for the 4 stereographic projections for input panorama
    frames = panorama_to_stereo_multiprojections(pano_array, stereographic_image_size, FOV)

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
    output_panorama_np = stereo_bounding_boxes_to_panorama(frames_detections_with_meta_np, pano_array, stereographic_image_size, FOV)

    # Store the panorama image with bounding boxes
    if output_file_path:
        cv2.imwrite(output_file_path, output_panorama_np)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model-runs\\detect\\train\\weights\\yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--video", help="Path to input 360 video.")
    parser.add_argument("--img", help="Path to input 360 image.")
    parser.add_argument("--stereo_image_size", help="The size in pixels of the stereographic images derived from the panorama", default="640x640")
    parser.add_argument("--FOV", help="", default="180x180")
    parser.add_argument("--output", help="Path to output image.", default=None)
    parser.add_argument("--threads", type=int, help="Number of threads for parallelization (video only)", default=1)
    args = parser.parse_args()

    # Set variable values
    try:
        width, height = map(int, args.stereo_image_size.split('x'))
        stereographic_image_size = (width, height)
    except ValueError:
        raise argparse.ArgumentTypeError("Size must be WxH, where W and H are integers.")
    try:
        theta, phi = map(int, args.stereo_image_size.split('x'))
        FOV = (theta, phi)
    except ValueError:
        raise argparse.ArgumentTypeError("FOV Angles must be ThetaxPhi, where Theta and Phi are integers. See stereo.py description for specifics on angles.")

    output_file_path = args.output
    thread_count = args.threads

    if thread_count < 1:
        raise argparse.ArgumentTypeError("thread_count must be an integer greater than zero.")


    if args.video:
        input_video_path = args.video
        video_detection(input_video_path, stereographic_image_size, FOV, output_file_path, thread_count)
    elif args.img:
        input_panorama_path = args.img
        image_detection(input_panorama_path, stereographic_image_size, FOV, output_file_path)

    

if __name__ == '__main__':
    main()
