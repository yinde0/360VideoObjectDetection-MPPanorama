#!/usr/bin/env python3
"""
Optimized Object Detection on Panorama Pictures and Videos

Usage:
    $ python detection.py --img <input_file> --output_frames <output_file>
    $ python detection.py --video <input_file> --output_frames <output_file>

    input_file (str):  the input panorama image or video
    output_file (str): the output panorama image or video with bounding boxes
"""

import argparse
import cv2
import imageio
import json
import numpy as np
from yolov8_model_run import detect
from stereo import panorama_to_stereo_multiprojections, stereo_bounding_boxes_to_panorama
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def video_detection(input_video_path, stereographic_image_size, FOV, output_image_file_path, output_json_file_path, thread_count, seconds_process=-1):
    """
    Process a 360° video: for each frame, create stereographic projections, run detection,
    map the detections back onto the panorama, and then write the result to an output video.
    
    Memory Optimizations:
      - In single-threaded mode, each processed frame is written immediately to disk
        (avoiding the accumulation of all frames in memory).
      - In multi-threaded mode, results are temporarily stored in a dictionary and then
        written in order; if your video is extremely long, consider processing segments.
    """
    video_reader = cv2.VideoCapture(input_video_path)
    if not video_reader.isOpened():
        print("Error: Could not open video.")
        exit(1)

    fps = video_reader.get(cv2.CAP_PROP_FPS)
    total_num_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    if seconds_process > 0:
        frames_specified = seconds_process * int(fps)
        total_num_frames = min(total_num_frames, frames_specified)
    print("Frame Rate:", fps)
    print("Total number of frames to process:", total_num_frames)

    # We'll collect detections in a (hopefully small) dictionary.
    panorama_detections = {}

    def process_frame(frame_count, pano_array, stereographic_image_size, FOV):
        # Generate stereographic projections from the panorama.
        frames = panorama_to_stereo_multiprojections(pano_array, stereographic_image_size, FOV)
        frames_detections_with_meta = []
        for frame in frames:
            # Run YOLOv8 detection on the stereographic image.
            detections = detect(frame['image'], confidence_threshold=0.45)
            # Filter out detections that are too close to the frame’s edges.
            cleaned_detections = []
            for detection in detections:
                box = detection['box']
                if not (box[0] < 5 or box[1] < 5 or 
                        box[0] + box[2] > frame['image'].shape[0] - 5 or 
                        box[1] + box[3] > frame['image'].shape[1] - 5):
                    cleaned_detections.append(detection)
            # Append tuple: (detections, yaw, pitch)
            frames_detections_with_meta.append((cleaned_detections, frame['yaw'], frame['pitch']))

        # Convert the list into a structured NumPy array required by stereo_bounding_boxes_to_panorama.
        dtype = np.dtype([('image_detections', object), ('yaw', int), ('pitch', int)])
        frames_detections_with_meta_np = np.array(frames_detections_with_meta, dtype=dtype)
        output_panorama_np, pano_detections = stereo_bounding_boxes_to_panorama(frames_detections_with_meta_np, pano_array, stereographic_image_size, FOV)
        return (output_panorama_np, pano_detections, frame_count, 0)

    # === Process Video Frames ===
    if thread_count == 1:
        print("Processing frames (single-threaded)...")
        # Open the video writer; writing each processed frame immediately avoids buildup in memory.
        video_writer = imageio.get_writer(output_image_file_path, fps=int(fps))
        for frame_count in tqdm(range(total_num_frames)):
            ret, pano_array = video_reader.read()
            if not ret:
                print("Finished reading all frames before expected.")
                break
            output_panorama_np, pano_detections, fr_index, code = process_frame(frame_count, pano_array, stereographic_image_size, FOV)
            if code != 0:
                print(f"Frame {frame_count} processing failed with code: {code}")
            else:
                # Write frame immediately.
                rgb_image = cv2.cvtColor(output_panorama_np, cv2.COLOR_BGR2RGB)
                video_writer.append_data(rgb_image)
                panorama_detections[fr_index] = pano_detections
        video_writer.close()
    elif thread_count > 1:
        print("Processing frames (multi-threaded)...")
        annotated_panoramas = {}
        futures = []
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            for frame_count in range(total_num_frames):
                ret, pano_array = video_reader.read()
                if not ret:
                    print("Finished reading all frames before expected.")
                    break
                future = executor.submit(process_frame, frame_count, pano_array, stereographic_image_size, FOV)
                futures.append(future)
            # Collect results as they complete.
            for future in tqdm(as_completed(futures), total=len(futures)):
                output_panorama_np, pano_detections, fr_index, code = future.result()
                if code != 0:
                    print(f"Frame {fr_index} processing failed with code: {code}")
                else:
                    annotated_panoramas[fr_index] = output_panorama_np
                    panorama_detections[fr_index] = pano_detections
        # Write frames in order (sorted by frame index).
        video_writer = imageio.get_writer(output_image_file_path, fps=int(fps))
        for i in range(total_num_frames):
            if i in annotated_panoramas:
                rgb_image = cv2.cvtColor(annotated_panoramas[i], cv2.COLOR_BGR2RGB)
                video_writer.append_data(rgb_image)
        video_writer.close()

    video_reader.release()

    # Convert detections to JSON-friendly format.
    json_pano_detections = {}
    for fr_index, detections in panorama_detections.items():
        json_list = []
        if isinstance(detections, np.ndarray):
            for det_group in detections:
                if isinstance(det_group, (list, tuple)):
                    json_list.extend(det_group)
                else:
                    json_list.append(det_group)
        else:
            json_list = detections
        json_pano_detections[fr_index] = json_list

    if output_json_file_path:
        with open(output_json_file_path, "w") as outfile:
            json.dump(json_pano_detections, outfile, indent=4)
        print(f"Detections JSON saved to {output_json_file_path}")

    print("The annotated 360° video file has been written successfully.")

def image_detection(input_panorama_path, stereographic_image_size, FOV, output_image_file_path, output_json_file_path):
    """
    Process a 360° image: create stereographic projections, run object detection,
    map detections back onto the panorama, and write the annotated image.
    """
    pano_array = cv2.imread(input_panorama_path)
    if pano_array is None:
        raise IOError("The image could not be opened or is empty.")

    frames = panorama_to_stereo_multiprojections(pano_array, stereographic_image_size, FOV)
    frames_detections_with_meta = []
    for frame in frames:
        detections = detect(frame['image'], confidence_threshold=0.45)
        cleaned_detections = []
        for detection in detections:
            box = detection['box']
            if not (box[0] < 5 or box[1] < 5 or 
                    box[0] + box[2] > frame['image'].shape[0] - 5 or 
                    box[1] + box[3] > frame['image'].shape[1] - 5):
                cleaned_detections.append(detection)
        frames_detections_with_meta.append((cleaned_detections, frame['yaw'], frame['pitch']))

    dtype = np.dtype([('image_detections', object), ('yaw', int), ('pitch', int)])
    frames_detections_with_meta_np = np.array(frames_detections_with_meta, dtype=dtype)
    output_panorama_np, panorama_detections = stereo_bounding_boxes_to_panorama(frames_detections_with_meta_np, pano_array, stereographic_image_size, FOV)
    print("Detections:", panorama_detections)

    json_pano_detections = []
    if isinstance(panorama_detections, np.ndarray):
        for det in panorama_detections:
            json_pano_detections.append(det[0])
    else:
        json_pano_detections = panorama_detections

    if output_json_file_path:
        with open(output_json_file_path, "w") as outfile:
            json.dump(json_pano_detections, outfile, indent=4)
    if output_image_file_path:
        cv2.imwrite(output_image_file_path, output_panorama_np)
    print("The annotated 360° image has been written successfully.")

def main():
    parser = argparse.ArgumentParser(description="Object Detection on Panorama pictures and videos with memory optimization.")
    parser.add_argument("--model", default="model-runs\\detect\\train\\weights\\yolov8n.onnx", help="Path to your ONNX model.")
    parser.add_argument("--video", help="Path to input 360° video.")
    parser.add_argument("--img", help="Path to input 360° image.")
    parser.add_argument("--stereo_image_size", help="Size of stereographic images (WxH), e.g. 640x640", default="640x640")
    parser.add_argument("--FOV", help="Field of view angles (Theta x Phi), e.g. 180x180", default="180x180")
    parser.add_argument("--output_detections", help="Path to output JSON file for detections.", default=None)
    parser.add_argument("--output_frames", help="Path to output image or video file.", default=None)
    parser.add_argument("--threads", type=int, help="Number of threads for parallelization (video only)", default=1)
    parser.add_argument("--seconds_process", type=int, help="Number of seconds of the video to process", default=-1)
    args = parser.parse_args()

    # Parse stereographic image size.
    try:
        width, height = map(int, args.stereo_image_size.split('x'))
        stereographic_image_size = (width, height)
    except ValueError:
        raise argparse.ArgumentTypeError("Stereo image size must be in WxH format, e.g. 640x640.")

    # Parse FOV angles from the --FOV argument.
    try:
        theta, phi = map(int, args.FOV.split('x'))
        FOV = (theta, phi)
    except ValueError:
        raise argparse.ArgumentTypeError("FOV must be in Theta x Phi format, e.g. 180x180.")

    output_image_file_path = args.output_frames
    output_json_file_path = args.output_detections
    thread_count = args.threads
    seconds_process = args.seconds_process

    if thread_count < 1:
        raise argparse.ArgumentTypeError("Threads must be an integer greater than zero.")

    if args.video:
        video_detection(args.video, stereographic_image_size, FOV, output_image_file_path, output_json_file_path, thread_count, seconds_process)
    elif args.img:
        image_detection(args.img, stereographic_image_size, FOV, output_image_file_path, output_json_file_path)
    else:
        print("Error: No input file provided. Please specify --video or --img.")
        exit(1)

if __name__ == '__main__':
    main()
