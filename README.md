# Multi-Projection YOLO
Using object detection for 360 degree videos outlined in 'Object Detection in Equirectangular Panorama' (2018 ICPR) (https://arxiv.org/abs/1805.08009).

# Set Up
```
pip install -r requirements.txt
```

# Usage
Windows:
```
python detection.py --video <input_file> --output <output_file>
```

input_file (str):  the input 360-degree video you want to run object detection on.

output_file (str): the output 360-degree video with bounding boxes (some examples can be found in datasets folder).

```
python detection.py --img <input_file> --output <output_file>
```
input_file (str):  the input 360-degree image you want to run object detection on (some examples can be found in datasets folder).

output_file (str): the output 360-degree image with bounding boxes.

Stable version with on .mp4 and .mp3 files.


# License
This repository is released under the MIT License (refer to the LICENSE file for details).
