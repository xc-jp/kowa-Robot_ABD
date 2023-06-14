#  ***** DESCRIPTION *****
# This script generates the map of alowed regions in .png format.
# The map is created based on a given video that is used as reference
# Also, the allowed regions are mapped depending on a variety of inputs specified by the user
# For the moment, we only take a radius as an input argument

# TO DO: add methods for creating allowed ranges of angles and distances


# UNDER CONSTRUCTION


import argparse
from datetime import datetime
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2

from abnormality_detection import abnormality_detection


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        instance that has parsed arguments.
    """
    parser = argparse.ArgumentParser(
        'Create a map defining allowed areas for objects from a video that is used as reference')
    parser.add_argument(
        'model_path', type=str, default='C:/Users/GBM/Downloads/XC/kowa_infer/grasping',
        help='Path to  the trained 2d picking model weights')
    parser.add_argument('saving_path', type=str, default='C:/Users/GBM/Downloads/XC/allowed_regions/',
                        help='Path to store the created map of allowed regions')
    parser.add_argument('--video', type=str, default='C:/Users/GBM/Downloads/testvideo.mp4',
                        help='Path to the reference video for creating the map of allowed regions'
                        'Video must have the same dimensions as the map of allowed regions')
    parser.add_argument(
        '--radius',
        type=int,
        default=0,
        help='Radius of the tracing point on the map')
    parser.add_argument('--gpu', action='store_true', help='Select device: CPU or CUDA')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # select type of device
    radius = args.radius
    video_path = args.video
    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    saving_path = args.saving_path

    # load grasping model
    model = abnormality_detection.load_model(Path(args.model_path), device)

    allowed_regions = None

    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file!")
            exit(0)
        list_of_frames = []
        while cap.isOpened():
            status, frame = cap.read()
            if not status:
                break
            # opencv stores color channels in BGR, so we need to reorder them to RGB
            img = frame[:, :, (2, 1, 0)]
            list_of_frames.append(img)

        created_allowed_regions_map = abnormality_detection.create_allowed_regions(
            list_of_frames, model, device, radius)
        # save image file
        dt = datetime.now()
        ts = str(datetime.timestamp(dt))
        file_name = f"map_{ts}.jpg"
        cv2.imwrite(f"{saving_path}/{file_name}", created_allowed_regions_map)
        print("Allowed regions map was successfully created and saved as:", f"{file_name}")
