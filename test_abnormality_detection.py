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
    parser = argparse.ArgumentParser('Perform 2d picking model inference and over an image and judge whether'
                                     'the detected objects are within allowed area')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to  the trained 2d picking model weights')
    parser.add_argument('image_path', type=str, help='Path to the image to be tested')
    parser.add_argument('--allowed-regions', type=str, help='Path to an image describing the allowed regions.'
                        ' It must contain only black and white pixels, black meaning not allowed regions.'
                        ' It must have the same dimensions as the input image')
    parser.add_argument('--min-dist', '-d', type=int)
    parser.add_argument('--max-dist', '-D', type=int)
    parser.add_argument('--min-angle', '-a', type=float)
    parser.add_argument('--max-angle', '-A', type=float)

    parser.add_argument('--gpu', action='store_true', help='Select device: CPU or CUDA')
    parser.add_argument('--video', type=str, required=False,
                        help='Path to the reference video for creating the map of allowed regions'
                        'Video must have the same dimensions ad the map of allowed regions')
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0,
        help='It must be between 0 and 1')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # select type of device
    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    video_path = args.video
    conf_threshold = args.conf_threshold

    # load input image (convert from RGBA to RGB)
    image = Image.open(args.image_path).convert('RGB')

    # load grasping model
    model = abnormality_detection.load_model(Path(args.model_path), device)

    allowed_regions = None
    if args.allowed_regions:
        # load allowed_regions map
        allowed_regions_img = Image.open(args.allowed_regions)
        allowed_regions = np.asarray(allowed_regions_img.convert('L'))
        allowed_regions = allowed_regions.T

    allowed_angle_range = None
    if args.min_angle is not None and args.max_angle is not None:
        allowed_angle_range = (args.min_angle, args.max_angle)
    elif args.min_angle is not None or args.max_angle is not None:
        raise ValueError("Both or Neither min_angle and max_angle must be specified")

    # # OR create allowed regions map
    # created_allowed_regions = abnormality_detection.create_allowed_regions(video, model, device)
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
            list_of_frames, model, device)
        print("Allowed_regions_map was created")
        cv2.imwrite("./created_map.png", created_allowed_regions_map)
        # created_allowed_regions_map.save(f"{no_extension}_created_allowed_regions_{ts}.png")
        print("Allowed regions map created and saved")

    # pass image and allowed_regions as arguments for judge_image()
    # judge image according to allowed_regions
    detected_items, visualization = abnormality_detection.judge_image(
        model, image, device, grasping_conf_threshold=args.conf_threshold,
        allowed_regions=allowed_regions, min_allowed_separation_dist=args.min_dist,
        max_allowed_separation_dist=args.max_dist, angle_range=allowed_angle_range)

    # save visualization_results under format inputImagePath_timestamp.jpg
    dt = datetime.now()
    ts = str(datetime.timestamp(dt))
    no_extension = os.path.splitext(args.image_path)[0]
    # visualization_results.show()
    visualization.save(f"{no_extension}_{ts}.jpg")
    print(f"Grasping visualization saved at: {no_extension}_{ts}.jpg")

    for object in detected_items:
        #  if object is outside allowed region, print its information
        if (
            not object.get('inside_allowed_region', True)
            or object.get('too_far')
            or object.get('too_close')
            or not object.get('judge_angle', True)
        ):
            print(object)

    # plot objects' positions in blue/green on the allowed_region map
    output = abnormality_detection.plot_results(detected_items, image)
    output.save(f"{no_extension}_allowed_regions_{ts}.png")
    print(
        f"Allowed regions judgement visualization saved at: {no_extension}_allowed_regions_{ts}.png")
