import argparse
from datetime import datetime
import os
from pathlib import Path
# from typing import List

import torch
import numpy as np
from PIL import Image

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
    parser.add_argument('allowed_regions', type=str, help='Path to an image describing the allowed regions.'
                        ' It must contain only black and white pixels, black meaning not allowed regions.'
                        ' It must have the same dimensions as the input image')
    parser.add_argument('--gpu', action='store_true', help='Select device: CPU or CUDA')
    parser.add_argument(
        '--conf_threshold',
        type=float,
        default=0,
        help='It must be between 0 and 1')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # select type of device
    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    # load grasping model
    model = abnormality_detection.load_model(Path(args.model_path), device)
    # load input image (convert from RGBA to RGB)
    image = Image.open(args.image_path).convert('RGB')
    # load allowed_regions map
    map_ = Image.open(args.allowed_regions).convert('L')
    map_rgb = Image.open(args.allowed_regions)
    allowed_regions = np.asarray(map_)
    allowed_regions = allowed_regions.T
    # load confidence threshold
    conf_threshold = args.conf_threshold

    # pass image and allowed_regions as arguments for judge_image()
    # judge image according to allowed_regions
    detected_items, visualization = abnormality_detection.judge_image(
        model, image, allowed_regions, device)

    # save visualization_results under format inputImagePath_timestamp.jpg
    dt = datetime.now()
    ts = str(datetime.timestamp(dt))
    no_extension = os.path.splitext(args.image_path)[0]
    # visualization_results.show()
    visualization.save(f"{no_extension}_{ts}.jpg")

    # plot objects' positions in blue/green on the allowed_region map
    output = abnormality_detection.plot_object(detected_items, map_rgb)
    output.save('C:/Users/GBM/Downloads/XC/output.png')
