import argparse
from pathlib import Path
# from typing import List
# from datetime import datetime

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda')

    # loading model, image, and allowed_regions
    model = abnormality_detection.load_model(Path(args.model_path), device)
    image_path = Path(args.image_path)
    map = Image.open('C:/Users/GBM/Downloads/XC/map.png').convert('L')
    allowed_regions = np.asarray(map)
    allowed_regions = allowed_regions.T

    # pass image and allowed_regions as arguments for judge_image()
    detected_items = abnormality_detection.judge_image(model, image_path, allowed_regions, device)

    for object in detected_items:
        #  if object is outside allowed region, print its information
        if object["judge"] == False:
            print(object)
