import argparse
from pathlib import Path

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # select type of device
    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    # load grasping model
    grasping_model = abnormality_detection.load_model(Path(args.model_path), device)

    # load input image from argument (converted from RGBA to RGB)
    image = Image.open(args.image_path).convert('RGB')

    # call grasping inference and print what's been returned
    detected_objects, visualization_results = abnormality_detection.grasping_inference(
        grasping_model, image, device)
    print(detected_objects)
    visualization_results.show()
    # OR TODO passing image and allowed_regions as arguments for judge_image()

    # for object in detected_objects:
    #     # TODO if object is outside allowed region, print its information
    #     pass
