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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # TODO install pytorch with cuda supported
    device = torch.device('cuda')
    grasping_model = abnormality_detection.load_model(Path(args.model_path), device)
    # print(grasping_model)
    image_png = Image.open(args.image_path)
    print(type(image_png))

    background = Image.new('RGBA', image_png.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_png)
    # if you check the matrix dimension, channel, it would be still 4.
    # h, w, channel = np.asarray(alpha_composite)
    alpha_composite_3 = alpha_composite.convert('RGB')
    print(type(alpha_composite_3))

    # call grasping inference and print what's been returned
    detected_objects, visualization_results = abnormality_detection.grasping_inference(
        grasping_model, alpha_composite_3, device)
    print(detected_objects)
    visualization_results.show()
    # OR TODO passing image and allowed_regions as arguments for judge_image()

    # for object in detected_objects:
    #     # TODO if object is outside allowed region, print its information
    #     pass
