import argparse
from pathlib import Path

import torch
from abnormality_detection import abnormality_detection
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
        instance that has parsed arguments.
    """
    parser = argparse.ArgumentParser("Perform 2d picking model inference and over an image and judge whether"
                                     "the detected objects are within allowed area")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to  the trained 2d picking model weights")
    parser.add_argument("image_path", type=str, help="Path to the image to be tested")
    parser.add_argument("allowed_regions", type=str, help="Path to an image describing the allowed regions."
                        " It must contain only black and white pixels, black meaning not allowed regions."
                        " It must have the same dimensions as the input image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # handle device
    device = torch.device("cuda")
    grasping_model = abnormality_detection.load_model(Path(args.model_path), device)
    print(grasping_model)
    image = Image.open(args.image_path)

    # TODO load image (As a PIL.Image) from path, load allowed regions from path (as np.ndarray) and pass them as
    # arguments for judge_image()
    # image = Image.open("C:/Users/GBM/Downloads/XC/images/images/sample_000002.jpg")
    # allowed_regions = np.array(Image.open("path_to_allowed_regions_map"))
    # detected_objects = abnormality_detection.judge_image(grasping_model, image, allowed_regions, device)
    # instead call grasping inference and print what's been returned
    # detected_objects = abnormality_detection.grasping_inference(grasping_model, image, device)
    # for object in detected_objects:
    #     # TODO if object is outside allowed region, print its information
    #     pass
