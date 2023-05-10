"""Module Description"""
from typing import Any, Dict, List, Optional
import json
from pathlib import Path


import numpy as np
import torch
from PIL import Image, ImageDraw

from grasping import infer as grasping_infer
from src.networks.build_network import build_model as build_grasping_model


def load_model(model_path: Path, device: torch.device) -> dict[str, Any]:
    # TODO load pytorch model weights
    with open(model_path.joinpath("obj", "build_parameter.json"), encoding="utf-8") as f:
        build_parameters = json.load(f)
    return _load_grasping(model_path, build_parameters, device)


def _load_grasping(
        path: Path, build_parameters: dict[str, Any], device: torch.device) -> dict[str, Any]:
    input_width = build_parameters["input_width"]
    input_height = build_parameters["input_height"]
    nb_classes = build_parameters["nb_classes"]
    subdivs = build_parameters["subdivs"]
    network_name = build_parameters["network"]
    dim_mins = tuple(build_parameters["dim_mins"])
    dim_maxs = tuple(build_parameters["dim_maxs"])

    network_path = path.joinpath("obj", "model.pth")
    network = build_grasping_model(network_name, model_path=network_path, eval_mode=True,
                                   image_sizes=(input_height, input_width), nb_classes=nb_classes, subdivs=subdivs)
    network = network.to(device)

    return {
        "method": "grasping",
        "network": network,
        "input_width": input_width,
        "input_height": input_height,
        "dim_mins": dim_mins,
        "dim_maxs": dim_maxs,
    }


def grasping_inference(model: dict[str, Any], image: Image.Image, device: torch.device, conf_threshold: float = 0
                       ) -> tuple[list[dict[str, Any]], Image.Image]:
    # implement Grasping Inference Call
    # It returns a list with the detected objects centers, satisfying input threshold
    # return grasping_infer(model, image, device)
    prediction_points, prediction_image = grasping_infer.infer(
        model["network"], image, model["input_width"], model["input_height"], device, visualization=True,
        dim_mins=model["dim_mins"], dim_maxs=model["dim_maxs"])
    filtered_prediction_points = [
        point for point in prediction_points if point["confidence"] >= conf_threshold]
    return filtered_prediction_points, prediction_image


def is_allowed(x: float, y: float, allowed_regions: np.ndarray) -> bool:
    # evaluates whether or not the object is outside allowed area
    # returns true  if the coordinates is inside allowed region, false otherwise
    # allowed_regions is a B&W image (ideally 0-1 values), 1 for allowed coordinates
    return allowed_regions[round(x - 1)][round(y - 1)] != 0


def judge_positions(positions: List[Dict[str, Any]],
                    allowed_regions: np.ndarray) -> None:
    # Modifies in-place the items of a list containing the detected positions,
    # adding a new key: "inside_allowed_region": True/False
    for position in positions:
        position["inside_allowed_region"] = is_allowed(
            position['x'],
            position['y'],
            allowed_regions)


def judge_image(model: Dict[str, Any], image: Image.Image, allowed_regions: np.ndarray, device: torch.device, conf_threshold: float = 0
                ) -> List[Dict[str, Any]]:
    # call grasping inference
    detected_objects, visualization = grasping_inference(
        model, image, device, conf_threshold=conf_threshold)
    judge_positions(detected_objects, allowed_regions)
    return detected_objects, visualization


def plot_object(judged_items: List[Dict[str, Any]], allowed_regions_rgb: Image.Image):
    draw = ImageDraw.Draw(allowed_regions_rgb)
    for item in judged_items:
        x = round(item['x']) - 1
        y = round(item['y']) - 1
        if item['inside_allowed_region']:
            # mark position in blue
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='blue', outline=None)
        else:
            # mark position in red
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='red', outline=None)
    return allowed_regions_rgb


def create_allowed_regions(video: list[np.ndarray], radius: int = 0, dimensions: Optional[list[int]] = None,
                           shape_mask: Optional[np.ndarray] = None) -> np.ndarray:  # type: ignore
    # TODO
    # for now, i defined the mask shape as equal to the allowed_region map
    pass


def update_allowed_regions(allowed_regions: np.ndarray, position: tuple[int, int], radius: int = 0, dimensions: Optional[
                           list[int]] = None, shape_mask: Optional[np.ndarray] = None, angle: Optional[float] = None) -> None:
    # TODO
    pass
