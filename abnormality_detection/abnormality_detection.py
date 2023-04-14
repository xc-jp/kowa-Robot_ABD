"""Module Description"""
from typing import Any, Dict, List
import json
from pathlib import Path
# from typing import Any


import numpy as np
import torch
from PIL import Image

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


def grasping_inference(model: dict[str, Any], image: Image.Image,
                       device: torch.device) -> dict[str, Any]:
    # implement Grasping Inference Call
    # It returns a list with the detected objects centers
    # return grasping_infer(model, image, device)
    prediction_points, prediction_image = grasping_infer.infer(
        model["network"], image, model["input_width"], model["input_height"], device, visualization=True,
        dim_mins=model["dim_mins"], dim_maxs=model["dim_maxs"],
    )
    return prediction_points, prediction_image


def is_allowed(x: float, y: float, allowed_regions: np.ndarray) -> bool:
    # evaluates whether or not the object is outside allowed area
    # returns true  if the coordinates is inside allowed region, false otherwise
    # allowed_regions is a B&W image (ideally 0-1 values), 0 for allowed coordinates
    return allowed_regions[round(x - 1)][round(y - 1)] != 0


def judge_positions(positions: List[Dict[str, Any]],
                    allowed_regions: np.ndarray) -> List[Dict[str, Any]]:
    # returns a list with the positions with the addition of a new key: "allowed_region": true/false
    for position in positions:
        position["judge"] = is_allowed(
            position['x'],
            position['y'],
            allowed_regions)
    return positions


def judge_image(model: Dict[str, Any], image: Image.Image, allowed_regions: np.ndarray, device: torch.device
                ) -> List[Dict[str, Any]]:

    # call grasping inference
    detected_objects, visualization = grasping_inference(model, image, device)

    return judge_positions(detected_objects, allowed_regions), visualization
