"""Module Description"""
from typing import Any, Dict, List
from itertools import combinations
import json
from pathlib import Path


import numpy as np
import torch
from PIL import Image, ImageDraw
from math import sqrt

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


def judge_angle(positions: List[Dict[str, Any]], min_angle: float,
                max_angle: float) -> List[Dict[str, Any]]:
    # evaluates whether or not the object's angle is within allowed interval
    for position in positions:
        position['judge_angle'] = min_angle <= position['beta'] <= max_angle
    return positions


def distance(object1: Dict[str, Any], object2: Dict[str, Any]) -> float:
    x1 = object1['x']
    y1 = object1['y']
    x2 = object2['x']
    y2 = object2['y']
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)


def judge_within_radius_range(objects: List[Dict[str, Any]], min_allowed_separation_dist: float,
                              max_allowed_separation_dist: float) -> List[Dict[str, Any]]:
    for object in objects:
        object["too_close"] = False
        object["too_far"] = False
        object["closest_distance"] = 10000

    # for a_index, b_index, pair in combinations(enumerate(objects), 2):
    for ((a_index, _), (b_index, _)) in combinations(enumerate(objects), 2):
        d = distance(objects[a_index], objects[b_index])
        objects[a_index]["closest_distance"] = min(objects[a_index]["closest_distance"], d)
        objects[b_index]["closest_distance"] = min(objects[b_index]["closest_distance"], d)

    for object in objects:
        if object["closest_distance"] < min_allowed_separation_dist:
            object["too_close"] = True
        if object["closest_distance"] > max_allowed_separation_dist:
            object["too_far"] = True

    return objects


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
