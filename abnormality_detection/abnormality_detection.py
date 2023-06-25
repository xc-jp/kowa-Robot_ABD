"""Module Description"""
from typing import Any, Dict, List, Optional
from itertools import combinations
import json
from pathlib import Path
import cv2
import imutils


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


def judge_within_radius_range(objects: List[Dict[str, Any]], min_allowed_separation_dist: Optional[float] = None,
                              max_allowed_separation_dist: Optional[float] = None) -> List[Dict[str, Any]]:
    if min_allowed_separation_dist is None and max_allowed_separation_dist is None:
        raise ValueError(
            "At least on of min_allowed_separation_dist or max_allowed_separation_dist must be provided")
    for object in objects:
        object["too_close"] = False
        object["too_far"] = False
        object["closest_distance"] = np.inf

    # for a_index, b_index, pair in combinations(enumerate(objects), 2):
    for ((a_index, _), (b_index, _)) in combinations(enumerate(objects), 2):
        d = distance(objects[a_index], objects[b_index])
        objects[a_index]["closest_distance"] = min(objects[a_index]["closest_distance"], d)
        objects[b_index]["closest_distance"] = min(objects[b_index]["closest_distance"], d)

    for object in objects:
        if min_allowed_separation_dist is not None and object["closest_distance"] < min_allowed_separation_dist:
            object["too_close"] = True
        if max_allowed_separation_dist is not None and object["closest_distance"] > max_allowed_separation_dist:
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


def judge_image(model: Dict[str, Any], image: Image.Image, device: torch.device, grasping_conf_threshold: float = 0,
                allowed_regions: Optional[np.ndarray] = None, min_allowed_separation_dist: Optional[float] = None,
                max_allowed_separation_dist: Optional[float] = None, angle_range: Optional[tuple[float, float]] = None
                ) -> List[Dict[str, Any]]:

    if (allowed_regions is None and min_allowed_separation_dist is None and
            max_allowed_separation_dist is None and angle_range is None):
        raise ValueError("At least one of allowed_regions or min_allowed_separation_dist or"
                         " max_allowed_separation_dist or angle_range must be provided")

    # call grasping inference
    detected_objects, visualization = grasping_inference(
        model, image, device, conf_threshold=grasping_conf_threshold)
    if allowed_regions is not None:
        judge_positions(detected_objects, allowed_regions)
    if min_allowed_separation_dist or max_allowed_separation_dist:
        judge_within_radius_range(detected_objects, min_allowed_separation_dist=min_allowed_separation_dist,
                                  max_allowed_separation_dist=max_allowed_separation_dist)
    if angle_range:
        judge_angle(detected_objects, min_angle=angle_range[0], max_angle=angle_range[1])
    return detected_objects, visualization


def plot_results(judged_items: List[Dict[str, Any]], draw_image: Image.Image):
    draw = ImageDraw.Draw(draw_image)
    for item in judged_items:
        x = round(item['x']) - 1
        y = round(item['y']) - 1
        if item.get('too_far'):
            # mark position in red rectangle
            draw.rectangle([x - 7, y - 5, x + 7, y + 5], outline="red")

        if item.get('too_close'):
            # mark position in red rectangle
            draw.rectangle([x - 5, y - 7, x + 5, y + 7], outline="purple")

        if not item.get('judge_angle', True):
            # mark position in yellow square
            draw.rectangle((x - 3, y - 3, x + 3, y + 3), fill='orange', outline=None)

        if not item.get('inside_allowed_region', True):
            # mark position in red circle
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='red', outline=None)

        if (item.get('inside_allowed_region', True) and not item.get('too_far') and not item.get('too_close')
                and item.get('judge_angle', True)):
            # mark position in blue
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='blue', outline=None)
    return draw_image


def create_allowed_regions(video: list[np.ndarray], model: dict[str, Any], device: torch.device,
                           radius: int = 0, dimensions: Optional[list[int]] = None, conf_threshold: float = 0,
                           shape_mask: Optional[np.ndarray] = None, angle: Optional[int] = 0) -> np.ndarray:
    # UNDER CONSTRUCTION ...

    radius_given, dimensions_given, shape_mask_given = 0, 0, 0

    if radius > 0:
        radius_given = 1
    if dimensions is not None:
        dimensions_given = 1
    if shape_mask is not None:
        shape_mask_given = 1
    input_score = radius_given + dimensions_given + shape_mask_given
    if input_score > 1:
        raise ValueError

    created_map = np.zeros([391, 568])
    for frame in video:
        # need to change frame from ndarray to image.image
        objects_in_frame, _ = grasping_inference(
            model, Image.fromarray(frame), device, conf_threshold)
        for object in objects_in_frame:
            position = (int(object['x']), int(object['y']))
            # or wouldn't it be better to call the update function here?
            update_allowed_regions(
                created_map,
                position[0],
                position[1],
                radius=radius,
                dimensions=dimensions,
                shape_mask=shape_mask,
                angle=angle)
            
    return created_map


def update_allowed_regions(allowed_regions: np.ndarray, x: int, y: int, radius: int = 0, dimensions: Optional[
                           list[int]] = None, shape_mask: Optional[np.ndarray] = None, angle: Optional[float] = None) -> None:
    # ONGOING
    
    if shape_mask:
        if not angle:
            raise ValueError
        rotated_shape_mask = imutils.rotate(shape_mask, angle)
        cv2.drawContours(allowed_regions, (x, y), rotated_shape_mask, 0, 255, cv2.FILLED)
    elif dimensions:
        if not angle:
            raise ValueError
        rotated_dimensions = imutils.rotate(dimensions, angle)
        cv2.drawContours(allowed_regions, (x, y), rotated_dimensions, 0, 255, cv2.FILLED)
    elif radius > 0:
        cv2.circle(allowed_regions, (x,y), radius, 255, thickness=cv2.FILLED)
    else:
        allowed_regions[y][x] = 255


def create_allowed_angles(video: list[np.ndarray] ) -> tuple[float, float]:
    #TODO
    return(tuple(0,0))