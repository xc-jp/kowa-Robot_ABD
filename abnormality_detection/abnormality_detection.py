"""Module Description"""
from typing import Any, Dict, List
from datetime import datetime
import os
from itertools import combinations
from operator import itemgetter

import numpy as np
import torch
from PIL import Image, ImageDraw
from math import sqrt

from grasping import infer as grasping_infer
from src.networks.build_network import build_model as build_grasping_model
from pathlib import Path
import json


def load_model(model_path: Path, device: torch.device) -> Dict[str, Any]:
    # loads pytorch model weights
    with open(model_path.joinpath('obj', 'build_parameter.json'), encoding='utf-8') as f:
        build_parameters = json.load(f)
    return _load_grasping(model_path, build_parameters, device)


def _load_grasping(
        path: Path, build_parameters: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    # loads the grasping model from the device
    input_width = build_parameters['input_width']
    input_height = build_parameters['input_height']
    nb_classes = build_parameters['nb_classes']
    subdivs = build_parameters['subdivs']
    network_name = build_parameters['network']
    dim_mins = tuple(build_parameters['dim_mins'])
    dim_maxs = tuple(build_parameters['dim_maxs'])

    network_path = path.joinpath('obj', 'model.pth')
    network = build_grasping_model(network_name, model_path=network_path, eval_mode=True,
                                   image_sizes=(input_height, input_width), nb_classes=nb_classes, subdivs=subdivs)
    network = network.to(device)

    return {
        'method': 'grasping',
        'network': network,
        'input_width': input_width,
        'input_height': input_height,
        'dim_mins': dim_mins,
        'dim_maxs': dim_maxs,
    }


def grasping_inference(model: Dict[str, Any], image: Image.Image,
                       device: torch.device, conf_threshold=0) -> List[Dict[str, Any]]:
    # calls Grasping Inference
    # returns a list with the detected objects' centers
    prediction_points, visualization_results = grasping_infer.infer(
        model['network'], image, model['input_width'], model['input_height'], device, visualization=True,
        dim_mins=model['dim_mins'], dim_maxs=model['dim_maxs']
    )
    new_list = []
    for point in prediction_points:
        if point['confidence'] >= conf_threshold:
            new_list.append(point)
    return new_list, visualization_results


def is_allowed(x: float, y: float, allowed_regions: np.ndarray) -> bool:
    # evaluates whether or not the object is outside allowed area
    # returns true  if the coordinates is inside allowed region, false otherwise
    # allowed_regions is a B&W image (ideally 0-1 values), 0 for allowed coordinates
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
                    allowed_regions: np.ndarray) -> List[Dict[str, Any]]:
    # returns a list of detected objects with the additional key "allowed_region": true/false
    for position in positions:
        position["judge"] = is_allowed(
            position['x'],
            position['y'],
            allowed_regions)
    return positions


def judge_image(model: Dict[str, Any], image_path: Path, allowed_regions: np.ndarray,
                device: torch.device, conf_threshold: float = 0) -> List[Dict[str, Any]]:
    image = Image.open(image_path)
    # call the grasping inference function and return the objects' list
    detected_objects, visualization = grasping_inference(model, image, device, conf_threshold)

    # save visualization_results as inputimage_timestamp.jpg format (under same path as the image)
    dt = datetime.now()
    ts = str(datetime.timestamp(dt))
    no_extension = os.path.splitext(image_path)[0]
    # visualization_results.show()
    visualization.save(f"{no_extension}_{ts}.jpg")
    return judge_positions(detected_objects, allowed_regions)


def plot_object(judged_items: List[Dict[str, Any]], allowed_regions_rgb: Image.Image):
    draw = ImageDraw.Draw(allowed_regions_rgb)
    for item in judged_items:
        x = round(item['x']) - 1
        y = round(item['y']) - 1
        if item['judge']:
            # mark position in blue
            draw.regular_polygon((x, y, 3), 6, rotation=0, fill='blue', outline=None)
        else:
            # mark position in red
            draw.regular_polygon((x, y, 3), 6, rotation=0, fill='red', outline=None)
    return allowed_regions_rgb
