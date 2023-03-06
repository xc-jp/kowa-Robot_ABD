"""Module Description"""
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

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
                       device: torch.device) -> List[Dict[str, Any]]:
    # calls Grasping Inference
    # returns a list with the detected objects' centers
    prediction_points, visualization_results = grasping_infer.infer(
        model['network'], image, model['input_width'], model['input_height'], device, visualization=True,
        dim_mins=model['dim_mins'], dim_maxs=model['dim_maxs']
    )
    return prediction_points, visualization_results


def is_allowed(x: float, y: float, allowed_regions: np.ndarray) -> bool:
    # evaluates whether or not the object is outside allowed area
    # returns true  if the coordinates is inside allowed region, false otherwise
    # allowed_regions is a black&white image (0 to 1 values), 0 for allowed coordinates
    return allowed_regions[round(x)][round(y)] != 0


def judge_positions(positions: Dict[str, Any], allowed_regions: np.ndarray) -> List[Dict[str, Any]]:
    # TODO implement method
    # It should return a list with the positions with the addition of a new key: "allowed_region": true/false
    raise NotImplementedError()


def judge_image(model: Dict[str, Any], image: Image.Image, allowed_regions: np.ndarray, device: torch.device
                ) -> List[Dict[str, Any]]:
    detected_objects = grasping_inference(model, image, device)
    return judge_positions(detected_objects, allowed_regions)
