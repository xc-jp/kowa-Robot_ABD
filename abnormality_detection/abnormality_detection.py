"""Module Description"""
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image


def load_model(model_path: str, device: torch.device) -> Dict[str, Any]:
    # TODO load pytorch model weights
    raise NotImplementedError()


def grasping_inference(model: Dict[str, Any], image: Image.Image,
                       device: torch.device) -> List[Dict[str, Any]]:
    # TODO implement Grasping Inference Call
    # It should return a list with the detected objects centers
    raise NotImplementedError()


def is_allowed(x: float, y: float, allowed_regions: np.ndarray) -> bool:
    # TODO implement method
    # It should return true  if the coordinates is inside allowed region, false otherwise
    raise NotImplementedError()


def judge_positions(positions: List[Dict[str, Any]], allowed_regions: np.ndarray) -> List[Dict[str, Any]]:
    # TODO implement method
    # It should return a list with the positions with the addition of a new key: "allowed_region": true/false
    raise NotImplementedError()


def judge_image(model: Dict[str, Any], image: Image.Image, allowed_regions: np.ndarray, device: torch.device
                ) -> List[Dict[str, Any]]:
    detected_objects = grasping_inference(model, image, device)
    return judge_positions(detected_objects, allowed_regions)
