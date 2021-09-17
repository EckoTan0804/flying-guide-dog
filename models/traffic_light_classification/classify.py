from yacs.config import CfgNode as CN
from PIL import Image
from typing import List, Union
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T

from ..registry import Registry

CLASSIFICATION_MODEL_REGISTRY = Registry()

_TRAFFIC_LIGHT_CLASSES = {
    "others": 0,
    "pedestrian-green": 1,
    "pedestrian-red": 2,
    "vehicle-green": 3,
    "vehicle-red": 4,
}

_TRAFFIC_LIGHT_ID2CLASS = {id: _cls for _cls, id in _TRAFFIC_LIGHT_CLASSES.items()}

_PEDESTRIAN_TRAFFIC_LIGHTS = {_cls: id for _cls, id in _TRAFFIC_LIGHT_CLASSES.items() if "pedestrian" in _cls}


def build_classification_model(model_name: str) -> nn.Module:
    return CLASSIFICATION_MODEL_REGISTRY[model_name]


def predict(
    classification_model: nn.Module, images: List[Union[Image.Image, np.ndarray]], device: torch.device, cfg: CN,
) -> np.ndarray:
    model = classification_model.to(device).eval()

    # Preprocessing
    w, h = cfg.TRAFFIC_LIGHT_CLASSIFICATION.IMAGE_SIZE
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize([h, w]),  # Image size in PyTorch is HxW
            T.Normalize(mean=cfg.EXTRAS.IMAGENET_MEAN, std=cfg.EXTRAS.IMAGENET_STD),
        ]
    )

    inputs = torch.stack([transform(image) for image in images]).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(inputs)
        _, predictions = torch.max(outputs.data, 1)
        predictions = predictions.cpu().numpy()

    return predictions


def predict_one(
    classification_model: nn.Module, image: Union[Image.Image, np.ndarray], device: torch.device, cfg: CN,
) -> np.ndarray:
    return predict(classification_model, [image], device, cfg)[0]


def get_final_prediction(predictions: np.ndarray):
    # keep only the pedestrian traffic lights, and choose the first one (pedestrian traffic light with biggest contour) as final prediction
    final_prediction = None
    predictions = predictions.tolist()
    pedestrian_traffic_light_predictions = [
        prediction for prediction in predictions if prediction in _PEDESTRIAN_TRAFFIC_LIGHTS.values()
    ]
    if pedestrian_traffic_light_predictions:
        final_prediction = _TRAFFIC_LIGHT_ID2CLASS[pedestrian_traffic_light_predictions[0]]
    return final_prediction


def get_traffic_light_class(id):
    return _TRAFFIC_LIGHT_ID2CLASS.get(id)


def get_average_prediction(preds):
    average_pred = list(Counter(preds).keys())[0]  # Choose the traffic light which occurs the most
    preds.clear()
    return average_pred

