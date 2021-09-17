from yacs.config import CfgNode as CN
from PIL import Image
from typing import List, Union
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T

from ..registry import Registry
from .SegFormer.mmseg.apis import inference_segmentor


SEG_MODEL_REGISTRY = Registry()


def build_seg_model(seg_model_name: str) -> nn.Module:
    """
    Build a segmentation model based on the given model name.

    Parameters
    ----------
    seg_model_name : str
        Name of the model

    Returns
    -------
    nn.Module
        Corresponding segmentation model
    """
    return SEG_MODEL_REGISTRY[seg_model_name]


def predict(
    seg_model: nn.Module, images: List[Union[Image.Image, np.ndarray]], device: torch.device, cfg: CN,
) -> np.ndarray:
    """
    Generate segmentation predictions for a batch of images.

    Parameters
    ----------
    seg_model : nn.Module
        Segmentation model
    images : List[Union[Image.Image, np.ndarray]]
        A list of PIL images or Numpy arrays
    device : torch.device
        Device, on which the prediction runs. (It must match the device that the model is currently on)
    cfg : CN
        Yacs config node

    Returns
    -------
    np.ndarray
        - If `cfg.SEGMENTATION.RETURN_PROB == False`, a Numpy array of shape (len(images), height, width) containing predicted classes 
        - If `cfg.SEGMENTATION.RETURN_PROB == True`, a Numpy array of shape (len(images), height, width) containing predicted log-probabilities of each class
    """
    seg_model = seg_model.to(device).eval()

    # Preprocessing
    w, h = cfg.IMAGE.SIZE
    transform = T.Compose(
        [
            T.Resize([h, w]),  # Image size in PyTorch is HxW
            T.ToTensor(),
            T.Normalize(mean=cfg.EXTRAS.IMAGENET_MEAN, std=cfg.EXTRAS.IMAGENET_STD),
        ]
    )

    input = torch.stack([transform(image) for image in images]).to(device)

    # Inference
    with torch.no_grad():
        output = seg_model(input)

    if not cfg.SEGMENTATION.RETURN_PROB:
        output = output.argmax(dim=1)

    return output.detach().cpu().numpy()


def predict_one(
    seg_model: nn.Module, image: Union[Image.Image, np.ndarray], device: torch.device, cfg: CN,
) -> np.ndarray:
    """
    Generate segmentation prediction for a single image.
    (This is a convenient wrapper of `predict()` function that generates segmentation prediction for a single image.)

    Parameters
    ----------
    seg_model : nn.Module
        Segmentation model
    image : Union[Image.Image, np.ndarray]
        A PIL image or Numpy array
    device : torch.device
        Device, on which the prediction runs. (It must match the device that the model is currently on)
    cfg : CN
        Yacs config node

    Returns
    -------
    np.ndarray
        - If `cfg.SEGMENTATION.RETURN_PROB == False`, a Numpy array of shape (1, height, width) containing predicted classes 
        - If `cfg.SEGMENTATION.RETURN_PROB == True`, a Numpy array of shape (1, height, width) containing predicted log-probabilities of each class
    """
    if seg_model.cfg:
        image = np.asarray(image)
        output = np.array(inference_segmentor(seg_model, image)).astype(np.uint8).squeeze()
        output = Image.fromarray(output)
        return output
    else:
        return predict(seg_model, [image], device, cfg)[0]
