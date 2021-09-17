import torch
import torch.nn as nn
import torchvision.models as models

from .classify import CLASSIFICATION_MODEL_REGISTRY
from .simple_cnn import Net


@CLASSIFICATION_MODEL_REGISTRY.register("resnet18")
def build_resnet_18() -> nn.Module:
    resnet18 = models.resnet18(num_classes=5)
    resnet18.load_state_dict(torch.load("weights/traffic_light_classification/resnet18-best.pth"))
    return resnet18


@CLASSIFICATION_MODEL_REGISTRY.register("simple_cnn")
def build_simple_cnn() -> nn.Module:
    simple_cnn = Net()
    simple_cnn.load_state_dict(torch.load("weights/traffic_light_classification/simple_cnn-best.pth"))
    return simple_cnn
