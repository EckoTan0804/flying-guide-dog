import logging

import torch
import torch.nn as nn

from .segment import SEG_MODEL_REGISTRY
from fastseg import MobileV3Large, MobileV3Small
from .swaftnet.resnet.resnet_single_scale import resnet18
from .swaftnet.swaftnet import Net
from .SegFormer.mmseg.apis import init_segmentor

logger = logging.getLogger(__name__)


@SEG_MODEL_REGISTRY.register("mobilenet_v3_large")
def build_mobilenet_v3_large() -> nn.Module:
    """
    Get MobileNet V3 large pre-trained on Cityscapes Dataset
    (More see: https://github.com/ekzhang/fastseg)

    Returns
    -------
    nn.Module
        MobileNet V3 large pre-trained on Cityscapes Dataset
    """
    return MobileV3Large.from_pretrained()


@SEG_MODEL_REGISTRY.register("mobilenet_v3_small")
def build_mobilenet_v3_small() -> nn.Module:
    """
    Get MobileNet V3 small pre-trained on Cityscapes Dataset
    (More see: https://github.com/ekzhang/fastseg)

    Returns
    -------
    nn.Module
        MobileNet V3 small pre-trained on Cityscapes Dataset
    """
    return MobileV3Small.from_pretrained()


@SEG_MODEL_REGISTRY.register("swaftnet")
def build_swaftnet() -> nn.Module:
    resnet = resnet18(pretrained=True, efficient=False, use_bn=True)
    # TODO: Adjust size and num_classes based on requirement
    model = Net(resnet, size=(512, 1024), num_classes=28)  # Note: size needs to match IMAGE.SIZE in config file
    model = torch.nn.DataParallel(model)

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()

        # for a,b in zip(own_state.keys(),state_dict.keys()):
        #     print(a,'      ',b)
        # print('-----------')

        for name, param in state_dict.items():
            # print('#####', name)
            name = name[7:]
            if name not in own_state:
                print("{} not in own_state".format(name))
                continue
            # if name not in except_list:
            own_state[name].copy_(param)

        return model

    weight_path = "../weights/segmentation/swaftnet-best.pth"
    model = load_my_state_dict(model, torch.load(weight_path))
    logger.info("Load swaftnet successfully")

    return model


@SEG_MODEL_REGISTRY.register("segformer-b0")
def build_segformer_b0() -> nn.Module:
    checkpoint = "weights/segmentation/SegFormer_b0_Mapillary.pth"
    config_file = "models/segmentation/segformer.b0.768x768.mapillary.160k.py"
    model = init_segmentor(config_file, checkpoint)
    return model
