import numpy as np
from PIL import Image
from collections import namedtuple
from typing import List

################################ Cityscapes ################################

cityscapes = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

# Based on https://github.com/mcordts/cityscapesScripts
CityscapesClass = namedtuple(
    "CityscapesClass",
    ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
)

_classes = [
    CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
    CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
    CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
    CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
    CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
    CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
    CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
    CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
    CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
    CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
    CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
]


def get_cityscapes_palette() -> List[int]:
    """
    Get segmentation palette for cityscapes dataset.

    Returns
    -------
    List[int]
        Palette for cityscapes dataset
    """
    train_classes = [_cls for _cls in _classes if _cls.train_id != 255]
    palette = []
    for _cls in train_classes:
        palette.extend(list(_cls.color))
    return palette


################################


################################ PASCAL VOC ################################

_PASCAL_VOC = [
    (0, 0, 0),  # 0=background
    (128, 0, 0),  # 1=aeroplane
    (0, 128, 0),  # 2=bicycle
    (128, 128, 0),  # 3=bird
    (0, 0, 128),  # 4=boat
    (128, 0, 128),  # 5=bottle
    (0, 128, 128),  # 6=bus
    (128, 128, 128),  # 7=car
    (64, 0, 0),  # 8=cat
    (192, 0, 0),  # 9=chair
    (64, 128, 0),  # 10=cow
    (192, 128, 0),  # 11=dining table
    (64, 0, 128),  # 12=dog
    (192, 0, 128),  # 13=horse
    (64, 128, 128),  # 14=motorbike
    (192, 128, 128),  # 15=person
    (0, 64, 0),  # 16=potted plant
    (128, 64, 0),  # 17=sheep
    (0, 192, 0),  # 18=sofa
    (128, 192, 0),  # 19=train
    (0, 64, 128),  # 20=tv/monitor
]

################################


################################ Modified cityscapes paletted used by swaftnet ################################

_SWAFTNET_CITYSCAPES = [
    [153, 153, 153],
    [210, 170, 100],
    [220, 220, 220],
    [250, 170, 30],
    [0, 0, 142],
    [0, 0, 70],
    [119, 11, 32],
    [0, 0, 230],
    [0, 60, 100],
    [220, 220, 0],
    [192, 192, 192],
    [128, 64, 128],
    [244, 35, 232],
    [170, 170, 170],
    [140, 140, 200],
    [128, 64, 255],
    [196, 196, 196],
    [190, 153, 153],
    [102, 102, 156],
    [70, 70, 70],
    [220, 20, 60],
    [255, 0, 0],
    [70, 130, 180],
    [107, 142, 35],
    [152, 251, 152],
    [255, 255, 255],
    [200, 128, 128],
    [0, 0, 0],
]

################################

################################ Mapillary Vistas ################################
_MAPILLARY_VISTAS = [
    [165, 42, 42],
    [0, 192, 0],
    [196, 196, 196],
    [190, 153, 153],
    [180, 165, 180],
    [90, 120, 150],
    [102, 102, 156],
    [128, 64, 255],
    [140, 140, 200],
    [170, 170, 170],
    [250, 170, 160],
    [96, 96, 96],
    [230, 150, 140],
    [128, 64, 128],
    [110, 110, 110],
    [244, 35, 232],
    [150, 100, 100],
    [70, 70, 70],
    [150, 120, 90],
    [220, 20, 60],
    [255, 0, 0],
    [255, 0, 100],
    [255, 0, 200],
    [200, 128, 128],
    [255, 255, 255],
    [64, 170, 64],
    [230, 160, 50],
    [70, 130, 180],
    [190, 255, 255],
    [152, 251, 152],
    [107, 142, 35],
    [0, 170, 30],
    [255, 255, 128],
    [250, 0, 30],
    [100, 140, 180],
    [220, 220, 220],
    [220, 128, 128],
    [222, 40, 40],
    [100, 170, 30],
    [40, 40, 40],
    [33, 33, 33],
    [100, 128, 160],
    [142, 0, 0],
    [70, 100, 150],
    [210, 170, 100],
    [153, 153, 153],
    [128, 128, 128],
    [0, 0, 80],
    [250, 170, 30],
    [192, 192, 192],
    [220, 220, 0],
    [140, 140, 20],
    [119, 11, 32],
    [150, 0, 255],
    [0, 60, 100],
    [0, 0, 142],
    [0, 0, 90],
    [0, 0, 230],
    [0, 80, 100],
    [128, 64, 64],
    [0, 0, 110],
    [0, 0, 70],
    [0, 0, 192],
    [32, 32, 32],
    [120, 10, 10],
    [0, 0, 0],
]


################################


def get_palette(palette_name):
    palette = []
    for cls in palette_name:
        palette.extend(list(cls))
    return palette


all_palettes = {
    "pascal voc": get_palette(_PASCAL_VOC),
    "cityscapes": get_cityscapes_palette(),
    "swaftnet_cityscapes": get_palette(_SWAFTNET_CITYSCAPES),
    "mapillary": get_palette(_MAPILLARY_VISTAS),
}


def colorize(seg_pred: np.ndarray, palette: str = "cityscapes", convert=True) -> Image.Image:
    """
    Colorize segementation mask

    Parameters
    ----------
    seg_pred : np.ndarray
        Segmentation mask as a 2D numpy array of integers [0..classes-1]
    palette : str, optional
        Palette to applay, by default "cityscapes"

    Returns
    -------
    Image.Image
        Colorized segementation mask
    """
    mask_img = Image.fromarray(seg_pred.astype(np.uint8)).convert("P") if convert else seg_pred
    mask_img.putpalette(all_palettes[palette])
    mask_img = mask_img.convert("RGB")
    return mask_img


def blend(input_img: Image.Image, colorized_seg_pred: Image.Image, alpha: float = 0.5) -> Image.Image:
    """
    Blend an input image with its colorized segmentation labels

    Parameters
    ----------
    input_img : Image.Image
        Original input image
    colorized_seg_pred : Image.Image
        Colorized segmenation mask
    alpha: float
        Interpolation alpha factor, by default 0.5

    Returns
    -------
    Image
        Image composing input image and colorized segmentation mask with specified alpha channel
    """
    return Image.blend(input_img, colorized_seg_pred, alpha)
