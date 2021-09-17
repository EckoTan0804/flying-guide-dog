from yacs.config import CfgNode as CN
from argparse import Namespace

_C = CN()

# ---------- General config ----------
_C.READY = False
_C.LOG_DIR = "log"
_C.HALF_PRECISION = False  # half precision (fp16)
_C.GPUS = (0,)

# ---------- color ----------
_C.COLOR = CN()
_C.COLOR.RED = [255, 0, 0]
_C.COLOR.GREEN = [0, 255, 0]
_C.COLOR.BLUE = [0, 0, 255]

# ---------- Image ----------
_C.IMAGE = CN()
_C.IMAGE.SIZE = [720, 480]  # size of input image (width x height)

_C.IMAGE.LINE = CN()
_C.IMAGE.LINE.THICKNESS = 2
_C.IMAGE.LINE.COLOR = [0, 255, 0]

_C.IMAGE.FONT = CN()
_C.IMAGE.FONT.THICKNESS = 2
_C.IMAGE.FONT.SCALE = 0.6
_C.IMAGE.FONT.COLOR = [255, 0, 0]

_C.IMAGE.CIRCLE = CN()
_C.IMAGE.CIRCLE.RADIUS = 20
_C.IMAGE.CIRCLE.COLOR = [0, 255, 0]


# ---------- Drone control ----------
_C.DRONE = CN()
_C.DRONE.TIME_DELAY = 1
_C.DRONE.HEIGHT = 120  # cm
_C.DRONE.HEIGHT_THRESHOLD = 10  # cm
_C.DRONE.SPEED = 35
_C.DRONE.ACCELEARATION = 10
_C.DRONE.NUM_SPLIT = 3
_C.DRONE.THRESHOLD = 50
_C.DRONE.SENSITIVITY = 3
_C.DRONE.TRANSLATION_VLEOCITY_THRE = 10
_C.DRONE.YAW_VELOCITY_SCALE = [-25, -15, 0, 15, 25]
_C.DRONE.DISPLAY_INFO = True

_C.DRONE.HIGH_PASS = True
_C.DRONE.HIGH_PASS_ALPHA = 0.1
_C.DRONE.LOW_PASS = True
_C.DRONE.LOW_PASS_ALPHA = 0.5

# Choose image for displaying, possible choice are:
# - original: original frame captured by drone's camera
# - colorize: colorized segmentation prediction
# - blend: composition of colorized segmentatio prediction and original frame
# - annotation: original frame with annotation (centroid, contours, bounding boxes)
_C.DRONE.DISPLAY_IMAGE = "annotation"


# ---------- Segmentation ----------
_C.SEGMENTATION = CN()
_C.SEGMENTATION.EXECUTE = True
_C.SEGMENTATION.PALETTE = "cityscapes"
_C.SEGMENTATION.MODEL = "mobilenet_v3_large"  # options: mobilenet_v3_large, mobilenet_v3_small, swaftnet
_C.SEGMENTATION.BLEND = True  # combine original image and colorized segmentation mask (with specified alpha channel)
_C.SEGMENTATION.ALPHA = 0.6  # interpolation factor for blending
_C.SEGMENTATION.RETURN_PROB = False  # whether to return the predicted log-probabilities of each class

_C.SEGMENTATION.COLOR = CN()
_C.SEGMENTATION.COLOR.TRAFFIC_LIGHT = [250, 170, 30]
_C.SEGMENTATION.COLOR.SIDEWALK = [244, 35, 232]
_C.SEGMENTATION.COLOR.CROSSWALK_PLAIN = [140, 140, 200]
_C.SEGMENTATION.COLOR.CROSSWALK_ZEBRA = [200, 128, 128]
_C.SEGMENTATION.COLOR.ROAD = [128, 64, 128]  # color of road in Cityscapes


# ---------- Traffic light recognition ----------
_C.TRAFFIC_LIGHT_CLASSIFICATION = CN()
_C.TRAFFIC_LIGHT_CLASSIFICATION.MODEL = "resnet18"  # options: resnet18, simple_cnn
_C.TRAFFIC_LIGHT_CLASSIFICATION.IMAGE_SIZE = [224, 224]  # [w, h]
_C.TRAFFIC_LIGHT_CLASSIFICATION.NUM_AVERAGE = 7


# ---------- Extras ----------
_C.EXTRAS = CN()
_C.EXTRAS.IMAGENET_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
_C.EXTRAS.IMAGENET_STD = [0.229, 0.224, 0.225]  # ImageNet standard deviation


def get_cfg_defaults() -> CN:
    """
    Get a yacs CfgNode object with default values.

    Returns
    -------
    CN
        Default yacs CfgNode 
    """
    return _C.clone()


def update_config(cfg: CN, args: Namespace) -> None:
    """
    Update default config based on arguments.

    Parameters
    ----------
    cfg : CN
        Default config
    args : Namespace
        Command line arguments
    """
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # Further handling of cfg based on CLI arguments
    if args.ready:
        cfg.READY = True

    cfg.freeze()
