import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import csv

from yacs.config import CfgNode as CN
import cv2
import numpy as np
from djitellopy import Tello
import pyttsx3

WHITE_COLOR = np.array([255, 255, 255], dtype=np.uint8)
BLACK_COLOR = np.array([0, 0, 0], dtype=np.uint8)


def create_logger(cfg: CN) -> logging.Logger:
    """
    Create logger 

    Parameters
    ----------
    cfg : CN
        Yacs config node

    Returns
    -------
    logging.Logger
        Logger
    """
    # Set up Log dir
    root_log_dir = Path(cfg.LOG_DIR)
    if not root_log_dir.exists():
        print(f"=> Creating {str(root_log_dir)}")
        root_log_dir.mkdir(parents=True)

    # Set up log file
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_file = f"{time_str}.log"
    final_log_file = root_log_dir.joinpath(log_file)

    # Set up logger
    logging_format = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(final_log_file), format=logging_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    return logger


def obj_to_color(cfg: CN) -> Dict:
    """
    Mapping objects to their specified color

    Parameters
    ----------
    cfg : CN
        Yacs config node

    Returns
    -------
    Dict
        Dictionary mapping objects to colors
    """
    obj2color_dict = {
        "sidewalk": cfg.SEGMENTATION.COLOR.SIDEWALK,
        "crosswalk plain": cfg.SEGMENTATION.COLOR.CROSSWALK_PLAIN,
        "crosswalk zebra": cfg.SEGMENTATION.COLOR.CROSSWALK_ZEBRA,
        "traffic light": cfg.SEGMENTATION.COLOR.TRAFFIC_LIGHT,
        "road": cfg.SEGMENTATION.COLOR.ROAD,
    }
    return obj2color_dict


def mask_target_obj(label_img: np.ndarray, objs: List[str], cfg: CN) -> np.ndarray:
    """
    Filter out target objects from a colorized segmentation prediction based on their specified colors 

    Parameters
    ----------
    label_img : np.ndarray
        Segmentation prediction, each object is colorized with its pre-defined color
    objs : List[str]
        Target objects
    cfg : CN
        Yacs config node

    Returns
    -------
    np.ndarray
        Grayscale mask, in which pixels of target objects are assigned white, while other pixels are assigned black
    """
    mask = np.zeros(label_img.shape, dtype=np.uint8)
    conditions = np.all(mask, axis=-1)

    # Find and filter objs based on their corresponding color
    obj_color_dict = obj_to_color(cfg)
    for obj in objs:
        obj_color = obj_color_dict[obj]
        conditions |= np.all(label_img == obj_color, axis=-1)  # False | True = True, False | False = False

    # Set target pixel to white
    indicies = np.where(conditions)
    mask[indicies] = WHITE_COLOR

    # Post-processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    return mask


def extract_traffic_light(label_img: np.ndarray, cfg: CN) -> np.ndarray:
    """
    Mask out traffic light from a colorized segmentation prediction 

    Parameters
    ----------
    label_img : np.ndarray
        Segmentation model prediction, each object is colorized with its pre-defined color
    cfg : CN
        Yacs config node

    Returns
    -------
    np.ndarray
        Grayscale mask, in which color of traffic lights pixels is white, while color of other pixels is black
    """
    return mask_target_obj(label_img, ["traffic light"], cfg)


def extract_walkable_area(label_img: np.ndarray, cfg: CN) -> np.ndarray:
    """
    Mask out walkable area (sidewalk, crosswalk zebra, and crosswalk plain) from a colorized segmentation prediction

    Parameters
    ----------
    label_img : np.ndarray
        Segmentation model prediction, each object is colorized with its pre-defined color
    cfg : CN
        Yacs config node

    Returns
    -------
    np.ndarray
        Grayscale mask, in which color of walkable area pixels is white, while color of other pixels is black
    """
    return mask_target_obj(label_img, ["crosswalk plain", "crosswalk zebra", "sidewalk"], cfg)


def extract_road(label_img: np.ndarray, cfg: CN) -> np.ndarray:
    """
    Mask out road from a colorized segmentation prediction

    Parameters
    ----------
    label_img : np.ndarray
        Segmentation model prediction, each object is colorized with its pre-defined color
    cfg : CN
        Yacs config node

    Returns
    -------
    np.ndarray
        Grayscale mask, in which color of road pixels is white, while color of other pixels is black
    """
    return mask_target_obj(label_img, ["road"], cfg)


def split_img(mask_img: np.ndarray, cfg: CN, show_text: bool = False) -> Tuple[List[np.float64], np.ndarray]:
    """
    Split up mask image into specified numbers of partitions, compute confidence score for each partition

    Parameters
    ----------
    mask_img : np.ndarray
        Grayscale mask
    cfg : CN
        Yacs config node
    show_text : bool, optional
        Whether to show annotation on mask image, by default False

    Returns
    -------
    Tuple[List[np.float64], np.ndarray]
        - List of partitions' confidence scores 
        - Mask image
    """
    num_split = cfg.DRONE.NUM_SPLIT
    img_partitions = np.array_split(mask_img, num_split, axis=1)
    scores = [img_partition.mean() for img_partition in img_partitions]
    mask_img_copy = mask_img.copy()

    if show_text:
        text_scale = cfg.IMAGE.FONT.SCALE
        text_thickness = cfg.IMAGE.FONT.THICKNESS
        text_font = cv2.FONT_HERSHEY_SIMPLEX

        line_x_coordinate = 0
        for i, img_partition in enumerate(img_partitions):
            h, w = img_partition.shape

            # Draw split lines
            line_x_coordinate += w
            cv2.line(
                mask_img_copy, (line_x_coordinate, 0), (line_x_coordinate, h), (255, 0, 0), cfg.IMAGE.LINE.THICKNESS,
            )

            # Annotate text for each partition
            text = f"{scores[i]:.3f}"
            text_w, text_h = cv2.getTextSize(text, text_font, text_scale, text_thickness)[0]
            text_pos = (line_x_coordinate - (w + text_w) // 2, h // 2)
            cv2.putText(
                mask_img_copy, text, text_pos, text_font, text_scale, (255, 0, 0), text_thickness,
            )

    return scores, mask_img_copy


def get_contour_center(contour: np.ndarray) -> Tuple[int, int]:
    """
    Get center point of given contour

    Parameters
    ----------
    contour : np.ndarray
        Given contour

    Returns
    -------
    Tuple[int, int]
        Coordinate of contour's center point
    """
    x, y, w, h = cv2.boundingRect(contour)
    x_center = x + w // 2
    y_center = y + h // 2
    return x_center, y_center


def get_contour_centroid(contour: np.ndarray) -> Tuple[int, int]:
    """
    Get centroid of given contour

    Parameters
    ----------
    contour : np.ndarray
        Given contour

    Returns
    -------
    Tuple[int, int]
        Coordinate of contour's centroid
    """
    moment = cv2.moments(contour)
    x_centorid = int(moment["m10"] / moment["m00"])
    y_centorid = int(moment["m01"] / moment["m00"])
    return np.array([x_centorid, y_centorid])


def estimate_centroid(cur_measurement, last_measurement, last_estimate, cfg):
    cur_estimate = np.zeros(last_estimate)
    if cfg.DRONE.HIGH_PASS:
        cur_estimate = cfg.DRONE.HIGH_PASS_ALPHA * (last_estimate + cur_measurement - last_measurement)
    if cfg.DRONE.LOW_PASS:
        cur_estimate = last_estimate + cfg.DRONE.LOW_PASS_ALPHA * (cur_measurement - last_estimate)
    return cur_estimate


def annotate_bounding_box(
    ori_img: np.ndarray, contours: List[np.ndarray], cfg: CN, crop: bool = False, w_min: int = 32, h_min: bool = 32
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Annotate bounding boxes according to the given contours. Crop image based on these bounding boxes.

    Parameters
    ----------
    ori_img : np.ndarray
        Original image, on which bounding boxes will be annotated
    contours : List[np.ndarray]
        List of contours    
    cfg : CN
        Yacs config node    
    crop : bool, optional
        whether to crop images, by default False
    w_min : int, optional
        minimal width of bounding box, by default 32
    h_min : bool, optional
        minimal height of bounding box, by default 32

    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray]]
        - anno_img: Image annotated with bounding boxes
        - cropped_images: images cropped from original image based on bounding boxes
    """
    cropped_images = []
    anno_img = ori_img.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if crop and w >= w_min and h >= h_min:
            cropped_images.append(ori_img[y : y + h, x : x + w])
            anno_img = cv2.rectangle(anno_img, (x, y), (x + w, y + h), cfg.COLOR.RED, cfg.IMAGE.LINE.THICKNESS)

    return anno_img, cropped_images


def crop_traffic_lights(
    ori_img: np.ndarray, traffic_light_mask: np.ndarray, cfg: CN, crop: bool = False
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Annotate bounding boxes for traffic lights on original image and crop traffic lights out

    Parameters
    ----------
    ori_img : np.ndarray
        Original image
    traffic_light_mask : np.ndarray
        Extracted traffic light mask image
    cfg : CN
        Yacs config node
    crop : bool, optional
        whether to crop traffic lights out, by default False

    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray]]
        - anno_img: Image, on which traffic lights are annotated with bounding boxes
        - cropped_images: traffic lights cropped from original image based on bounding boxes
    """
    contours, _ = cv2.findContours(traffic_light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return annotate_bounding_box(ori_img, contours, cfg, crop)


def annotate_circle(img: np.ndarray, x_circle: int, y_circle: int, cfg: CN) -> np.ndarray:
    """
    Annotate circle on image according to the given position

    Parameters
    ----------
    img : np.ndarray
        Image on which the circle will be annotated
    x_circle : int
        x coordinate of the circle
    y_circle : int
        y coordinate of the circle
    cfg : CN
        Yacs config node

    Returns
    -------
    np.ndarray
        Image annotated with circle
    """
    anno_img = img.copy()
    return cv2.circle(anno_img, (x_circle, y_circle), cfg.IMAGE.CIRCLE.RADIUS, cfg.IMAGE.CIRCLE.COLOR, cv2.FILLED,)


def display(
    ori_img: np.ndarray, colorized: Image, blend: Image, anno_img: np.ndarray, display="annotation"
) -> np.ndarray:
    """
    Choose image to display

    Parameters
    ----------
    ori_img : np.ndarray
        Original image
    colorized : Image
        Colorized segmentation prediction
    blend : Image
        Composition of original image and colorized segmentaiton prediction
    anno_img : np.ndarray
        Original image with annotation (centroid, contours, bounding box, etc)
    display : str, optional
        Selection of display, by default "annotation"

    Returns
    -------
    np.ndarray
        Image to display
    """

    assert display in ["original", "colorize", "blend", "annotation"]

    img_display = ori_img
    if display == "colorize":
        img_display = np.asarray(colorized)
    elif display == "blend":
        img_display = np.asarray(blend)
    elif display == "annotation":
        img_display = anno_img
    return img_display


def display_info(
    img_display: np.ndarray, drone: Tello, velocities: List[int], height: int, traffic_light_info: str, cfg: CN
):
    """
    Display information on the image

    Parameters
    ----------
    img_display : np.ndarray
        Image on which the information to be displayed
    drone : Tello
        DJI Tello drone
    velocities : List[int]
        List of velocities (left-right, forward-backward, up-down, yaw)
    height : int
        Current height of the drone
    cfg : CN
        Yacs config node
    """
    font_thickness = cfg.IMAGE.FONT.THICKNESS
    font_scale = cfg.IMAGE.FONT.SCALE
    font_color = cfg.IMAGE.FONT.COLOR

    # Battery
    battery_info = f"Battery: {drone.get_battery()}"
    cv2.putText(
        img_display, battery_info, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness,
    )

    # Height
    height_info = f"Height: {height}"
    cv2.putText(
        img_display, height_info, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness,
    )

    # Traffic light
    traffic_light_color = cfg.COLOR.RED if traffic_light_info == "Red" else cfg.COLOR.GREEN
    # traffic_light_text = f"Traffic light: {traffic_light_info}" if traffic_light_info != "" else ""
    cv2.putText(
        img_display,
        f"Traffic light: {traffic_light_info}",
        (5, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        traffic_light_color,
        font_thickness,
    )

    # Velocities
    vel_types = [
        "LR",  # LR: Left-Right
        "FB",  # FB:Forward-Backward
        "UD",  # UD: Up-down
        "Yaw",
    ]
    vel_infos = [f"{vel_type}: {vel}" for vel_type, vel in zip(vel_types, velocities.values())]
    y0, dy = 30, 30
    for i, vel_info in enumerate(vel_infos):
        y = y0 + i * dy
        cv2.putText(
            img_display,
            vel_info,
            (cfg.IMAGE.SIZE[0] - 100, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_color,
            font_thickness,
        )


def sound_alarm(voice_engine: pyttsx3.Engine, content: str):
    """
    Speak the specified content

    Parameters
    ----------
    voice_engine : pyttsx3.Engine
        pyttsx3 Engine instance
    content : str
        content to speak
    """
    voice_engine.say(content)
    voice_engine.runAndWait()


def export_data(centroid_measurements, centroid_estimates):
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    for name, centroids in {"measurement": centroid_measurements, "estimate": centroid_estimates}.items():
        csv_file = f"csv/{time_str}_{name}.csv"
        np.savetxt(csv_file, centroids, delimiter=",")

