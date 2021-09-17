import logging
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
from djitellopy import Tello
from yacs.config import CfgNode as CN

from . import _init_paths
from utils.utils import (
    get_contour_center,
    get_contour_centroid,
    estimate_centroid,
    extract_walkable_area,
    annotate_bounding_box,
    annotate_circle,
    split_img,
)

logger = logging.getLogger(__name__)

centroid_measurements = []
centroid_estimates = []


def init_drone(cfg) -> Tello:
    """
    Initialize DJI Tello drone 

    Returns
    -------
    [Tello]
        Initialized Tello drone which connects computer with wifi and turns on its camera.
    """
    tello = Tello()
    tello.connect()

    battery = tello.get_battery()
    logger.info(f"Battery: {battery}")
    if battery < 30:
        logger.warning("Low battery!!!")

    # Turn on drone's camera
    tello.streamoff()
    tello.streamon()

    logger.info("Drone is initialized.")

    w, h = cfg.IMAGE.SIZE
    centroid_measurements.append(np.array([w // 2, h // 2], dtype=np.uint8))
    centroid_estimates.append(np.array([w // 2, h // 2], dtype=np.uint8))

    return tello


def get_control_code(scores: List[int], cfg: CN) -> List[int]:
    """
    Compute control codes for yaw velocity adjustment based on scores of mask partitions

    Parameters
    ----------
    scores : List[int]
        Scores of mask partitions
    cfg : CN
        Yacs config node        

    Returns
    -------
    List[int]
        Control codes for yaw velocity adjustment
    """
    threshold = cfg.DRONE.THRESHOLD
    control_codes = [1 if score > threshold else 0 for score in scores]
    return control_codes


def get_translation_velocity(ori_img: np.ndarray, walkable_area_mask: np.ndarray, cfg: CN) -> Tuple[np.ndarray, int]:
    """
    Compute translation (left-right) velocity to keep the drone in the middle of walkable area

    Parameters
    ----------
    ori_img : np.ndarray
        Original image
    walkable_area_mask : np.ndarray
        Extracted walkable area mask
    cfg : CN
        Yacs config node

    Returns
    -------
    Tuple[np.ndarray, int]
        - anno_img: Original image with contour and centroid annotation
        - translation_velocity: translation (left-right) velocity to keep the drone in the middle of walkable area
    """
    anno_img = ori_img.copy()
    translation_velocity = 0
    contours, _ = cv2.findContours(walkable_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        biggest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(anno_img, biggest_contour, -1, cfg.IMAGE.LINE.COLOR, cfg.IMAGE.LINE.THICKNESS)

        # Approximate with polygon
        # epsilon = 0.1 * cv2.arcLength(biggest_contour, True)
        # approximation = cv2.approxPolyDP(biggest_contour, epsilon, True)
        # cv2.polylines(anno_img, [approximation], True, cfg.IMAGE.LINE.COLOR, cfg.IMAGE.LINE.THICKNESS)

        # Bounding rectangle
        anno_img, _ = annotate_bounding_box(anno_img, [biggest_contour], cfg)

        # Center/centroid point
        cur_centroid_measurement = get_contour_centroid(biggest_contour)
        cur_centroid_estimate = estimate_centroid(
            cur_centroid_measurement, centroid_measurements[-1], centroid_estimates[-1], cfg
        ).astype(int)
        centroid_measurements.append(cur_centroid_measurement)
        centroid_estimates.append(cur_centroid_estimate)
        x_center, y_center = cur_centroid_estimate
        anno_img = annotate_circle(anno_img, x_center, y_center, cfg)

        # Compute translation velocity based on image width and center point of walkable area
        img_width = cfg.IMAGE.SIZE[0]
        translation_velocity = (x_center - img_width // 2) // cfg.DRONE.SENSITIVITY
        translation_velocity_threshold = cfg.DRONE.TRANSLATION_VLEOCITY_THRE
        translation_velocity = int(
            np.clip(translation_velocity, -translation_velocity_threshold, translation_velocity_threshold)
        )

    return anno_img, translation_velocity


def get_yaw_velocity(walkable_area_mask: np.ndarray, cfg: CN) -> int:
    """
    Compute yaw velocity to keep the drone in the middle of walkable area

    Parameters
    ----------
    walkable_area_mask : np.ndarray
        Extracted walkable area mask
    cfg : CN
        Yacs config node

    Returns
    -------
    int
        Yaw velocity to keep the drone in the middle of walkable area
    """
    yaw_vel_scale = cfg.DRONE.YAW_VELOCITY_SCALE
    confidence_scores, _ = split_img(walkable_area_mask, cfg)
    left, center, right = get_control_code(confidence_scores, cfg)
    if left == right:  # 000, 010, 101, 111
        yaw_vel = yaw_vel_scale[2]
    elif left > right:  # 100, 110
        yaw_vel = yaw_vel_scale[0] if left > center else yaw_vel_scale[1]
    elif left < right:  # 001, 011
        yaw_vel = yaw_vel_scale[4] if center < right else yaw_vel_scale[3]
    return yaw_vel


def is_at_target_height(cur_height: int, cfg: CN) -> bool:
    """
    Determine whether drone is at target height.

    Parameters
    ----------
    cur_height : int
        Current height of drone
    cfg : CN
        Yacs config node    

    Returns
    -------
    bool
        True, if the drone is at the range of target height, otherwise False
    """
    target_height = cfg.DRONE.HEIGHT
    target_height_thre = cfg.DRONE.HEIGHT_THRESHOLD
    return target_height - target_height_thre <= cur_height and cur_height <= target_height + target_height_thre


def get_maintain_flight_height_velocity(current_height: int, cfg: CN) -> int:
    """
    Compute up-down velocity to keep the drone at range of target flight height

    Parameters
    ----------
    current_height : int
        Current height of drone
    cfg : CN
        Yacs config node

    Returns
    -------
    int
        Up-down velocity to maintain the drone at target flight height
    """
    target_height = cfg.DRONE.HEIGHT
    velocity = math.floor((target_height - current_height) / target_height * cfg.DRONE.SPEED)
    return velocity
