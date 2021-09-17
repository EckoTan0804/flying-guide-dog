import argparse

import cv2
import numpy as np
from PIL import Image

import torch

import pyttsx3

from models import segmentation, traffic_light_classification
from config import cfg, update_config
from drone import drone
from utils import colorize
from utils.utils import (
    create_logger,
    extract_walkable_area,
    display,
    display_info,
    extract_traffic_light,
    crop_traffic_lights,
    sound_alarm,
    export_data,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Walkable path discovery")
    parser.add_argument("--cfg", help="Experiment config file", required=True, type=str)
    parser.add_argument("--ready", help="Ready for flight", action="store_true")
    parser.add_argument(
        "opts", help="Modify config options using command line", default=None, nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def main():
    # Update config
    args = parse_args()
    update_config(cfg, args)
    print(cfg)

    # global logger
    logger = create_logger(cfg)

    # global tello
    tello = drone.init_drone(cfg)

    voice_engine = pyttsx3.init()

    # video = cv2.VideoCapture(2)  # for webcam

    # Segmentation model
    if cfg.SEGMENTATION.EXECUTE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model = segmentation.build_seg_model(cfg.SEGMENTATION.MODEL)()
        logger.info(f"Seg model: {cfg.SEGMENTATION.MODEL}, Device: {device}")

        classification_model = traffic_light_classification.build_classification_model(
            cfg.TRAFFIC_LIGHT_CLASSIFICATION.MODEL
        )()

    if cfg.READY:
        logger.info("Ready to fly!")

        current_height = 0
        velocities = {
            "left_right_velocity": 0,
            "forward_backward_velocity": cfg.DRONE.SPEED,
            "up_down_velocity": 0,
            "yaw_velocity": 0,
        }

        num_frame = 0
        traffic_light_info = ""
        traffic_light_preds = []

        while True:
            # ret_val, _frame = video.read()  # for webcam

            _frame = tello.get_frame_read().frame
            _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
            _frame = cv2.resize(_frame, tuple(cfg.IMAGE.SIZE))
            img_display = _frame

            if cfg.SEGMENTATION.EXECUTE:
                img = Image.fromarray(_frame)
                seg_prediction = segmentation.predict_one(seg_model, img, device, cfg)

                if not cfg.SEGMENTATION.RETURN_PROB:
                    # Get (colorized) label image prediction
                    model_name = cfg.SEGMENTATION.MODEL
                    convert = "segformer" not in model_name
                    colorized = colorize.colorize(
                        seg_prediction, palette=cfg.SEGMENTATION.PALETTE, convert=convert
                    )  # for display
                    colorized_seg_pred = np.asarray(
                        colorized.copy()
                    )  # for further drone control and traffic light recognition

                    # Drone control adjustment based on (colorized) label image prediction
                    walkable_area_mask = extract_walkable_area(colorized_seg_pred, cfg)
                    anno_frame, left_right_velocity = drone.get_translation_velocity(_frame, walkable_area_mask, cfg)
                    velocities.update({"left_right_velocity": left_right_velocity})
                    yaw_velocity = drone.get_yaw_velocity(walkable_area_mask, cfg)
                    # velocities.update({"yaw_velocity": yaw_velocity})

                    # Handle traffic lights
                    traffic_light_mask = extract_traffic_light(colorized_seg_pred, cfg)
                    anno_frame, cropped_traffic_lights = crop_traffic_lights(
                        anno_frame, traffic_light_mask, cfg, crop=True
                    )
                    if cropped_traffic_lights:
                        predictions = traffic_light_classification.predict(
                            classification_model, cropped_traffic_lights, device, cfg
                        )
                        final_pred = traffic_light_classification.get_final_prediction(predictions)

                        traffic_light_preds.append(final_pred)

                        # Choose traffic light which occurs the most in {cfg.TRAFFIC_LIGHT_CLASSIFICATION.NUM_AVERAGE} as final overall prediction
                        if len(traffic_light_preds) >= cfg.TRAFFIC_LIGHT_CLASSIFICATION.NUM_AVERAGE:
                            final_pred_overall = traffic_light_classification.get_average_prediction(
                                traffic_light_preds
                            )
                            logger.info(f"Traffic light: {final_pred_overall}")

                            # Adjust forward_backward_velocity based on traffic light prediction
                            forward_backward_velocity = cfg.DRONE.SPEED
                            if final_pred_overall == "pedestrian-red":
                                traffic_light_info = "Red"
                                sound_alarm(voice_engine, "Red! Stop!")
                                forward_backward_velocity = 0
                            elif final_pred_overall == "pedestrian-green":
                                traffic_light_info = "Green"
                                sound_alarm(voice_engine, "Green")
                                # Speed up a litte bit to cross the street when pedestrian traffic light is green
                                forward_backward_velocity = cfg.DRONE.SPEED + cfg.DRONE.ACCELEARATION
                            else:
                                traffic_light_info = ""

                            velocities.update({"forward_backward_velocity": forward_backward_velocity})

                    if tello.is_flying:
                        # Adjust up-down velocity to maintain drone at target height
                        current_height = tello.get_height()
                        logger.info(f"Current height: {current_height}")
                        up_down_velocity = drone.get_maintain_flight_height_velocity(current_height, cfg)
                        velocities.update({"up_down_velocity": up_down_velocity})

                        vel_values = list(velocities.values())
                        vel_info = (
                            f"LR: {vel_values[0]}| FB: {vel_values[1]}| UD: {vel_values[2]}| Yaw: {vel_values[3]}"
                        )
                        logger.info(f"Vel: {vel_info}")

                        tello.send_rc_control(*velocities.values())

                        if drone.is_at_target_height(current_height, cfg):
                            logger.info("At target height")

                    # Display
                    if cfg.SEGMENTATION.BLEND:
                        blended = colorize.blend(img, colorized, cfg.SEGMENTATION.ALPHA)

                    img_display = display(_frame, colorized, blended, anno_frame, cfg.DRONE.DISPLAY_IMAGE)
                    img_display = np.asarray(img_display)

                    if cfg.DRONE.DISPLAY_INFO:
                        display_info(img_display, tello, velocities, current_height, traffic_light_info, cfg)

                    img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

            cv2.imshow("Tello", img_display)

            # Manual keyboard control
            key = cv2.waitKey(1) & 0xFF
            if key == ord("t"):  # Press "T" to take off
                tello.takeoff()
                logger.info("=> Take off")
            elif key == ord("l"):  # Press "L" to land
                tello.land()
                logger.info("=> Land")
            elif key == ord("q"):  # Press "Q" to stop normally
                break
            elif key == 27:  # Press Esc to stop emergencily
                tello.emergency()
                break

            num_frame += 1

        export_data(drone.centroid_measurements, drone.centroid_estimates)

        tello.land()
        tello.end()
        # video.release() # for webcam

        cv2.destroyAllWindows()
        logger.info("=> End")


if __name__ == "__main__":
    main()
