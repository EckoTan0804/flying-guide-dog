from djitellopy import Tello
import time
import pygame
import cv2
import numpy as np


SPEED = 30
FPS = 120


class Frontend(object):
    """
    Maintains the Tello display and moves it using the keyboard keys.
    - Escape key: Quit
    - T: Takeoff 
    - L: Land
    - Arrow keys: Forward, backward, left, right
    - A and D: Counter clockwise and clockwise rotations (yaw)
    - W and S: Up and down
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Create pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities (between -100 and 100)
        self.forward_backward_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # Create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def run(self):
        self.tello.connect()
        self.tello.set_speed(self.speed)

        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()
        should_stop = False

        while not should_stop:

            frame = frame_read.frame
            surface = frame

            # pygame surface
            self.screen.fill([0, 0, 0])
            text = f"Battery: {self.tello.get_battery()}"
            cv2.putText(frame, text, (5, 720 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            surface = cv2.cvtColor(surface, cv2.COLOR_BGR2RGB)
            surface = np.rot90(surface)
            surface = np.flipud(surface)

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key, frame)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key, frame)

            if frame_read.stopped:
                break

            screen = pygame.surfarray.make_surface(surface)
            self.screen.blit(screen, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Deallocate resources
        self.tello.end()
        print("End")

    def keydown(self, key, frame):
        """
        Update velocities based on key pressed

        Parameters
        ----------
        key : pygame key
            pressed key
        """
        if key == pygame.K_UP:
            self.forward_backward_velocity = SPEED
        elif key == pygame.K_DOWN:
            self.forward_backward_velocity = -SPEED
        elif key == pygame.K_LEFT:
            self.left_right_velocity = -SPEED
        elif key == pygame.K_RIGHT:
            self.left_right_velocity = SPEED
        elif key == pygame.K_w:
            self.up_down_velocity = SPEED
        elif key == pygame.K_s:
            self.up_down_velocity = -SPEED
        elif key == pygame.K_a:
            self.yaw_velocity = -SPEED
        elif key == pygame.K_d:
            self.yaw_velocity = SPEED

    def keyup(self, key, frame):
        """
        Update velocities based on key released

        Parameters
        ----------
        key : pygame key
            Released key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:
            self.forward_backward_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:
            self.yaw_velocity = 0
        elif key == pygame.K_t:
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:
            not self.tello.land()
            self.send_rc_control = False
        elif key == pygame.K_p:
            # Take picture
            cv2.imwrite(f"pic_{time.time()}.jpg", frame)
            print("Take picture")

    def update(self):
        """
        Update routine. Send velocities to Tello
        """
        if self.send_rc_control:
            self.tello.send_rc_control(
                self.left_right_velocity, self.forward_backward_velocity, self.up_down_velocity, self.yaw_velocity
            )


def main():
    frontend = Frontend()
    frontend.run()


if __name__ == "__main__":
    main()
