import argparse
import pygame
import numpy as np
from carla import Carla
import cv2
import glob
import os
import time
import subprocess

import embodied


def frame_to_rgb(frames):
    if frames.shape[-1] > 3:
        factor = frames / 255.0

        color_mappings = np.array(
            [
                [50, 50, 50],  # Road
                [150, 150, 150],  # Route
                [255, 255, 255],  # Ego
                [100, 100, 100],  # Lane
                [0, 255, 255],  # Yellow lines
                [255, 0, 255],  # White lines
                # Dynamic objects
                [230, 0, 0],  # Vehicle at t=0
                [230, 230, 0],  # Walker at t=0
                [230, 0, 0],  # Emergency car at t=0
                [0, 0, 230],  # Obstacle at t=0
                [0, 230, 0],  # Green traffic light at t=0
                [0, 0, 230],  # Yellow & Red traffic light at t=0
                [0, 170, 170],  # Stop sign at t=0
                [230, 50, 50],  # Vehicle at t=-5
                [230, 230, 50],  # Walker at t=-5
                [230, 50, 50],  # Emergency car at t=-5
                [50, 50, 230],  # Obstacle at t=-5
                [50, 230, 50],  # Green traffic light at t=-5
                [50, 230, 230],  # Yellow & Red traffic light at t=-5
                [50, 170, 170],  # Stop sign at t=-5
                [230, 100, 100],  # Vehicle at t=-10
                [230, 230, 100],  # Walker at t=-10
                [230, 100, 100],  # Emergency car at t=-10
                [100, 100, 230],  # Obstacle at t=-10
                [100, 230, 100],  # Green traffic light at t=-10
                [100, 230, 230],  # Yellow & Red traffic light at t=-10
                [100, 170, 170],  # Stop sign at t=-10
                [230, 150, 150],  # Vehicle at t=-15
                [230, 230, 150],  # Walker at t=-15
                [230, 150, 150],  # Emergency car at t=-15
                [150, 150, 230],  # Obstacle at t=-15
                [100, 230, 100],  # Green traffic light at t=-15
                [150, 230, 230],  # Yellow & Red traffic light at t=-15
                [150, 170, 170],  # Stop sign at t=-15
            ]
        )

        video_frames = factor[:, :, 0, None] * color_mappings[0]
        # Process remaining channels
        for i in [
            0,
            3,
            1,
            4,
            5,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            2,
        ]:
            channel_factor = factor[:, :, i, None]
            channel_color = frames[:, :, i, None] / 255.0 * color_mappings[i]
            video_frames = (1 - channel_factor) * video_frames + channel_factor * channel_color

        video_frames = np.round(video_frames).clip(0, 255).astype(np.uint8)
    else:
        video_frames = frames.copy()

        if video_frames.shape[-1] == 1:
            video_frames = video_frames.reshape(video_frames.shape[:-1])

    return video_frames


def main(args):
    # stop CARLA simulators from previous executions and delete previous inner files
    _ = [os.remove(os.path.join("/tmp", file)) for file in os.listdir("/tmp") if file.startswith("inner")]
    _ = [subprocess.call("pkill Carla", shell=True) for _ in range(3)]
    time.sleep(0.1)

    # Initialize Pygame for keyboard input and display
    pygame.init()
    screen = pygame.display.set_mode((1600, 1600))
    pygame.display.set_caption("CARLA Keyboard Control")

    # Initialize font for text rendering
    font = pygame.font.Font(None, 72)

    # evaluation
    # Initialize the CARLA environment
    env = Carla(
        task="manual_control",
        carla_installation_path=args.carla_path,
        image_size=(128, 128),
        index=0,
        eval=True,
        results_directory="results",
        eval_routes=glob.glob(args.route_path + "/*.xml"),
        num_envs=1,
        seed=42,
        eval_routes_counter=embodied.MPCounter(),
        render_off_screen=False,
    )

    # training
    # # Initialize the CARLA environment
    # env = Carla(
    #     task="manual_control",
    #     carla_installation_path=args.carla_path,
    #     image_size=(128, 128),
    #     index=0,
    #     eval=False,
    #     num_envs=1,
    #     seed=42,
    #     render_off_screen=False,
    # )

    action = {"action": 25, "reset": False}  # Default action
    running = True

    pressed_keys = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    action["reset"] = True
                pressed_keys.append(event.key)
            elif event.type == pygame.KEYUP:
                if event.key in pressed_keys:
                    pressed_keys.remove(event.key)

        if pygame.K_UP in pressed_keys and pygame.K_LEFT in pressed_keys:
            action["action"] = 1
        elif pygame.K_UP in pressed_keys and pygame.K_RIGHT in pressed_keys:
            action["action"] = 9
        elif pygame.K_UP in pressed_keys:
            action["action"] = 5
        elif pygame.K_DOWN in pressed_keys:
            action["action"] = 0
        elif pygame.K_LEFT in pressed_keys:
            action["action"] = 21
        elif pygame.K_RIGHT in pressed_keys:
            action["action"] = 29
        else:
            action["action"] = 25

        # Step the environment
        obs = env.step(action)
        action["reset"] = False  # Reset only lasts for one step

        # Display the image
        image = frame_to_rgb(obs["image"])
        image = cv2.resize(image, (1600, 1600))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pygame_image = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        screen.blit(pygame_image, (0, 0))

        # Display info on the Pygame window
        screen.blit(font.render(f"Reward: {obs['reward']:.2f}", True, (255, 255, 255)), (10, 10))
        screen.blit(font.render(f"Is Last: {obs['is_last']}", True, (255, 255, 255)), (10, 70))
        screen.blit(font.render(f"Is Terminal: {obs['is_terminal']}", True, (255, 255, 255)), (10, 130))

        pygame.display.flip()

        if pygame.key.get_pressed()[pygame.K_q]:
            break

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA Keyboard Control")
    parser.add_argument("--carla_path", type=str, default="../../../CARLA_0.9.15/", help="Path to CARLA installation")
    parser.add_argument(
        "--route_path",
        type=str,
        default="../../custom_leaderboard/leaderboard/data/bug_test",
        help="Path to the route file",
    )
    args = parser.parse_args()

    main(args)
