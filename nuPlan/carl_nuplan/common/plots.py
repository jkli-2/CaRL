from pathlib import Path
from typing import List, Union

import numpy as np
import numpy.typing as npt
from PIL import Image

from carl_nuplan.common.colors import (
    BLACK,
    DARK_GREY,
    DARKER_GREY,
    ELLIS_5,
    LIGHT_GREY,
    NEW_TAB_10,
)


def numpy_images_to_gif(
    numpy_images: List[npt.NDArray[np.uint8]], gif_path: Union[str, Path], duration: int = 50
) -> None:
    """
    Helper function to convert images into a GIF file.
    :param numpy_images: list of images as uint8 numpy arrays.
    :param gif_path: outout path for the GIF file.
    :param duration: duration between frames (TODO: check), defaults to 50
    """
    pil_images = [Image.fromarray(img) for img in numpy_images]
    pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], duration=duration, loop=0)


def image_to_rbg(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Helper function to convert an observation image to RGB format.
    :param image: _description_
    :return: _description_
    """
    _, width, height = image.shape
    rgb_image = np.zeros((width, height, 3), dtype=np.uint8)
    rgb_image.fill(255)
    # drivable area
    rgb_image[image[0] > 0] = LIGHT_GREY.rgb
    rgb_image[image[1] > 0] = DARK_GREY.rgb
    rgb_image[image[2] > 0] = BLACK.rgb
    rgb_image[image[5] > 0] = DARKER_GREY.rgb

    rgb_image[image[3] == 80] = NEW_TAB_10[4].rgb
    rgb_image[image[3] == 255] = NEW_TAB_10[2].rgb
    # rgb_image[image[4] > 0] = ELLIS_5[1].rgb
    rgb_image[image[6] > 0] = ELLIS_5[4].rgb
    rgb_image[image[7] > 0] = NEW_TAB_10[6].rgb
    rgb_image[image[8] > 0] = ELLIS_5[0].rgb

    rgb_image = np.rot90(rgb_image[::-1])
    return rgb_image
