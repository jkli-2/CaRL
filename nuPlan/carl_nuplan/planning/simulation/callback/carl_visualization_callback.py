import io
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import (
    SimulationHistory,
    SimulationHistorySample,
)
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from PIL import Image

from carl_nuplan.planning.simulation.visualization.utils.bev import (
    add_configured_bev_on_ax,
    configure_ax,
    configure_bev_ax,
)
from carl_nuplan.planning.simulation.visualization.utils.config import BEV_PLOT_CONFIG


class CarlVisualizationCallback(AbstractCallback):
    def __init__(
        self,
        output_directory: Union[str, Path],
        visualization_dir: Union[str, Path] = "visualization",
    ):
        self._output_directory = Path(output_directory) / visualization_dir
        self._ego_view: bool = True

        # lazy loaded
        self._images: Optional[List[np.ndarray]] = None

        self._fig = None
        self._ax = None

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        In initialization start just render scenario
        """

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""

    def on_step_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        sample: SimulationHistorySample,
    ) -> None:
        """
        Render sample after a step
        """
        assert self._images is not None, "Images should be initialized in on_simulation_start"

        add_configured_bev_on_ax(self._ax, setup.scenario, sample)
        configure_bev_ax(self._ax)
        configure_ax(self._ax)
        self._fig.tight_layout()

        # Creating PIL image from fig
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        pil_image = Image.open(buf).copy()

        # Convert PIL image to numpy array and append to images
        self._images.append(np.array(pil_image))
        print(np.array(pil_image).shape)

        # close buffer and figure
        buf.close()
        self._ax.cla()

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""

    def on_planner_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        trajectory: AbstractTrajectory,
    ) -> None:
        """Inherited, see superclass."""

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""
        self._fig, self._ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
        self._images = []

    def on_simulation_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        history: SimulationHistory,
    ) -> None:
        """
        On reached_end just call step_end
        """
        assert self._images is not None, "Images should be initialized in on_simulation_start"

        # Save images as a video to disk
        self._output_directory.mkdir(parents=True, exist_ok=True)
        video_path = self._output_directory / f"{setup.scenario.token}.avi"

        # Assuming all images are the same size, get dimensions from the first image
        height, width, _ = self._images[0].shape
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        video = cv2.VideoWriter(str(video_path), fourcc, 10, (width, height))

        for image in self._images:
            # Convert numpy array to BGR format for OpenCV
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(frame)

        video.release()
        plt.close(self._fig)
