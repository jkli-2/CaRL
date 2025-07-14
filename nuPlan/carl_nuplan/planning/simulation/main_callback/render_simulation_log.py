import io
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.gridspec import GridSpec

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.history.simulation_history import SimulationHistorySample

from carl_nuplan.planning.simulation.trajectory.action_trajectory import ActionTrajectory
from carl_nuplan.planning.simulation.visualization.utils.bev import (
    add_configured_bev_on_ax,
    configure_ax,
    configure_bev_ax,
)
from carl_nuplan.planning.simulation.visualization.utils.config import BEV_PLOT_CONFIG


def render_simulation_log(simulation_log: SimulationLog, visualization_folder: Path) -> None:
    """
    Visualizes a single simulation log by rendering the BEV and saving it as a video.
    TODO: Refactor this function.
    :param simulation_log: SimulationLog object of nuPlan to visualize.
    :param visualization_folder: Folder to save the visualization results.
    """

    visualization_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    images = []
    for sample in simulation_log.simulation_history.data:

        # 1. plot data in matplotlib
        plot_bev_sample(ax, simulation_log.scenario, sample)
        fig.tight_layout()

        # 2. retrieve the image from the plot
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150)
        buffer.seek(0)
        image = np.array(Image.open(buffer).copy())
        buffer.close()
        ax.cla()
        images.append(image)

    height, width, _ = images[0].shape
    video_path = visualization_folder / f"{simulation_log.scenario.token}.avi"
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    video = cv2.VideoWriter(str(video_path), fourcc, 20, (width, height))
    for image in images:
        # Convert numpy array to BGR format for OpenCV
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()
    plt.close(fig)
    plt.close("all")


def render_simulation_log_v2(simulation_log: SimulationLog, visualization_folder: Path) -> None:
    """
    Visualizes a single simulation log by rendering the BEV and plotting the action distributions.
    TODO: Refactor this function.
    :param simulation_log: SimulationLog object of nuPlan to visualize.
    :param visualization_folder: Folder to save the visualization results.
    """

    visualization_folder.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    images = []
    acceleration = []
    steer = []
    route_roadblock_ids = simulation_log.planner._observation_builder._route_roadblock_ids
    for iteration, sample in enumerate(simulation_log.simulation_history.data):

        trajectory = sample.trajectory
        assert isinstance(trajectory, ActionTrajectory), f"Expected Trajectory, got {type(trajectory)}"

        distribution = trajectory._store

        device = distribution.concentration1.device
        granularity = torch.arange(start=0.0, end=1.0, step=0.001).unsqueeze(1)
        granularity = torch.ones((granularity.shape[0], 2)) * granularity
        granularity = granularity.to(device)
        granularity_numpy = deepcopy(granularity).cpu().numpy()

        distribution = distribution.log_prob(granularity)
        distribution = torch.exp(distribution).cpu().numpy()

        # 1. plot data in matplotlib
        plot_bev_sample(ax1, simulation_log.scenario, sample, route_roadblock_ids)

        acceleration.append(trajectory._raw_action[0])
        steer.append(trajectory._raw_action[1])

        ax2.set_title("Acceleration")
        ax2.plot(granularity_numpy[..., 0] * 2 - 1, distribution[..., 0] / 25)
        ax2.axvline(0, ymax=1, color="black")

        ax3.set_title("Steering")
        ax3.plot(granularity_numpy[..., 1] * 2 - 1, distribution[..., 1] / 25)
        ax3.axvline(0, ymax=1, color="black")

        for ax_ in [ax2, ax3]:
            ax_.set_ylim(0, 1)
            ax_.set_xlim(-1, 1)

        ax3.invert_xaxis()
        # fig.tight_layout()

        # 2. retrieve the image from the plot
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image = np.array(Image.open(buffer).copy())
        buffer.close()

        ax1.cla()
        ax2.cla()
        ax3.cla()

        images.append(image)

    height, width, _ = images[0].shape
    video_path = visualization_folder / f"{simulation_log.scenario.token}.avi"
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    video = cv2.VideoWriter(str(video_path), fourcc, 10, (width, height))
    for image in images:
        # Convert numpy array to BGR format for OpenCV
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()
    plt.close(fig)
    plt.close("all")


def plot_bev_sample(
    ax: plt.Axes,
    scenario: AbstractScenario,
    sample: SimulationHistorySample,
    route_roadblock_ids: Optional[List[str]] = None,
) -> None:
    if route_roadblock_ids is None:
        route_roadblock_ids = scenario.get_route_roadblock_ids()

    add_configured_bev_on_ax(ax, scenario, sample, route_roadblock_ids)
    configure_bev_ax(ax)
    configure_ax(ax)
