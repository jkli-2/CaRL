from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.transform import translate_longitudinally
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.scenario_builder.abstract_scenario import (
    AbstractScenario,
    DetectionsTracks,
)
from nuplan.planning.simulation.history.simulation_history import (
    SimulationHistorySample,
)
from nuplan.planning.simulation.observation.observation_type import TrackedObjects
from shapely import affinity
from shapely.geometry import LineString, Polygon

from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    state_se2_to_array,
)
from carl_nuplan.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from carl_nuplan.planning.simulation.visualization.utils.config import (
    AGENT_CONFIG,
    BEV_PLOT_CONFIG,
    MAP_LAYER_CONFIG,
    TRAFFIC_LIGHT_CONFIG,
)


def add_configured_bev_on_ax(
    ax: plt.Axes,
    scenario: AbstractScenario,
    sample: SimulationHistorySample,
    route_roadblock_ids: Optional[List[str]] = None,
    ego_frame: bool = True,
) -> plt.Axes:

    if route_roadblock_ids is None:
        route_roadblock_ids = scenario.get_route_roadblock_ids()

    if "map" in BEV_PLOT_CONFIG["layers"]:
        add_map_to_bev_ax(
            ax,
            scenario.map_api,
            sample.ego_state,
            sample.traffic_light_status,
            route_roadblock_ids,
            ego_frame,
        )

    if "tracks" in BEV_PLOT_CONFIG["layers"]:
        assert isinstance(sample.observation, DetectionsTracks)
        add_tracks_to_bev_ax(ax, sample.observation.tracked_objects, sample.ego_state, ego_frame)

    return ax


def add_tracks_to_bev_ax(
    ax: plt.Axes,
    tracked_objects: TrackedObjects,
    ego_state: EgoState,
    ego_frame: bool = True,
) -> plt.Axes:

    height = max([abs(elem) for elem in BEV_PLOT_CONFIG["figure_xlim"]])
    width = max([abs(elem) for elem in BEV_PLOT_CONFIG["figure_ylim"]])
    radius = np.sqrt(width**2 + height**2)

    for tracked_object in tracked_objects:
        tracked_object: TrackedObject
        if ego_state.rear_axle.distance_to(tracked_object.box.center) <= radius:
            oriented_box = (
                _oriented_box_local_coords(tracked_object.box, ego_state.rear_axle) if ego_frame else tracked_object.box
            )
            add_oriented_box_to_bev_ax(ax, oriented_box, AGENT_CONFIG[tracked_object.tracked_object_type])

    ego_oriented_box = (
        _oriented_box_local_coords(ego_state.car_footprint.oriented_box, ego_state.rear_axle)
        if ego_frame
        else ego_state.car_footprint.oriented_box
    )
    add_oriented_box_to_bev_ax(
        ax,
        ego_oriented_box,
        AGENT_CONFIG[TrackedObjectType.EGO],
    )
    return ax


def add_map_to_bev_ax(
    ax: plt.Axes,
    map_api: AbstractMap,
    ego_state: EgoState,
    traffic_light_status_data: Optional[List[TrafficLightStatusData]] = None,
    route_roadblock_ids: Optional[List[str]] = None,
    ego_frame: bool = True,
) -> plt.Axes:

    polygon_layers: List[SemanticMapLayer] = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.CROSSWALK,
    ]
    lane_layers: List[SemanticMapLayer] = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
    ]

    traffic_light_dict: Dict[str, TrafficLightStatusType] = {}
    if traffic_light_status_data is not None:
        for data in traffic_light_status_data:
            traffic_light_dict[str(data.lane_connector_id)] = TrafficLightStatusType.RED

    height = max([abs(elem) for elem in BEV_PLOT_CONFIG["figure_xlim"]])
    width = max([abs(elem) for elem in BEV_PLOT_CONFIG["figure_ylim"]])
    radius = np.sqrt(width**2 + height**2)

    # query map api with interesting layers
    map_object_dict = map_api.get_proximal_map_objects(
        point=ego_state.rear_axle.point,
        radius=radius,
        layers=list(set(polygon_layers + lane_layers)),
    )

    for polygon_layer in polygon_layers:
        for map_object in map_object_dict[polygon_layer]:
            polygon: Polygon = (
                _geometry_local_coords(map_object.polygon, ego_state.rear_axle) if ego_frame else map_object.polygon
            )
            add_polygon_to_bev_ax(ax, polygon, MAP_LAYER_CONFIG[polygon_layer])

    for lane_layer in lane_layers:
        for map_object in map_object_dict[lane_layer]:
            if map_object.get_roadblock_id() in route_roadblock_ids:

                polygon: Polygon = (
                    _geometry_local_coords(map_object.polygon, ego_state.rear_axle) if ego_frame else map_object.polygon
                )
                add_polygon_to_bev_ax(ax, polygon, MAP_LAYER_CONFIG[SemanticMapLayer.ROADBLOCK])

    for lane_layer in lane_layers:
        for map_object in map_object_dict[lane_layer]:
            # 2. Baseline
            linestring: LineString = (
                _geometry_local_coords(map_object.baseline_path.linestring, ego_state.rear_axle)
                if ego_frame
                else map_object.baseline_path.linestring
            )
            if map_object.id in traffic_light_dict.keys():
                polyline_config = TRAFFIC_LIGHT_CONFIG[traffic_light_dict[map_object.id]]
            else:
                polyline_config = MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS]

            add_linestring_to_bev_ax(ax, linestring, polyline_config)

    return ax


# FIXME:
# def add_trajectory_to_bev_ax(ax: plt.Axes, trajectory: Trajectory, config: Dict[str, Any]) -> plt.Axes:
#     """
#     Add trajectory poses as lint to plot
#     :param ax: matplotlib ax object
#     :param trajectory: navsim trajectory dataclass
#     :param config: dictionary with plot parameters
#     :return: ax with plot
#     """
#     poses = np.concatenate([np.array([[0, 0]]), trajectory.poses[:, :2]])
#     ax.plot(
#         poses[:, 1],
#         poses[:, 0],
#         color=config["line_color"],
#         alpha=config["line_color_alpha"],
#         linewidth=config["line_width"],
#         linestyle=config["line_style"],
#         marker=config["marker"],
#         markersize=config["marker_size"],
#         markeredgecolor=config["marker_edge_color"],
#         zorder=config["zorder"],
#     )
#     return ax


def add_oriented_box_to_bev_ax(ax: plt.Axes, box: OrientedBox, config: Dict[str, Any]) -> plt.Axes:
    """
    Adds birds-eye-view visualization of surrounding bounding boxes
    :param ax: matplotlib ax object
    :param box: nuPlan dataclass for 2D bounding boxes
    :param config: dictionary with plot parameters
    :return: ax with plot
    """

    box_corners = box.all_corners()
    corners = [[corner.x, corner.y] for corner in box_corners]
    corners = np.asarray(corners + [corners[0]])

    ax.fill(
        corners[:, 1],
        corners[:, 0],
        color=config["fill_color"].hex,
        alpha=config["fill_color_alpha"],
        zorder=config["zorder"],
    )
    ax.plot(
        corners[:, 1],
        corners[:, 0],
        color=config["line_color"].hex,
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )

    if config["heading_style"] is not None:
        if config["heading_style"] == "line":
            future = translate_longitudinally(box.center, distance=box.length / 2 + 1)
            line = np.array([[box.center.x, box.center.y], [future.x, future.y]])
            ax.plot(
                line[:, 1],
                line[:, 0],
                color=config["line_color"].hex,
                alpha=config["line_color_alpha"],
                linewidth=config["line_width"],
                linestyle=config["line_style"],
                zorder=config["zorder"],
            )
        elif config["heading_style"] == "arrow":
            arrow_length = 1
            dx = arrow_length * np.cos(box.center.heading)
            dy = arrow_length * np.sin(box.center.heading)

            # Add the arrowhead
            arrow = patches.FancyArrow(
                box.center.y,
                box.center.x,
                dy,
                dx,
                width=config["line_width"],
                head_width=1,
                head_length=1,
                color=config["line_color"].hex,
                zorder=config["zorder"],
                length_includes_head=True,
            )
            ax.add_patch(arrow)

    return ax


def add_polygon_to_bev_ax(ax: plt.Axes, polygon: Polygon, config: Dict[str, Any]) -> plt.Axes:
    """
    Adds shapely polygon to birds-eye-view visualization
    :param ax: matplotlib ax object
    :param polygon: shapely Polygon
    :param config: dictionary containing plot parameters
    :return: ax with plot
    """

    def _add_element_helper(element: Polygon):
        """Helper to add single polygon to ax"""
        exterior_x, exterior_y = element.exterior.xy
        ax.fill(
            exterior_y,
            exterior_x,
            color=config["fill_color"].hex,
            alpha=config["fill_color_alpha"],
            zorder=config["zorder"],
        )
        ax.plot(
            exterior_y,
            exterior_x,
            color=config["line_color"].hex,
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            zorder=config["zorder"],
        )
        for interior in element.interiors:
            x_interior, y_interior = interior.xy
            ax.fill(
                y_interior,
                x_interior,
                color=BEV_PLOT_CONFIG["background_color"].hex,
                zorder=config["zorder"],
            )
            ax.plot(
                y_interior,
                x_interior,
                color=config["line_color"].hex,
                alpha=config["line_color_alpha"],
                linewidth=config["line_width"],
                linestyle=config["line_style"],
                zorder=config["zorder"],
            )

    if isinstance(polygon, Polygon):
        _add_element_helper(polygon)
    else:
        # NOTE: in rare cases, a map polygon has several sub-polygons.
        for element in polygon:
            _add_element_helper(element)

    return ax


def add_linestring_to_bev_ax(ax: plt.Axes, linestring: LineString, config: Dict[str, Any]) -> plt.Axes:
    """
    Adds shapely linestring (polyline) to birds-eye-view visualization
    :param ax: matplotlib ax object
    :param linestring: shapely LineString
    :param config: dictionary containing plot parameters
    :return: ax with plot
    """

    x, y = linestring.xy
    ax.plot(
        y,
        x,
        color=config["line_color"].hex,
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )
    return ax


def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
    """Helper for transforming shapely geometry in coord-frame"""
    a = np.cos(origin.heading)
    b = np.sin(origin.heading)
    d = -np.sin(origin.heading)
    e = np.cos(origin.heading)
    xoff = -origin.x
    yoff = -origin.y
    translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
    rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])
    return rotated_geometry


def _oriented_box_local_coords(box: OrientedBox, origin: StateSE2) -> Any:
    """Helper for transforming oriented box in coord-frame"""
    center = StateSE2(*convert_absolute_to_relative_se2_array(origin, state_se2_to_array(box.center))[0])
    return OrientedBox(
        center=center,
        width=box.width,
        length=box.length,
        height=box.height,
    )


def configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the plt ax object for birds-eye-view plots
    :param ax: matplotlib ax object
    :return: configured ax object
    """
    ax.set_aspect("equal")

    # NOTE: x forward, y sideways
    ax.set_xlim(BEV_PLOT_CONFIG["figure_xlim"])
    ax.set_ylim(BEV_PLOT_CONFIG["figure_ylim"])

    # NOTE: left is y positive, right is y negative
    ax.invert_xaxis()

    return ax


def configure_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the ax object for general plotting
    :param ax: matplotlib ax object
    :return: ax object without a,y ticks
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
