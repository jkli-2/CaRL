from collections import defaultdict
from typing import Any, Dict, List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from shapely.geometry import Polygon

from carl_nuplan.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)

DRIVABLE_MAP_LAYERS = [
    SemanticMapLayer.ROADBLOCK,
    SemanticMapLayer.ROADBLOCK_CONNECTOR,
    SemanticMapLayer.CARPARK_AREA,
]


def get_drivable_area_map(
    map_api: AbstractMap,
    ego_state: EgoState,
    map_radius: float = 50,
) -> PDMOccupancyMap:

    # query all drivable map elements around ego position
    position: Point2D = ego_state.center.point
    drivable_area = map_api.get_proximal_map_objects(position, map_radius, DRIVABLE_MAP_LAYERS)

    # collect lane polygons in list, save on-route indices
    drivable_polygons: List[Polygon] = []
    drivable_polygon_ids: List[str] = []

    for type in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for roadblock in drivable_area[type]:
            for lane in roadblock.interior_edges:
                drivable_polygons.append(lane.polygon)
                drivable_polygon_ids.append(lane.id)

    for carpark in drivable_area[SemanticMapLayer.CARPARK_AREA]:
        drivable_polygons.append(carpark.polygon)
        drivable_polygon_ids.append(carpark.id)

    # create occupancy map with lane polygons
    drivable_area_map = PDMOccupancyMap(drivable_polygon_ids, drivable_polygons)

    return drivable_area_map


# TODO: move function for general use
def get_proximal_map_objects(
    map_patch: Polygon,
    map_api: AbstractMap,
    layers: List[SemanticMapLayer],
) -> Dict[str, Any]:
    assert isinstance(map_api, NuPlanMap)

    supported_layers = map_api.get_available_map_objects()
    unsupported_layers = [layer for layer in layers if layer not in supported_layers]

    assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"

    object_map: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)
    for layer in layers:
        object_map[layer] = map_api._get_proximity_map_object(map_patch, layer)

    return object_map
