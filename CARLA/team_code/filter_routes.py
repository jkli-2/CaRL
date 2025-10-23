import xml.etree.ElementTree as ET
from copy import deepcopy

INPUT_XML = "/data/junkali/CaRL/CARLA/custom_leaderboard/leaderboard/data/routes_validation.xml"
OUTPUT_XML = "/data/junkali/CaRL/CARLA/custom_leaderboard/leaderboard/data/inference_validation.xml"
USED_SCENARIOS = {
    'ControlLoss',
    'HardBreakRoute',
    'DynamicObjectCrossing',
    'VehicleTurningRoute',
    'SignalizedJunctionLeftTurn',
    'OppositeVehicleRunningRedLight',
    'SignalizedJunctionRightTurn',
    'Accident',
    'AccidentTwoWays',
    'BlockedIntersection',
    'ConstructionObstacle',
    'ConstructionObstacleTwoWays'
}
INCLUDED_TOWNS = {
    'Town01',
    'Town02',
    'Town03',
    'Town04',
    'Town05',
    'Town06',
    'Town07',
    'Town10HD'
}

def main():
    tree = ET.parse(INPUT_XML)
    root = tree.getroot()

    if root.tag != "routes":
        raise ValueError(f"Expected root <routes>, got <{root.tag}>")

    filtered_routes = ET.Element("routes")
    kept_routes = 0
    removed_scenarios_total = 0

    for route in root.findall("route"):
        town = route.get("town", "")
        if town not in INCLUDED_TOWNS:
            continue
        route_copy = deepcopy(route)
        scenarios_elem = route_copy.find("scenarios")
        if scenarios_elem is not None:
            to_remove = []
            for scen in scenarios_elem.findall("scenario"):
                scen_type = scen.get("type", "")
                if scen_type not in USED_SCENARIOS:
                    to_remove.append(scen_type)
            for scen in to_remove:
                scenarios_elem.remove(scen)
            removed_scenarios_total += len(to_remove)
            if len(scenarios_elem.findall("scenario")) == 0:
                continue
        else:
            continue
        filtered_routes.append(route_copy)
        kept_routes += 1

    new_tree = ET.ElementTree(filtered_routes)
    ET.indent(new_tree, space="   ", level=0)
    new_tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)
    print(
        f"Kept {kept_routes} routes (town filter)."
        f"Removed {removed_scenarios_total} scenarios not in allow-list."
        f"Saved to {OUTPUT_XML}"
    )

if __name__ == "__main__":
    main()
