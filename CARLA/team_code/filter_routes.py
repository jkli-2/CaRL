import xml.etree.ElementTree as ET

INPUT_XML = "/data/junkali/CaRL/CARLA/custom_leaderboard/leaderboard/data/bench2drive220.xml"
OUTPUT_XML = "/data/junkali/CaRL/CARLA/custom_leaderboard/leaderboard/data/inference_route.xml"
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

def main():
    tree = ET.parse(INPUT_XML)
    root = tree.getroot()

    if root.tag != "routes":
        raise ValueError(f"Expected root <routes>, got <{root.tag}>")

    filtered_routes = ET.Element("routes")

    for route in root.findall("route"):
        scenario_elem = route.find("scenarios/scenario")
        if scenario_elem is not None:
            scenario_type = scenario_elem.get("type", "")
            if scenario_type in USED_SCENARIOS:
                filtered_routes.append(route)

    new_tree = ET.ElementTree(filtered_routes)
    ET.indent(new_tree, space="   ", level=0)
    new_tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)
    print(f"Filtered {len(filtered_routes)} routes saved to {OUTPUT_XML}")

if __name__ == "__main__":
    main()
