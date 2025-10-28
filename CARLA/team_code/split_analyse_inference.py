import json
import xml.etree.ElementTree as ET
import pandas as pd

# CaRL
# json_path = "/home/junkali/Data/CaRL/CARLA/results/CaRL_10M_01/carl_inference_result_1.json"
# split_json_a_path = "/home/junkali/Data/CaRL/CARLA/results/CaRL_10M_01/carl_inference_result_1_baseline.json"
# split_json_b_path = "/home/junkali/Data/CaRL/CARLA/results/CaRL_10M_01/carl_inference_result_1_extension.json"
# summary_a_path = "/home/junkali/Data/CaRL/CARLA/results/CaRL_10M_01/carl_inference_result_1_baseline_summary.csv"
# summary_b_path = "/home/junkali/Data/CaRL/CARLA/results/CaRL_10M_01/carl_inference_result_1_extension_summary.csv"

# CaRL 300M
json_path = "/home/junkali/Data/CaRL/CARLA/results/CaRL_v1_1_PY_01/carl_inference_result_1.json"
split_json_a_path = "/home/junkali/Data/CaRL/CARLA/results/CaRL_v1_1_PY_01/carl_inference_result_1_baseline.json"
split_json_b_path = "/home/junkali/Data/CaRL/CARLA/results/CaRL_v1_1_PY_01/carl_inference_result_1_extension.json"
summary_a_path = "/home/junkali/Data/CaRL/CARLA/results/CaRL_v1_1_PY_01/carl_inference_result_1_baseline_summary.csv"
summary_b_path = "/home/junkali/Data/CaRL/CARLA/results/CaRL_v1_1_PY_01/carl_inference_result_1_extension_summary.csv"

# Roach
# json_path = "/home/junkali/Data/CaRL/CARLA/results/Roach_04/roach_inference_result_1.json"
# split_json_a_path = "/home/junkali/Data/CaRL/CARLA/results/Roach_04/roach_inference_result_1_baseline.json"
# split_json_b_path = "/home/junkali/Data/CaRL/CARLA/results/Roach_04/roach_inference_result_1_extension.json"
# summary_a_path = "/home/junkali/Data/CaRL/CARLA/results/Roach_04/roach_inference_result_1_baseline_summary.csv"
# summary_b_path = "/home/junkali/Data/CaRL/CARLA/results/Roach_04/roach_inference_result_1_extension_summary.csv"

xml_path = "/home/junkali/Data/CaRL/CARLA/custom_leaderboard/leaderboard/data/inference_route20.xml"

baseline = {
    'ControlLoss',
    'HardBreakRoute',
    'DynamicObjectCrossing',
    'VehicleTurningRoute',
    'SignalizedJunctionLeftTurn',
    'OppositeVehicleRunningRedLight',
    'SignalizedJunctionRightTurn'
}

extension = {
    'Accident',
    'AccidentTwoWays',
    'BlockedIntersection',
    'ConstructionObstacle',
    'ConstructionObstacleTwoWays'
}

tree = ET.parse(xml_path)
root = tree.getroot()
route_map = {}

for route in root.findall('route'):
    route_id = route.attrib['id']
    scenario_elem = route.find('scenarios/scenario')
    if scenario_elem is not None:
        route_map[route_id] = scenario_elem.attrib.get('type')

with open(json_path, 'r') as f:
    data = json.load(f)

records = data["_checkpoint"]["records"]

baseline_records, extension_records = [], []

for rec in records:
    rid = rec["route_id"].split('_')[1]  # format: RouteScenario_24211_rep0 -> 24211
    scenario_type = route_map.get(rid, None)
    rec["scenario_type"] = scenario_type

    if scenario_type in baseline:
        baseline_records.append(rec)
    elif scenario_type in extension:
        extension_records.append(rec)
    else:
        print(f"Unknown scenario for route {rid}: {scenario_type}")

def write_subset(records, fname):
    subset = data.copy()
    subset["_checkpoint"]["records"] = records
    with open(fname, 'w') as f:
        json.dump(subset, f, indent=4)
    print(f"Saved {fname} ({len(records)} records)")

write_subset(baseline_records, split_json_a_path)
write_subset(extension_records, split_json_b_path)

def records_to_df(records):
    df = pd.DataFrame([{
        "route_id": r["route_id"],
        "scenario_type": r.get("scenario_type"),
        "status": r.get("status"),
        "score_composed": r["scores"]["score_composed"],
        "score_penalty": r["scores"]["score_penalty"],
        "score_route": r["scores"]["score_route"],
        "duration_game": r["meta"]["duration_game"],
        "duration_system": r["meta"]["duration_system"],
        "num_infractions": r["num_infractions"]
    } for r in records])
    return df

df_base = records_to_df(baseline_records)
df_ext = records_to_df(extension_records)

df_base.to_csv(summary_a_path, index=False)
df_ext.to_csv(summary_b_path, index=False)

def summarize(df, name):
    print(f"\n--- {name.upper()} ---")
    print(f"Count: {len(df)}")
    print(f"Mean composed score: {df['score_composed'].mean():.2f}")
    print(f"Std dev composed score: {df['score_composed'].std():.2f}")
    completed = df['status'].isin(['Completed', 'Perfect']).sum()
    print(f"Completion rate: {completed / len(df) * 100:.1f}%")

summarize(df_base, "Baseline")
summarize(df_ext, "Extension")
