import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union, split
from parse_xodr_file import parse_xodr_file
from utils import plot_shapely_road_network, plot_opencv_road_network
import _pickle as pickle
import carla
import pathlib
from precompute_stop_sign_data import compute_stop_sign_data
from precompute_traffic_light_data import compute_traffic_light_data

def calculate_bounding_box(road_polygons, white_lines, yellow_lines):
    all_bounds = np.array([x.bounds for x in [road_polygons, white_lines, yellow_lines] if not x.is_empty])
    
    min_x, min_y = all_bounds[:, [0, 1]].min(axis=0)
    max_x, max_y = all_bounds[:, [2, 3]].max(axis=0)
    
    return np.array([min_x, min_y, max_x, max_y])

def convert_lines_to_opencv(multi_line):
    if multi_line.is_empty:
        return np.array([], dtype=np.float32).reshape((0, 2))
    
    if isinstance(multi_line, MultiLineString):
        return [np.array(line.xy, dtype=np.float32).T for line in multi_line.geoms]
    elif isinstance(multi_line, LineString):
        return [np.array(multi_line.xy, dtype=np.float32).T]

def split_complex_polygons(polygons, max_iterations=10, simplify_tolerance=0.01):
    for _ in range(max_iterations):
        simplified_polygons = []
        needs_further_splitting = False

        for poly in polygons:
            buffered_poly = poly.simplify(simplify_tolerance)
            if not buffered_poly.interiors:
                simplified_polygons.append(buffered_poly)
                continue

            minx, miny, maxx, maxy = buffered_poly.bounds
            split_x = np.mean(np.array(buffered_poly.interiors[0].xy)[0])
            split_line = LineString([(split_x, miny - 1), (split_x, maxy + 1)])
            split_result = split(buffered_poly, split_line)

            split_polygons = [geom.simplify(simplify_tolerance) for geom in split_result.geoms]
            simplified_polygons.extend(split_polygons)
            needs_further_splitting = needs_further_splitting or any(len(p.interiors) != 0 for p in split_polygons)

        polygons = simplified_polygons
        if not needs_further_splitting:
            break

    if needs_further_splitting:
        assert f"Increase max_iterations since some polygons could not be split successfully!"
    
    return polygons

def convert_polygons_to_opencv(multi_polygon):
    if multi_polygon.is_empty:
        return np.array([], dtype=np.float32).reshape((0, 2))
        
    if isinstance(multi_polygon, MultiPolygon):
        polygons = list(multi_polygon.geoms)
    elif isinstance(multi_polygon, Polygon):
        polygons = [multi_polygon]
    else:
        raise ValueError(f"Unsupported geometry type: {type(multi_polygon)}")
    
    return [np.array(poly.exterior.xy, dtype=np.float32).T for poly in split_complex_polygons(polygons)]

def convert_intersection_map_to_opencv(intersection_map):
    opencv_intersection_map = {}
    for key, (road, white_line, yellow_line) in intersection_map.items():
        road_opencv = convert_polygons_to_opencv(road)
        white_line_opencv = convert_lines_to_opencv(white_line)
        yellow_line_opencv = convert_lines_to_opencv(yellow_line)
        
        opencv_intersection_map[key] = (road_opencv, white_line_opencv, yellow_line_opencv)
    return opencv_intersection_map


def align_to_grid(bounds, grid_size):
    min_x, min_y = (np.floor(bounds[:2] / grid_size) * grid_size).astype(int)
    max_x, max_y = (np.ceil(bounds[2:] / grid_size) * grid_size).astype(int)

    return min_x, min_y, max_x, max_y

def create_intersection_map(road_polygons, white_lines, yellow_lines, grid_size):
    bounding_box = calculate_bounding_box(road_polygons, white_lines, yellow_lines)
    grid_min_x, grid_min_y, grid_max_x, grid_max_y = align_to_grid(bounding_box, grid_size)
    
    intersection_map = {}
    for x in range(grid_min_x, grid_max_x, grid_size):
        for y in range(grid_min_y, grid_max_y, grid_size):
            grid_cell = Polygon([
                (x, y),
                (x + grid_size, y),
                (x + grid_size, y + grid_size),
                (x, y + grid_size),
                (x, y)
            ])
            
            road_intersection = unary_union(road_polygons.intersection(grid_cell)).simplify(simplify_tolerance)
            white_line_intersection = unary_union(white_lines.intersection(grid_cell)).simplify(simplify_tolerance)
            yellow_line_intersection = unary_union(yellow_lines.intersection(grid_cell)).simplify(simplify_tolerance)
            
            if not all(geom.is_empty for geom in [road_intersection, white_line_intersection, yellow_line_intersection]):
                intersection_map[(x, y)] = (road_intersection, white_line_intersection, yellow_line_intersection)
                
    return intersection_map

client = carla.Client()
client.set_timeout(120)

simplify_tolerance = 0.01
grid_size = 64

for town in ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']:
    world = client.load_world(town)
    carla_map = world.get_map()
    file_data = carla_map.to_opendrive()
    multi_road_polygons, multi_white_line_string, multi_yellow_line_string = parse_xodr_file(file_data)

    plot_shapely_road_network(multi_road_polygons, 
                              multi_white_line_string, 
                              multi_yellow_line_string, 
                              town)

    simplified_multi_road_polygons = unary_union(multi_road_polygons).simplify(simplify_tolerance)
    simplified_multi_white_line_string = unary_union(multi_white_line_string).simplify(simplify_tolerance)
    simplified_multi_yellow_line_string = unary_union(multi_yellow_line_string).simplify(simplify_tolerance)

    plot_shapely_road_network(
        simplified_multi_road_polygons, 
        simplified_multi_white_line_string, 
        simplified_multi_yellow_line_string,
        town
    )

    intersection_map = create_intersection_map(
        simplified_multi_road_polygons,
        simplified_multi_white_line_string,
        simplified_multi_yellow_line_string,
        grid_size
    )

    opencv_format_map = convert_intersection_map_to_opencv(intersection_map)

    plot_opencv_road_network(opencv_format_map, town)

    opencv_format_map_with_centered_coords = {tuple(np.array(k) + grid_size/2.): v for k, v in opencv_format_map.items()}    
    road_network = {'grid_size': grid_size, 'grid': opencv_format_map_with_centered_coords}

    stop_sign_data = compute_stop_sign_data(world, carla_map)
    traffic_light_data = compute_traffic_light_data(world, carla_map)

    map_data = {
        'road_network': road_network,
        'stop_signs': stop_sign_data,
        'traffic_lights': traffic_light_data
    }

    pathlib.Path('map_data').mkdir(exist_ok=True)
    with open(f'map_data/{town}.pkl', 'wb') as file:
        pickle.dump(map_data, file)