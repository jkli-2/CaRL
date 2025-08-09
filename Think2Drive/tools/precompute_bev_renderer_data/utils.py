import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from shapely.geometry import MultiPolygon, MultiLineString
from matplotlib.collections import PatchCollection
import numpy as np
import pathlib

def plot_shapely_road_network(multi_road_polygons, multi_white_line_string, multi_yellow_line_string, map_name, xlimits=None, ylimits=None):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    
    # Plot road polygons
    if isinstance(multi_road_polygons, MultiPolygon):
        patches = [Polygon(poly.exterior.coords) for poly in multi_road_polygons.geoms]
        ax.add_collection(PatchCollection(patches, facecolor='gray', edgecolor='none', alpha=0.5))
    # Plot road polygons
    elif isinstance(multi_road_polygons, Polygon):
        patches = [multi_road_polygons]
        ax.add_collection(PatchCollection(patches, facecolor='gray', edgecolor='none', alpha=0.5))
        
    # Plot white lines
    if isinstance(multi_white_line_string, MultiLineString):
        for line in multi_white_line_string.geoms:
            ax.plot(*line.xy, color='pink', linewidth=1)

    # Plot yellow lines
    if isinstance(multi_yellow_line_string, MultiLineString):
        for line in multi_yellow_line_string.geoms:
            ax.plot(*line.xy, color='yellow', linewidth=1)
    
    if xlimits or ylimits:
        xlimits and ax.set_xlim(xlimits[0], xlimits[1])
        ylimits and ax.set_ylim(ylimits[0], ylimits[1])
    else:
        ax.set_aspect('equal')
        ax.autoscale_view()
        
    plt.title('Road Map')
    plt.show()

def plot_opencv_road_network(opencv_format_map, map_name, figsize=(12, 12), dpi=200, save_to='images'):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Collect all coordinates to determine plot bounds
    all_coords = []
    
    # Plot road polygons
    road_patches = []
    for cell_data in opencv_format_map.values():
        road_polygons = cell_data[0]
        for poly in road_polygons:
            road_patches.append(Polygon(poly, closed=True))
            all_coords.extend(poly)
    
    road_collection = PatchCollection(road_patches, alpha=0.4, facecolor='gray', edgecolor='none')
    ax.add_collection(road_collection)
    
    # Plot white lines
    for cell_data in opencv_format_map.values():
        white_lines = cell_data[1]
        for line in white_lines:
            ax.plot(line[:, 0], line[:, 1], color='pink', linewidth=1, alpha=0.7)
            all_coords.extend(line)
    
    # Plot yellow lines
    for cell_data in opencv_format_map.values():
        yellow_lines = cell_data[2]
        for line in yellow_lines:
            ax.plot(line[:, 0], line[:, 1], color='yellow', linewidth=1, alpha=0.7)
            all_coords.extend(line)
    
    # Set plot bounds
    if all_coords:
        all_coords = np.array(all_coords)
        x_min, y_min = all_coords.min(axis=0)
        x_max, y_max = all_coords.max(axis=0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    # Set equal aspect ratio
    ax.set_aspect('equal', 'datalim')
    
    # Remove axes
    # ax.set_axis_off()
    
    # Set title
    ax.set_title("OpenCV Road Network Visualization")
    
    # Tight layout
    plt.tight_layout()
    pathlib.Path(save_to).mkdir(exist_ok=True, parents=True)
    plt.savefig(f'{save_to}/road_network_{map_name}.jpg')
    plt.show()