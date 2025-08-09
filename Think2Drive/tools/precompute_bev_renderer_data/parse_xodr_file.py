import numpy as np
from lxml import etree
from abc import ABC, abstractmethod
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString

# There always exists exactly one planView element per road
# There always exist 1+ geometry elements per planView element
# Every road element contains exactly 1 lanes elements
# Every lanes element contains 1+ laneSection elements
# Every lanes element contains 0+ laneOffset elements
# Every laneSection element contains exactly 1 center element
# Every laneSection element contains 0..1 left and 0..1 right elements but at least one of them
# Every lane element contains 0+ roadMark elements

class Geometry(ABC):
    __slots__ = 'previous_geometry', 's', 'x', 'y', 'hdg', 'length'
        
    def __init__(self, s, x, y, hdg, length, previous_geometry):
        self.s = s
        self.x = x
        self.y = y
        self.hdg = hdg
        self.length = length

        self.previous_geometry = previous_geometry
    
    @staticmethod
    def parse_geometry_element_tree(element_tree, previous_geometry):
        if element_tree[0].tag == 'line':
            return Line(element_tree, previous_geometry)
        elif element_tree[0].tag == 'arc':
            return Arc(element_tree, previous_geometry)
        elif element_tree[0].tag == 'spiral':
            raise NotImplementedError(f'The spiral is not known')
        else:
            raise NotImplementedError(f'The {element_tree[0].tag} element is not known')
    
    @abstractmethod
    def geometry_to_coordinates(self):
        raise NotImplementedError()

class Arc(Geometry):
    __slots__ = 'curvature'
    
    def __init__(self, element_tree, previous_geometry):
        s = float(element_tree.attrib['s'])
        x = float(element_tree.attrib['x'])
        y = float(element_tree.attrib['y'])
        hdg = float(element_tree.attrib['hdg'])
        length = float(element_tree.attrib['length'])
        curvature = float(element_tree[0].attrib['curvature'])
        
        super().__init__(s, x, y, hdg, length, previous_geometry)
        self.curvature = curvature

    def geometry_to_coordinates(self):
        # Calculate the central angle of the arc
        central_angle = self.curvature * self.length
        
        # Calculate the radius (absolute value because curvature can be negative)
        radius = 1 / self.curvature
        
        # Calculate the center of the arc
        center_x = self.x - np.sin(self.hdg) / self.curvature
        center_y = self.y + np.cos(self.hdg) / self.curvature
        
        # Generate points along the arc
        theta = np.linspace(self.hdg, self.hdg + central_angle, 100)
        x = center_x + radius * np.sin(theta)
        y = center_y - radius * np.cos(theta)
        
        return np.column_stack((x, y))

class Line(Geometry):
    def __init__(self, element_tree, previous_geometry):
        s = float(element_tree.attrib['s'])
        x = float(element_tree.attrib['x'])
        y = float(element_tree.attrib['y'])
        hdg = float(element_tree.attrib['hdg'])
        length = float(element_tree.attrib['length'])
        
        super().__init__(s, x, y, hdg, length, previous_geometry)

    def geometry_to_coordinates(self):
        # Number of points to generate (adjust as needed)
        num_points = 2  # For a line, we only need start and end points
        
        # Generate points along the line
        t = np.linspace(0, self.length, num_points)
        x = self.x + t * np.cos(self.hdg)
        y = self.y + t * np.sin(self.hdg)
        
        return np.column_stack((x, y))
       
class LaneSection:
    def __init__(self, lane_section_elem):
        self.s = float(lane_section_elem.attrib['s'])
        self.lanes = {}
        
        for side in ['left', 'center', 'right']:
            side_elem = lane_section_elem.find(side)
            if side_elem is not None:
                for lane_elem in side_elem.findall('lane'):
                    lane_id = int(lane_elem.attrib['id'])
                    self.lanes[lane_id] = Lane(lane_elem)

class Lane:
    def __init__(self, lane_elem):
        self.id = int(lane_elem.attrib['id'])
        self.type = lane_elem.attrib['type']
        self.width_records = []
        self.road_mark_records = []
        
        for width_elem in lane_elem.findall('width'):
            self.width_records.append({
                'sOffset': float(width_elem.attrib['sOffset']),
                'a': float(width_elem.attrib['a']),
                'b': float(width_elem.attrib['b']),
                'c': float(width_elem.attrib['c']),
                'd': float(width_elem.attrib['d'])
            })

        for road_mark_elem in lane_elem.findall('roadMark'):
            self.road_mark_records.append({
                'sOffset': float(road_mark_elem.attrib['sOffset']),
                'type': road_mark_elem.attrib['type'],
                'color': road_mark_elem.attrib['color'] if 'color' in road_mark_elem.attrib else None,
                'width': float(road_mark_elem.attrib['width']) if 'width' in road_mark_elem.attrib else 0.15
            })

    def get_width(self, s):
        for record in reversed(self.width_records):
            if s >= record['sOffset']:
                ds = s - record['sOffset']

                return record['a'] + record['b']*ds + record['c']*ds**2 + record['d']*ds**3
        return 0

    def get_road_mark(self, s):
        for record in reversed(self.road_mark_records):
            if s >= record['sOffset']:
                return record
        return None

class Road:    
    def __init__(self, road_elem):
        self.name = road_elem.attrib['name']
        self.length = float(road_elem.attrib['length'])
        self.id = int(road_elem.attrib['id'])
        self.junction_id = int(road_elem.attrib['junction'])
        
        plan_view = road_elem.find('planView')
        lanes_element = road_elem.find('lanes')

        self.lane_sections = []
        self.geometry_elements = []
        self.lane_offsets = []
        
        previous_geometry = None
        for geometry_elem in plan_view:
            geometry = Geometry.parse_geometry_element_tree(geometry_elem, previous_geometry)
            self.geometry_elements.append(geometry)
            previous_geometry = geometry
        
        for lane_offset_elem in lanes_element.findall('laneOffset'):
            self.lane_offsets.append({
                's': float(lane_offset_elem.attrib['s']),
                'a': float(lane_offset_elem.attrib['a']),
                'b': float(lane_offset_elem.attrib['b']),
                'c': float(lane_offset_elem.attrib['c']),
                'd': float(lane_offset_elem.attrib['d'])
            })

        for lane_section_elem in lanes_element.findall('laneSection'):
            self.lane_sections.append(LaneSection(lane_section_elem))
    
    def get_lane_offset(self, s):
        for offset in reversed(self.lane_offsets):
            if s >= offset['s']:
                ds = s - offset['s']
                return offset['a'] + offset['b']*ds + offset['c']*ds**2 + offset['d']*ds**3
        return 0
        
    def get_coordinates(self, s_global, t):
        # Find the relevant geometry element
        geometry = next((g for g in reversed(self.geometry_elements) if s_global >= g.s), None)
        if geometry is None:
            return None, None

        # Calculate the local s within the geometry element
        local_s = s_global - geometry.s

        # Calculate base coordinates
        if isinstance(geometry, Arc):
            angle = geometry.hdg + local_s * geometry.curvature
            radius = 1 / geometry.curvature
            base_x = - np.sin(geometry.hdg) / geometry.curvature + radius * np.sin(angle)
            base_y = np.cos(geometry.hdg) / geometry.curvature - radius * np.cos(angle)
            
            normal_x = -np.sin(angle)
            normal_y = np.cos(angle)
        else:  # Line
            base_x = local_s * np.cos(geometry.hdg)
            base_y = local_s * np.sin(geometry.hdg)
            
            normal_x = -np.sin(geometry.hdg)
            normal_y = np.cos(geometry.hdg)

        # Apply lane offset
        lane_offset = self.get_lane_offset(s_global)
        t += lane_offset

        # Apply t offset
        x =  geometry.x + base_x + t * normal_x
        y = geometry.y + base_y + t * normal_y

        return x, y

    def road_to_primitives(self):
        primitives, lines = [], []
        
        for lane_section in self.lane_sections:
            section_start = lane_section.s
            section_end = self.length if lane_section == self.lane_sections[-1] else self.lane_sections[self.lane_sections.index(lane_section) + 1].s

            for lane_id, lane in lane_section.lanes.items():
                # Generate s values relative to the start of the road, not the start of the section
                section_s_values = np.arange(section_start+1e-8, section_end-1e-8, 0.25)
                section_s_values = np.concatenate([section_s_values, [section_end-1e-8]])

                if lane_id > 0:
                    lane_t_inner = [sum(lane_section.lanes[i].get_width(s - section_start) for i in range(1, lane_id)) for s in section_s_values]
                    lane_t_outer = [sum(lane_section.lanes[i].get_width(s - section_start) for i in range(1, lane_id + 1)) for s in section_s_values]
                else:
                    lane_t_inner = [-sum(lane_section.lanes[i].get_width(s - section_start) for i in range(-1, lane_id, -1)) for s in section_s_values]
                    lane_t_outer = [-sum(lane_section.lanes[i].get_width(s - section_start) for i in range(-1, lane_id - 1, -1)) for s in section_s_values]
                   
                inner_points = np.array([self.get_coordinates(s, t) for s, t in zip(section_s_values, lane_t_inner)])
                outer_points = np.array([self.get_coordinates(s, t) for s, t in zip(section_s_values, lane_t_outer)])

                # add lanes
                if lane_id != 0 and lane.type in ['driving']:
                    primitives.append((inner_points, outer_points))

                # add road mark
                if self.junction_id == -1 and (lane.type == 'driving' or \
                                               lane_id>=0 and lane_id+1 in lane_section.lanes and lane_section.lanes[lane_id+1].type=='driving' or \
                                               lane_id<=0 and lane_id-1 in lane_section.lanes and lane_section.lanes[lane_id-1].type=='driving'):
                    line = None
                    for (s, p) in zip(section_s_values, outer_points):
                        road_mark = lane.get_road_mark(s - section_start)  # Adjust s for road_mark lookup
                        if road_mark:
                            if not line:
                                line = (road_mark['color'], [])
                            elif line[0] != road_mark['color']:
                                lines.append(line)
                                line = (road_mark['color'], [])
                            
                            line[1].append(p)
                    
                    if line:
                        lines.append(line)
                
        return primitives, lines

def parse_xodr_file(file_data):
    root = etree.fromstring(file_data.encode('utf-8'))

    roads = []
    for road_element in root.findall('road'):
        roads.append(Road(road_element))

    all_primitives = []
    all_lines = []
    for road in roads:
        primitives, lines = road.road_to_primitives()
        all_primitives.extend(primitives)
        all_lines.extend(lines)

    shapely_polygons = []
    for p1, p2 in all_primitives:
        polygon = np.concatenate([p1, p2[::-1]], axis=0)
        shapely_polygons.append(Polygon(polygon))

    multi_road_polygons = MultiPolygon(shapely_polygons)

    white_lines = []
    yellow_lines = []
    for line in all_lines:
        if len(line[1]) <= 1:
            continue

        if line[0] == 'white':
            white_lines.append(LineString(np.array(line[1])))
        elif line[0] == 'yellow':
            yellow_lines.append(LineString(np.array(line[1])))
    
    multi_white_line_string = MultiLineString(white_lines)
    multi_yellow_line_string = MultiLineString(yellow_lines)

    return multi_road_polygons, multi_white_line_string, multi_yellow_line_string

if __name__ == '__main__':
    multi_road_polygons, multi_white_line_string, multi_yellow_line_string = parse_xodr_file('Town03.xodr')