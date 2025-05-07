# Module declaration at the beginning of file
import os
import re
import math
import json
import sys
import importlib
from datetime import datetime
from collections import defaultdict


try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None


import bpy
from mathutils import Vector, Matrix, Euler
from bpy.props import StringProperty, CollectionProperty, IntProperty, FloatProperty, PointerProperty
from bpy.types import Panel, Operator, PropertyGroup

# ------------------------------------------------------------------------------

HARD_CODED_GPT_API_KEY = ""
HARD_CODED_GPT_PROJECT_ID = ""
HARD_CODED_GPT_MODEL = "gpt-4o"
HARD_CODED_GPT_MAX_TOKENS = 1024
HARD_CODED_GPT_TEMPERATURE = 0.7
# ------------------------------------------------------------------------------

bl_info = {
    "name": "NL Control - AI Scene Interaction",
    "author": "AI Assistant",
    "version": (1, 2),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > NL Control",
    "description": "Natural Language Scene Control using GPT-4o",
    "category": "3D View",
}

SPATIAL_RELATIONS = {
    "LEFT_OF": "left of",
    "RIGHT_OF": "right of",
    "IN_FRONT_OF": "in front of",
    "BEHIND": "behind",
    "ON": "on",
    "UNDER": "under",
    "ABOVE": "above",
    "BELOW": "below",
    "INSIDE": "inside",
    "NEAR": "near"
}

OBJECT_TYPE_TRANSLATIONS = {

    "桌子": "table", "椅子": "chair", "沙发": "sofa", "灯": "lamp",
    "植物": "plant", "书架": "bookshelf", "书柜": "bookcase",
    "地板": "floor", "墙": "wall", "窗户": "window", "门": "door",
    "瓶子": "bottle", "杯子": "glass", "盘子": "plate", "碗": "bowl",
    "厨房": "kitchen", "客厅": "living", "卧室": "bedroom", "浴室": "bathroom",

    "table": "table", "chair": "chair", "sofa": "sofa", "lamp": "lamp",
    "plant": "plant", "bookshelf": "bookshelf", "bookcase": "bookcase",
    "floor": "floor", "wall": "wall", "window": "window", "door": "door",
    "bottle": "bottle", "glass": "glass", "plate": "plate", "bowl": "bowl",
    "kitchen": "kitchen", "living": "living", "bedroom": "bedroom", "bathroom": "bathroom"
}


OPERATION_TYPES = {
    "MOVE": "move",
    "ROTATE": "rotate",
    "COPY": "copy",
    "DELETE": "delete",
    "ENLARGE": "enlarge",
    "SHRINK": "shrink"
}


# 属性组
class NL_ControlProperties(PropertyGroup):
    command: StringProperty(
        name="Command",
        description="Natural language command to process",
        default=""
    )
    screenshot_dir: StringProperty(
        name="Screenshot Directory",
        description="Directory to save screenshots",
        default="F:/Blender_Screenshots",
        subtype='DIR_PATH'
    )
    scene_graph_dir: StringProperty(
        name="Scene Graph Directory",
        description="Directory to save scene graphs (and top surfaces)",
        default="F:/Blender_SceneGraphs",
        subtype='DIR_PATH'
    )


# 实用函数
def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def update_status(context, message, log=True):
    try:
        context.scene.nl_control_status = message
    except Exception as e:
        print(f"Warning: Could not update status UI: {str(e)}")
    if log:
        print(f"[NL Control] {message}")


def get_active_view3d_region(context):
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    return region, area.spaces.active.region_3d
    return None, None


def extract_semantic_type(name):
    base_name = re.sub(r'\.\d+$', '', name.lower())
    base_name = re.sub(r'[^a-zA-Z]', '', base_name)
    mapping = {
        'bkshlf': 'bookshelf', 'bkcase': 'bookcase', 'tbl': 'table',
        'chr': 'chair', 'sfa': 'sofa', 'lght': 'light', 'lmp': 'lamp',
        'plnt': 'plant', 'flr': 'floor', 'wll': 'wall', 'wndw': 'window',
        'dr': 'door', 'bttl': 'bottle', 'glss': 'glass',
        'plate': 'plate', 'bowl': 'bowl', 'ktchn': 'kitchen',
        'lv': 'living', 'brm': 'bedroom', 'bthrm': 'bathroom'
    }
    return mapping.get(base_name, base_name)


def extract_object_types_from_command(nl_command):

    command_lower = nl_command.lower()
    found_types = set()
    for obj_name, obj_type in OBJECT_TYPE_TRANSLATIONS.items():
        if obj_name.lower() in command_lower:
            found_types.add(obj_type)
    return list(found_types)



def world_to_camera_coords(world_point, cam_loc, R_inv):

    P_cam = R_inv @ (world_point - cam_loc)
    new_x = P_cam.x
    new_y = -P_cam.z
    new_z = P_cam.y
    return Vector((new_x, new_y, new_z))


def camera_to_world_coords(cam_point, cam_loc, R):

    P_cam = Vector((cam_point.x, cam_point.z, -cam_point.y))
    P_world = R @ P_cam + cam_loc
    return P_world



def get_visible_objects_in_view(context):
    region, rv3d = get_active_view3d_region(context)
    if not region or not rv3d:
        print("No active 3D view found")
        return []

    view_width, view_height = region.width, region.height
    grid_size = 20
    visible_objects = set()
    view_matrix = rv3d.view_matrix
    view_matrix_inv = view_matrix.inverted()
    view_location = view_matrix_inv.translation

    for x in range(0, view_width, grid_size):
        for y in range(0, view_height, grid_size):
            coord_norm = ((x / view_width) * 2 - 1, (y / view_height) * 2 - 1)
            if hasattr(rv3d, 'window_matrix'):
                win_matrix = rv3d.window_matrix
                win_matrix_inv = win_matrix.inverted()
                p = Vector((coord_norm[0], coord_norm[1], 0.0, 1.0))
                p = win_matrix_inv @ p
                p = p / p.w
                v = Vector((coord_norm[0], coord_norm[1], 1.0, 1.0))
                v = win_matrix_inv @ v
                v = v / v.w
                v = (v - p).normalized()
                ray_origin = view_location
                ray_direction = view_matrix_inv.to_3x3() @ Vector((v.x, v.y, v.z))
            else:
                ray_origin = view_location
                ray_direction = (Vector((0, 0, -1)) @ view_matrix_inv.to_3x3()).normalized()

            hit, loc, norm, index, hit_obj, matrix = bpy.context.scene.ray_cast(
                bpy.context.view_layer.depsgraph, ray_origin, ray_direction)

            if hit and hit_obj and hit_obj.visible_get():
                visible_objects.add(hit_obj)

    result = list(visible_objects)
    result.sort(key=lambda obj: obj.name)
    return result


def get_viewport_screenshot(context):
    screenshot_dir = context.scene.nl_control_props.screenshot_dir
    os.makedirs(screenshot_dir, exist_ok=True)
    timestamp = get_timestamp()
    filepath = os.path.join(screenshot_dir, f"nl_command_{timestamp}.png")

    original_filepath = context.scene.render.filepath
    original_percentage = context.scene.render.resolution_percentage
    original_x = context.scene.render.resolution_x
    original_y = context.scene.render.resolution_y

    context.scene.render.filepath = filepath
    context.scene.render.resolution_percentage = 100

    region, rv3d = get_active_view3d_region(context)
    if region:
        context.scene.render.resolution_x = region.width
        context.scene.render.resolution_y = region.height

    bpy.ops.render.opengl(write_still=True)

    context.scene.render.filepath = original_filepath
    context.scene.render.resolution_percentage = original_percentage
    context.scene.render.resolution_x = original_x
    context.scene.render.resolution_y = original_y

    print(f"Screenshot saved to: {filepath}")
    return filepath


def collect_object_data(obj):
    if not hasattr(obj, "bound_box") or not obj.bound_box:
        return None

    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_x = min(corner.x for corner in corners)
    min_y = min(corner.y for corner in corners)
    min_z = min(corner.z for corner in corners)
    max_x = max(corner.x for corner in corners)
    max_y = max(corner.y for corner in corners)
    max_z = max(corner.z for corner in corners)

    center = [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
    dims = [max_x - min_x, max_y - min_y, max_z - min_z]

    bounds = {
        "min": [min_x, min_y, min_z],
        "max": [max_x, max_y, max_z],
        "center": center,
        "dimensions": dims
    }

    return {
        "id": obj.name,
        "name": obj.name,
        "semantic_type": extract_semantic_type(obj.name),
        "type": obj.type,
        "location": [obj.location.x, obj.location.y, obj.location.z],
        "rotation": [math.degrees(obj.rotation_euler.x),
                     math.degrees(obj.rotation_euler.y),
                     math.degrees(obj.rotation_euler.z)],
        "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
        "bounds": bounds,
        "parent": obj.parent.name if obj.parent else None
    }


def build_scene_graph(objects, context):
    bpy.context.view_layer.update()

    graph = {"objects": [], "relationships": []}
    region, rv3d = get_active_view3d_region(context)
    if not region or not rv3d:
        print("No active 3D view found")
        return graph

    cam_world_matrix = rv3d.view_matrix.inverted()
    cam_loc = cam_world_matrix.translation
    R = cam_world_matrix.to_3x3()
    R_inv = R.transposed()

    print("Camera location:", cam_loc)
    for obj in objects:
        data = collect_object_data(obj)
        if data:
            P = obj.location
            P_cam = R_inv @ (P - cam_loc)
            new_x = P_cam.x
            new_y = -P_cam.z
            new_z = P_cam.y
            data["new_coord"] = [new_x, new_y, new_z]
            graph["objects"].append(data)

    semantic_mapping = defaultdict(list)
    for data in graph["objects"]:
        semantic_mapping[data["semantic_type"]].append(data["id"])
    graph["semantic_mapping"] = dict(semantic_mapping)

    num_objs = len(graph["objects"])
    for i in range(num_objs):
        for j in range(i + 1, num_objs):
            obj1 = graph["objects"][i]
            obj2 = graph["objects"][j]
            coord1 = Vector(obj1["new_coord"])
            coord2 = Vector(obj2["new_coord"])

            relation_x = "left" if coord1.x < coord2.x else "right" if coord1.x > coord2.x else "equal"
            relation_y = "front" if coord1.y < coord2.y else "back" if coord1.y > coord2.y else "equal"
            relation_z = "down" if coord1.z < coord2.z else "up" if coord1.z > coord2.z else "equal"

            relationship = {
                "pair": [obj1["id"], obj2["id"]],
                "x": {"relation": relation_x, "strength": abs(coord1.x - coord2.x)},
                "y": {"relation": relation_y, "strength": abs(coord1.y - coord2.y)},
                "z": {"relation": relation_z, "strength": abs(coord1.z - coord2.z)}
            }
            graph["relationships"].append(relationship)


    graph["camera"] = {
        "location": [cam_loc.x, cam_loc.y, cam_loc.z],
        "rotation_matrix": [[R[i][j] for j in range(3)] for i in range(3)]
    }

    return graph


def analyze_scene(context, objects_list):
    return build_scene_graph(objects_list, context)


def save_scene_graph_to_file(context, scene_graph):
    graph_dir = context.scene.nl_control_props.scene_graph_dir
    os.makedirs(graph_dir, exist_ok=True)
    timestamp = get_timestamp()
    filepath = os.path.join(graph_dir, f"scene_graph_{timestamp}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(scene_graph, f, ensure_ascii=False, indent=2)
    print(f"Scene graph saved to: {filepath}")
    return filepath


def filter_objects_by_types(all_objects, object_types):

    if not object_types:
        return all_objects

    filtered_objects = [
        obj for obj in all_objects
        if extract_semantic_type(obj.name) in object_types
    ]

    return filtered_objects or all_objects


def filter_scene_graph_for_types(scene_graph, object_types):

    if not object_types:
        return scene_graph

    matching_ids = set()
    for obj in scene_graph["objects"]:
        if obj["semantic_type"] in object_types:
            matching_ids.add(obj["id"])

    if not matching_ids:
        return scene_graph

    filtered_objects = [obj for obj in scene_graph["objects"] if obj["id"] in matching_ids]
    filtered_relationships = [
        rel for rel in scene_graph["relationships"]
        if rel["pair"][0] in matching_ids and rel["pair"][1] in matching_ids
    ]

    filtered_mapping = {
        semantic_type: obj_ids
        for semantic_type, obj_ids in scene_graph["semantic_mapping"].items()
        if semantic_type in object_types
    }

    filtered_graph = {
        "objects": filtered_objects,
        "relationships": filtered_relationships,
        "semantic_mapping": filtered_mapping,
        "camera": scene_graph["camera"]
    }

    return filtered_graph


def extract_top_surface(obj, z_threshold=0.85, merge_tol=1e-3):
    import bmesh
    from mathutils.geometry import convex_hull_2d
    from mathutils import Vector
    if obj.type != 'MESH':
        return None


    region, rv3d = get_active_view3d_region(bpy.context)
    if not region or not rv3d:
        print("No active 3D view found")
        return None

    cam_world_matrix = rv3d.view_matrix.inverted()
    cam_loc = cam_world_matrix.translation
    R = cam_world_matrix.to_3x3()
    R_inv = R.transposed()

    mesh = obj.data
    world_matrix = obj.matrix_world


    faces = []
    for face in mesh.polygons:
        normal_world = world_matrix.to_3x3() @ face.normal
        center_world = world_matrix @ face.center
        if normal_world.z > z_threshold:
            faces.append((face, normal_world, center_world))
    if not faces:
        return None


    faces_sorted = sorted(faces, key=lambda f: f[2].z, reverse=True)


    candidate = faces_sorted[0]
    for f in faces_sorted[1:]:
        if f[0].area >= 1.5 * candidate[0].area:
            candidate = f

    selected_face = candidate
    normal_ref = selected_face[1]
    center_ref = selected_face[2]


    def is_similar(face):
        normal = world_matrix.to_3x3() @ face.normal
        center = world_matrix @ face.center
        return normal.dot(normal_ref) > 0.98 and abs(center.z - center_ref.z) < 0.01

    similar_faces = [f for f in mesh.polygons if is_similar(f)]
    verts = set()
    for f in similar_faces:
        for vid in f.vertices:
            v_world = world_matrix @ mesh.vertices[vid].co
            verts.add(v_world.freeze())
    verts = list(verts)
    if len(verts) < 3:
        return None


    projected = [(v.x, v.y) for v in verts]
    hull_indices = convex_hull_2d(projected)
    hull_verts = [list(verts[i]) for i in hull_indices]


    def rdp(points, epsilon):
        import math
        if len(points) < 3:
            return points

        def point_line_distance(pt, start, end):
            if start == end:
                return math.dist(pt, start)
            x0, y0 = pt
            x1, y1 = start
            x2, y2 = end
            num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            den = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            return num / den

        dmax = 0.0
        index = 0
        for i in range(1, len(points) - 1):
            d = point_line_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d
        if dmax > epsilon:
            rec_results1 = rdp(points[:index + 1], epsilon)
            rec_results2 = rdp(points[index:], epsilon)
            return rec_results1[:-1] + rec_results2
        else:
            return [points[0], points[-1]]

    simplified_2d = rdp([(v[0], v[1]) for v in hull_verts], epsilon=0.01)

    simplified_verts_world = [[pt[0], pt[1], center_ref.z] for pt in simplified_2d]


    simplified_verts = []
    for v_world in simplified_verts_world:
        v_cam = world_to_camera_coords(Vector(v_world), cam_loc, R_inv)
        simplified_verts.append([v_cam.x, v_cam.y, v_cam.z])


    sum_vec = Vector((0, 0, 0))
    for v in simplified_verts:
        sum_vec += Vector(v)
    center_cam = sum_vec / len(simplified_verts)


    normal_cam = world_to_camera_coords(normal_ref, Vector((0, 0, 0)), R_inv)

    total_area = sum(f.area for f in mesh.polygons if is_similar(f))

    return {
        "center": list(center_cam),
        "normal": list(normal_cam.normalized()),
        "vertices": simplified_verts,
        "area": total_area
    }



def execute_scene_operation(context, operation_data):

    print(f"Executing operation: {json.dumps(operation_data, indent=2)}")


    region, rv3d = get_active_view3d_region(context)
    if not region or not rv3d:
        print("No active 3D view found")
        return False, "No active 3D view found"

    cam_world_matrix = rv3d.view_matrix.inverted()
    cam_loc = cam_world_matrix.translation
    R = cam_world_matrix.to_3x3()

    operation_type = operation_data.get("action", "").lower()
    target_object_id = operation_data.get("target_object", {}).get("id", "")
    anchor_object_id = operation_data.get("anchor_object", {}).get("id", "")


    target_obj = bpy.data.objects.get(target_object_id)
    if not target_obj:
        print(f"Target object not found: {target_object_id}")
        return False, f"Target object not found: {target_object_id}"


    anchor_obj = None
    if operation_type != OPERATION_TYPES["DELETE"] and anchor_object_id:
        anchor_obj = bpy.data.objects.get(anchor_object_id)
        if not anchor_obj and operation_type != OPERATION_TYPES["DELETE"]:
            print(f"Anchor object not found: {anchor_object_id}")
            return False, f"Anchor object not found: {anchor_object_id}"


    if operation_type == OPERATION_TYPES["MOVE"]:
        return move_object(target_obj, anchor_obj, operation_data, cam_loc, R)
    elif operation_type == OPERATION_TYPES["ROTATE"]:
        return rotate_object(target_obj, operation_data)
    elif operation_type == OPERATION_TYPES["COPY"]:
        return copy_object(target_obj, anchor_obj, operation_data, cam_loc, R)
    elif operation_type == OPERATION_TYPES["DELETE"]:
        return delete_object(target_obj)
    elif operation_type == OPERATION_TYPES["ENLARGE"]:
        return scale_object(target_obj, True, operation_data)
    elif operation_type == OPERATION_TYPES["SHRINK"]:
        return scale_object(target_obj, False, operation_data)
    else:
        print(f"Unknown operation type: {operation_type}")
        return False, f"Unknown operation type: {operation_type}"


def move_object(obj, anchor_obj, operation_data, cam_loc, R):

    try:

        target_position_cam = None
        if "position" in operation_data:

            target_position_cam = Vector(operation_data["position"])
        elif anchor_obj and "top_surface" in operation_data.get("anchor_object", {}):

            top_surface = operation_data["anchor_object"]["top_surface"]
            target_position_cam = Vector(top_surface["center"])
        elif anchor_obj:

            top_surface = extract_top_surface(anchor_obj)
            if top_surface:
                target_position_cam = Vector(top_surface["center"])
            else:

                anchor_loc_world = anchor_obj.location.copy()
                anchor_loc_cam = world_to_camera_coords(anchor_loc_world, cam_loc, R.transposed())
                target_position_cam = anchor_loc_cam.copy()

                target_position_cam.z += obj.dimensions.z / 2

        if not target_position_cam:
            return False, "No valid target position found"


        original_position = obj.location.copy()


        target_position_world = camera_to_world_coords(target_position_cam, cam_loc, R)


        obj.location = target_position_world


        bpy.context.view_layer.update()

        result_data = {
            "action": OPERATION_TYPES["MOVE"],
            "target_object": {
                "id": obj.name,
                "original_position": [original_position.x, original_position.y, original_position.z],
                "new_position": [target_position_world.x, target_position_world.y, target_position_world.z]
            }
        }

        if anchor_obj:
            result_data["anchor_object"] = {
                "id": anchor_obj.name,
                "position": [anchor_obj.location.x, anchor_obj.location.y, anchor_obj.location.z]
            }

        return True, result_data

    except Exception as e:
        print(f"Error moving object: {str(e)}")
        return False, f"Error moving object: {str(e)}"


def rotate_object(obj, operation_data):

    try:

        angle = operation_data.get("angle", 90)
        axis = operation_data.get("axis", "Z").upper()


        original_rotation = obj.rotation_euler.copy()


        new_rotation = obj.rotation_euler.copy()
        if axis == "X":
            new_rotation.x += math.radians(angle)
        elif axis == "Y":
            new_rotation.y += math.radians(angle)
        else:
            new_rotation.z += math.radians(angle)


        obj.rotation_euler = new_rotation


        bpy.context.view_layer.update()

        result_data = {
            "action": OPERATION_TYPES["ROTATE"],
            "target_object": {
                "id": obj.name,
                "original_rotation": [math.degrees(original_rotation.x),
                                      math.degrees(original_rotation.y),
                                      math.degrees(original_rotation.z)],
                "new_rotation": [math.degrees(new_rotation.x),
                                 math.degrees(new_rotation.y),
                                 math.degrees(new_rotation.z)],
                "angle": angle,
                "axis": axis
            }
        }

        return True, result_data

    except Exception as e:
        print(f"Error rotating object: {str(e)}")
        return False, f"Error rotating object: {str(e)}"


def copy_object(obj, anchor_obj, operation_data, cam_loc, R):

    try:

        new_obj = obj.copy()
        new_obj.data = obj.data.copy()
        bpy.context.collection.objects.link(new_obj)


        base_name = re.sub(r'\.\d+$', '', obj.name)
        new_obj.name = f"{base_name}.copy"


        target_position_cam = None
        if "position" in operation_data:

            target_position_cam = Vector(operation_data["position"])
        elif anchor_obj and "top_surface" in operation_data.get("anchor_object", {}):

            top_surface = operation_data["anchor_object"]["top_surface"]
            target_position_cam = Vector(top_surface["center"])
        elif anchor_obj:

            top_surface = extract_top_surface(anchor_obj)
            if top_surface:
                target_position_cam = Vector(top_surface["center"])
            else:

                anchor_loc_world = anchor_obj.location.copy()
                anchor_loc_cam = world_to_camera_coords(anchor_loc_world, cam_loc, R.transposed())
                target_position_cam = anchor_loc_cam.copy()

                target_position_cam.z += new_obj.dimensions.z / 2

        if target_position_cam:

            target_position_world = camera_to_world_coords(target_position_cam, cam_loc, R)

            new_obj.location = target_position_world


        bpy.context.view_layer.update()

        result_data = {
            "action": OPERATION_TYPES["COPY"],
            "target_object": {
                "id": obj.name,
                "position": [obj.location.x, obj.location.y, obj.location.z]
            },
            "new_object": {
                "id": new_obj.name,
                "position": [new_obj.location.x, new_obj.location.y, new_obj.location.z]
            }
        }

        if anchor_obj:
            result_data["anchor_object"] = {
                "id": anchor_obj.name,
                "position": [anchor_obj.location.x, anchor_obj.location.y, anchor_obj.location.z]
            }

        return True, result_data

    except Exception as e:
        print(f"Error copying object: {str(e)}")
        return False, f"Error copying object: {str(e)}"


def delete_object(obj):
    """删除指定对象"""
    try:

        obj_data = {
            "id": obj.name,
            "position": [obj.location.x, obj.location.y, obj.location.z],
            "rotation": [math.degrees(obj.rotation_euler.x),
                         math.degrees(obj.rotation_euler.y),
                         math.degrees(obj.rotation_euler.z)],
            "scale": [obj.scale.x, obj.scale.y, obj.scale.z]
        }


        bpy.data.objects.remove(obj, do_unlink=True)


        bpy.context.view_layer.update()

        result_data = {
            "action": OPERATION_TYPES["DELETE"],
            "target_object": obj_data
        }

        return True, result_data

    except Exception as e:
        print(f"Error deleting object: {str(e)}")
        return False, f"Error deleting object: {str(e)}"


def scale_object(obj, enlarge, operation_data):

    try:

        scale_factor = operation_data.get("scale_factor", 1.5 if enlarge else 0.75)


        original_scale = obj.scale.copy()
        original_dimensions = obj.dimensions.copy()


        new_scale = original_scale.copy()
        new_scale.x *= scale_factor
        new_scale.y *= scale_factor
        new_scale.z *= scale_factor


        obj.scale = new_scale


        bpy.context.view_layer.update()

        new_dimensions = obj.dimensions.copy()

        action_type = OPERATION_TYPES["ENLARGE"] if enlarge else OPERATION_TYPES["SHRINK"]
        result_data = {
            "action": action_type,
            "target_object": {
                "id": obj.name,
                "original_scale": [original_scale.x, original_scale.y, original_scale.z],
                "new_scale": [new_scale.x, new_scale.y, new_scale.z],
                "original_dimensions": [original_dimensions.x, original_dimensions.y, original_dimensions.z],
                "new_dimensions": [new_dimensions.x, new_dimensions.y, new_dimensions.z],
                "scale_factor": scale_factor
            }
        }

        return True, result_data

    except Exception as e:
        print(f"Error scaling object: {str(e)}")
        return False, f"Error scaling object: {str(e)}"



def parse_gpt_response(response_text):
    """解析GPT的响应，提取操作指令和分析过程，支持返回多个命令"""
    parsed = None
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:

        json_match = re.search(r'```(?:json)?\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from: {json_str}")
    if parsed is None:
        return None

    if isinstance(parsed, dict):
        if "commands" in parsed:
            return parsed["commands"]
        elif "command" in parsed:
            return [parsed["command"]]
        else:
            return [parsed]

    if isinstance(parsed, list):
        return parsed
    return None


# Operator Classes
class NL_OT_SetScreenshotDir(Operator):
    bl_idname = "nl_control.set_screenshot_dir"
    bl_label = "Set Screenshot Directory"
    directory: StringProperty(
        name="Directory",
        description="Choose directory to save screenshots",
        subtype='DIR_PATH'
    )

    def execute(self, context):
        context.scene.nl_control_props.screenshot_dir = self.directory
        update_status(context, f"Screenshot directory set to: {self.directory}")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class NL_OT_SetSceneGraphDir(Operator):
    bl_idname = "nl_control.set_scene_graph_dir"
    bl_label = "Set Scene Graph Directory"
    directory: StringProperty(
        name="Directory",
        description="Choose directory to save scene graphs",
        subtype='DIR_PATH'
    )

    def execute(self, context):
        context.scene.nl_control_props.scene_graph_dir = self.directory
        update_status(context, f"Scene graph directory set to: {self.directory}")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class NL_OT_TakeScreenshot(Operator):
    bl_idname = "nl_control.take_screenshot"
    bl_label = "Take Screenshot"

    def execute(self, context):
        context.scene.nl_control_screenshot = get_viewport_screenshot(context)
        update_status(context, f"Screenshot saved: {context.scene.nl_control_screenshot}")
        return {'FINISHED'}


class NL_OT_GetVisibleObjects(Operator):
    bl_idname = "nl_control.get_visible_objects"
    bl_label = "Get Visible Objects"

    def execute(self, context):
        visible_objects = get_visible_objects_in_view(context)
        update_status(context, f"Found {len(visible_objects)} visible objects")
        context.scene.nl_control_visible_objects.clear()
        for obj in visible_objects:
            item = context.scene.nl_control_visible_objects.add()
            item.name = obj.name
            item.type = obj.type
            item.semantic_type = extract_semantic_type(obj.name)
        return {'FINISHED'}


class NL_OT_BuildSceneGraph(Operator):
    bl_idname = "nl_control.build_scene_graph"
    bl_label = "Build Scene Graph"

    def execute(self, context):
        visible_objects = get_visible_objects_in_view(context)
        scene_graph = analyze_scene(context, visible_objects)
        scene_graph_str = json.dumps(scene_graph, indent=2, ensure_ascii=False)
        context.scene.nl_control_scene_graph = scene_graph_str
        filepath = save_scene_graph_to_file(context, scene_graph)
        context.scene.nl_control_scene_graph_path = filepath
        update_status(context,
                      f"Built scene graph with {len(scene_graph['objects'])} objects and {len(scene_graph['relationships'])} relationships")
        return {'FINISHED'}


class NL_OT_AnalyzeScene(Operator):
    bl_idname = "nl_control.analyze_scene"
    bl_label = "Analyze Scene"

    def execute(self, context):
        visible_objects = get_visible_objects_in_view(context)
        update_status(context, f"Found {len(visible_objects)} visible objects")
        context.scene.nl_control_visible_objects.clear()
        for obj in visible_objects:
            item = context.scene.nl_control_visible_objects.add()
            item.name = obj.name
            item.type = obj.type
            item.semantic_type = extract_semantic_type(obj.name)

        scene_graph = analyze_scene(context, visible_objects)
        scene_graph_str = json.dumps(scene_graph, indent=2, ensure_ascii=False)
        context.scene.nl_control_scene_graph = scene_graph_str

        graph_filepath = save_scene_graph_to_file(context, scene_graph)
        context.scene.nl_control_scene_graph_path = graph_filepath

        screenshot_path = get_viewport_screenshot(context)
        context.scene.nl_control_screenshot = screenshot_path

        update_status(context,
                      f"Scene analysis complete: {len(visible_objects)} objects, screenshot and scene graph saved")
        return {'FINISHED'}


class NL_OT_ExtractSurfaces(Operator):
    bl_idname = "nl_control.extract_surfaces"
    bl_label = "Extract Top Surfaces"
    bl_description = "Extract the top surfaces of all visible objects and save as a JSON file"

    def execute(self, context):
        visible_objects = get_visible_objects_in_view(context)
        surface_info = {}


        region, rv3d = get_active_view3d_region(context)
        if not region or not rv3d:
            print("No active 3D view found")
            return {'CANCELLED'}

        cam_world_matrix = rv3d.view_matrix.inverted()
        cam_loc = cam_world_matrix.translation
        R = cam_world_matrix.to_3x3()
        R_inv = R.transposed()

        for obj in visible_objects:

            bottom_world = list(obj.location)
            bottom_cam = world_to_camera_coords(Vector(bottom_world), cam_loc, R_inv)

            top = extract_top_surface(obj)
            if top:
                surface_info[obj.name] = {
                    "bottom_point": list(bottom_cam),
                    "top_surface": top
                }

        out_dir = context.scene.nl_control_props.scene_graph_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"surface_info_{get_timestamp()}.json")

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(surface_info, f, indent=2, ensure_ascii=False)

        update_status(context, f"Surface info saved to: {out_path}")
        self.report({'INFO'}, f"Surface info saved to: {out_path}")
        return {'FINISHED'}


class NL_OT_VisualizeTopSurface(Operator):
    bl_idname = "nl_control.visualize_top_surface"
    bl_label = "Visualize Top Surface"
    bl_description = "Draw top surface of selected object(s) as a bright green outline"

    def execute(self, context):

        region, rv3d = get_active_view3d_region(context)
        if not region or not rv3d:
            print("No active 3D view found")
            return {'CANCELLED'}

        cam_world_matrix = rv3d.view_matrix.inverted()
        cam_loc = cam_world_matrix.translation
        R = cam_world_matrix.to_3x3()

        # Remove existing TopSurface objects
        for obj in bpy.data.objects:
            if obj.name.startswith("TopSurface_"):
                bpy.data.objects.remove(obj, do_unlink=True)

        for obj in context.selected_objects:
            result = extract_top_surface(obj)
            if not result:
                self.report({'WARNING'}, f"No valid top surface for {obj.name}")
                continue

            verts_cam = result["vertices"]
            if len(verts_cam) < 3:
                continue


            verts_world = []
            for v_cam in verts_cam:
                v_world = camera_to_world_coords(Vector(v_cam), cam_loc, R)
                verts_world.append(v_world)

            # Create curve data
            curve_data = bpy.data.curves.new(name=f"TopSurface_{obj.name}", type='CURVE')
            curve_data.dimensions = '3D'
            spline = curve_data.splines.new(type='POLY')

            spline.points.add(len(verts_world))
            for i, v in enumerate(verts_world + [verts_world[0]]):
                spline.points[i].co = (v.x, v.y, v.z, 1)

            curve_obj = bpy.data.objects.new(f"TopSurface_{obj.name}", curve_data)
            context.collection.objects.link(curve_obj)
            curve_obj.location = (0, 0, 0)

            # Create a green emissive material
            mat = bpy.data.materials.get("TopSurfaceMaterial")
            if not mat:
                mat = bpy.data.materials.new(name="TopSurfaceMaterial")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                nodes.clear()
                output_node = nodes.new(type='ShaderNodeOutputMaterial')
                emission_node = nodes.new(type='ShaderNodeEmission')
                emission_node.inputs["Color"].default_value = (0.0, 1.0, 0.0, 1.0)
                emission_node.inputs["Strength"].default_value = 20.0
                links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

            if len(curve_obj.data.materials) == 0:
                curve_obj.data.materials.append(mat)

        self.report({'INFO'}, "Top surface visualized as bright green.")
        return {'FINISHED'}




class NL_OT_ExecuteSceneOperation(Operator):
    bl_idname = "nl_control.execute_scene_operation"
    bl_label = "Execute Scene Operation"
    bl_description = "Execute scene operation(s) based on the AI's response, supports multiple operations"

    def execute(self, context):

        ai_response = context.scene.nl_control_ai_response
        if not ai_response:
            self.report({'ERROR'}, "No AI response available")
            return {'CANCELLED'}


        operations = parse_gpt_response(ai_response)
        if not operations:
            self.report({'ERROR'}, "Could not parse operation from AI response")
            return {'CANCELLED'}


        if isinstance(operations, dict) and "analysis" in operations:
            context.scene.nl_control_gpt_analysis = operations["analysis"]


        if not isinstance(operations, list):
            operations = [operations]


        region, rv3d = get_active_view3d_region(context)
        if not region or not rv3d:
            self.report({'ERROR'}, "No active 3D view found")
            return {'CANCELLED'}

        cam_world_matrix = rv3d.view_matrix.inverted()
        cam_loc = cam_world_matrix.translation
        R = cam_world_matrix.to_3x3()

        overall_results = []
        for op in operations:
            success, result = execute_scene_operation(context, op)
            overall_results.append(result)
            if not success:
                self.report({'WARNING'}, f"Operation failed: {result}")
                update_status(context, f"Operation failed: {result}")


        timestamp = get_timestamp()
        out_dir = context.scene.nl_control_props.scene_graph_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"operation_result_{timestamp}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, indent=2, ensure_ascii=False)

        update_status(context, f"Operations executed successfully. Results saved to: {out_path}")
        self.report({'INFO'}, f"Operations executed. Results saved to: {out_path}")

        return {'FINISHED'}


class NL_OT_ExecuteCommand(Operator):
    bl_idname = "object.execute_nl_command"
    bl_label = "Execute NL Command"
    bl_description = "Execute the natural language command using GPT API and support multiple operations"

    def execute(self, context):
        nl_command = context.scene.nl_command
        if not nl_command:
            self.report({'ERROR'}, "No command entered")
            return {'CANCELLED'}

        update_status(context, f"Executing command: {nl_command}")


        visible_objects = get_visible_objects_in_view(context)
        scene_graph = analyze_scene(context, visible_objects)


        object_types = extract_object_types_from_command(nl_command)
        if object_types:
            filtered_scene_graph = filter_scene_graph_for_types(scene_graph, object_types)
        else:
            filtered_scene_graph = scene_graph


        allowed_ids = {obj["id"] for obj in filtered_scene_graph["objects"]}


        surface_info = {}
        for obj in visible_objects:
            if obj.name in allowed_ids:
                top = extract_top_surface(obj)
                if top:
                    surface_info[obj.name] = {
                        "top_surface": top
                    }


        system_prompt = (
            "You are a helpful assistant for 3D scene manipulation. "
            "Note that the scene graph includes a 'new_coord' attribute for each object, which represents its position. Assisted judgment is based on the 'relation' between two objects "
            "in a transformed coordinate system: the x coordinate increases from left to right, the y coordinate is always positive with lower values meaning the object is closer to the viewer, "
            "and the z coordinate represents height, with larger values indicating higher positions. For instance: if x1>x2, then x2 is on the right; if y1>y2, then y1 is far away from the viewer (back) and y2 is in the front; "
            "if z1>z2, then the object with z1 is higher. Even if this conflicts with common sense, you MUST follow the above rules strictly. Never select an object based on subjective guesswork; "
            "the target object must be selected strictly according to the position calculation results using numbers only. Violation of this rule is a critical error. "
            "When responding to commands about manipulating objects in the scene, please output a JSON object containing two keys: 'commands' and 'analysis'.\n"
            "The 'commands' key should hold an array of operation objects, each with the following structure:\n"
            "{\n"
            "  \"action\": The operation to perform (move, rotate, copy, delete, enlarge, shrink),\n"
            "  \"target_object\": { \"id\": The ID of the object to manipulate },\n"
            "  \"anchor_object\": { \"id\": The ID of the reference object (if applicable), \"top_surface\": { \"center\": [x, y, z] } } (if applicable),\n"
            "  Additional parameters specific to the operation (e.g., position, angle, scale_factor)\n"
            "}\n\n"
            "The 'analysis' key should contain a detailed explanation of your reasoning process, including analysis of the scene graph, the detected object types, "
            "and why the particular commands were chosen. Do not include any additional text outside the JSON object."
        )

        try:
            if not openai:
                self.report({'ERROR'}, "OpenAI library not installed. Please install with 'pip install openai'")
                return {'CANCELLED'}

            client = OpenAI(
                api_key=HARD_CODED_GPT_API_KEY,
                project=HARD_CODED_GPT_PROJECT_ID
            )

            response = client.chat.completions.create(
                model=HARD_CODED_GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": json.dumps({
                            "command": nl_command,
                            "scene_graph": filtered_scene_graph,
                            "object_types": object_types,
                            "surface_info": surface_info
                        }, ensure_ascii=False)
                    }
                ],
                max_tokens=HARD_CODED_GPT_MAX_TOKENS,
                temperature=HARD_CODED_GPT_TEMPERATURE
            )
            ai_response = response.choices[0].message.content
            context.scene.nl_control_ai_response = ai_response

        except Exception as e:
            self.report({'ERROR'}, f"API call failed: {str(e)}")
            return {'CANCELLED'}


        out_dir = context.scene.nl_control_props.scene_graph_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"gpt_request_{get_timestamp()}.json")

        request_data = {
            "command": nl_command,
            "scene_graph": filtered_scene_graph,
            "object_types": object_types,
            "surface_info": surface_info,
            "ai_response": ai_response,
            "timestamp": get_timestamp()
        }

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)


        try:
            operations = parse_gpt_response(ai_response)
            if not operations:
                self.report({'WARNING'}, "Could not parse operations from AI response")
                update_status(context, "Could not parse operations from AI response")
            else:

                if not isinstance(operations, list):
                    operations = [operations]

                try:
                    overall = json.loads(ai_response)
                    if isinstance(overall, dict) and "analysis" in overall:
                        context.scene.nl_control_gpt_analysis = overall["analysis"]
                except:
                    pass


                region, rv3d = get_active_view3d_region(context)
                if not region or not rv3d:
                    self.report({'ERROR'}, "No active 3D view found")
                    return {'CANCELLED'}

                cam_world_matrix = rv3d.view_matrix.inverted()
                cam_loc = cam_world_matrix.translation
                R = cam_world_matrix.to_3x3()

                overall_results = []
                for op in operations:
                    success, result = execute_scene_operation(context, op)
                    overall_results.append(result)
                    if not success:
                        self.report({'WARNING'}, f"Operation failed: {result}")
                        update_status(context, f"Operation failed: {result}")

                # 保存所有操作结果
                result_path = os.path.join(out_dir, f"operation_result_{get_timestamp()}.json")
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(overall_results, f, indent=2, ensure_ascii=False)

                self.report({'INFO'}, f"Command executed. Results saved to: {result_path}")
                update_status(context, f"Command execution complete. Results saved to: {result_path}")
        except Exception as e:
            self.report({'ERROR'}, f"Error executing operation: {str(e)}")
            update_status(context, f"Error executing operation: {str(e)}")

        return {'FINISHED'}


class NL_OT_SaveCommandData(Operator):
    bl_idname = "nl_control.save_command_data"
    bl_label = "Save Command Data"
    bl_description = "Save all the data that would be sent to AI API for the current command"

    def execute(self, context):
        nl_command = context.scene.nl_command
        if not nl_command:
            self.report({'ERROR'}, "No command entered. Please enter a natural language command first.")
            return {'CANCELLED'}

        visible_objects = get_visible_objects_in_view(context)
        scene_graph = analyze_scene(context, visible_objects)
        object_types = extract_object_types_from_command(nl_command)
        if object_types:
            filtered_scene_graph = filter_scene_graph_for_types(scene_graph, object_types)
        else:
            filtered_scene_graph = scene_graph

        command_data = {
            "command": nl_command,
            "scene_graph": filtered_scene_graph,
            "object_types": object_types,
            "timestamp": get_timestamp()
        }

        out_dir = context.scene.nl_control_props.scene_graph_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"command_data_{get_timestamp()}.json")

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(command_data, f, indent=2, ensure_ascii=False)

        self.report({'INFO'}, f"Command data saved to: {out_path}")
        update_status(context, f"Command data saved to: {out_path}")

        return {'FINISHED'}


# Property Group for Visible Objects
class NL_VisibleObjectItem(PropertyGroup):
    name: StringProperty()
    type: StringProperty()
    semantic_type: StringProperty()


# UI Panel
class NL_PT_ControlPanel(Panel):
    bl_label = "NL Control - AI Scene Interaction"
    bl_idname = "NL_PT_ControlPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'NL Control'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.nl_control_props


        box = layout.box()
        box.label(text="Using GPT-4o", icon='URL')


        box = layout.box()
        box.label(text="Main Controls", icon='TOOL_SETTINGS')
        row = box.row()
        row.scale_y = 1.5
        row.operator("nl_control.analyze_scene", text="Analyze Scene", icon='SCENE')


        box = layout.box()
        box.label(text="Natural Language Command", icon='CONSOLE')
        box.prop(scene, "nl_command", text="")
        row = box.row()
        row.scale_y = 1.5
        split = row.split(factor=0.7)
        split.operator("object.execute_nl_command", text="Execute Command", icon='PLAY')
        split.operator("nl_control.save_command_data", text="Save Data", icon='FILE_TICK')


        if hasattr(scene, "nl_control_ai_response") and scene.nl_control_ai_response:
            box.separator()
            op_row = box.row()
            op_row.scale_y = 1.2
            op_row.operator("nl_control.execute_scene_operation", text="Execute Operation", icon='MOD_BUILD')


        if scene.nl_control_gpt_analysis:
            box = layout.box()
            box.label(text="GPT Analysis (truncated):")
            truncated_text = scene.nl_control_gpt_analysis if len(
                scene.nl_control_gpt_analysis) < 200 else scene.nl_control_gpt_analysis[:200] + "..."
            box.label(text=truncated_text)


        box = layout.box()
        box.label(text="Save Settings")
        row = box.row()
        row.label(text="Screenshots:")
        row = box.row()
        row.prop(props, "screenshot_dir", text="")
        row.operator("nl_control.set_screenshot_dir", text="", icon='FILE_FOLDER')

        row = box.row()
        row.label(text="Scene Graphs:")
        row = box.row()
        row.prop(props, "scene_graph_dir", text="")
        row.operator("nl_control.set_scene_graph_dir", text="", icon='FILE_FOLDER')


        box = layout.box()
        box.label(text="Viewport Screenshot")
        box.operator("nl_control.take_screenshot", text="Take Screenshot", icon='RENDER_STILL')
        if scene.nl_control_screenshot:
            box.label(text=f"Saved to: {os.path.basename(scene.nl_control_screenshot)}")


        box = layout.box()
        box.label(text="Visible Objects")
        box.operator("nl_control.get_visible_objects", text="Get Visible Objects", icon='VIEWZOOM')
        if len(scene.nl_control_visible_objects) > 0:
            box.label(text=f"Found {len(scene.nl_control_visible_objects)} objects:")
            for i, obj_item in enumerate(scene.nl_control_visible_objects):
                if i < 10:
                    box.label(text=f"- {obj_item.name} ({obj_item.semantic_type})")
            if len(scene.nl_control_visible_objects) > 10:
                box.label(text=f"... and {len(scene.nl_control_visible_objects) - 10} more")


        box = layout.box()
        box.label(text="Scene Graph")
        box.operator("nl_control.build_scene_graph", text="Build Scene Graph", icon='NODETREE')
        if scene.nl_control_scene_graph_path:
            box.label(text=f"Saved to: {os.path.basename(scene.nl_control_scene_graph_path)}")


        box = layout.box()
        box.label(text="Top Surface Extraction")
        row = box.row()
        row.operator("nl_control.extract_surfaces", text="Extract Top Surfaces", icon='MOD_BOOLEAN')
        row.operator("nl_control.visualize_top_surface", text="Visualize Top Surface", icon='MESH_GRID')


        box = layout.box()
        box.label(text="Supported Operations:")
        col = box.column(align=True)
        row = col.row()
        row.label(text="Move", icon='ORIENTATION_GLOBAL')
        row.label(text="Rotate", icon='DRIVER_ROTATIONAL_DIFFERENCE')
        row = col.row()
        row.label(text="Copy", icon='DUPLICATE')
        row.label(text="Delete", icon='TRASH')
        row = col.row()
        row.label(text="Enlarge", icon='FULLSCREEN_ENTER')
        row.label(text="Shrink", icon='FULLSCREEN_EXIT')


        if scene.nl_command:
            box = layout.box()
            box.label(text="Detected Objects in Command:")
            object_types = extract_object_types_from_command(scene.nl_command)
            if object_types:
                for obj_type in object_types:
                    box.label(text=f"- {obj_type}")
            else:
                box.label(text="No specific objects detected")


        layout.separator()
        layout.label(text="Status:")
        layout.label(text=scene.nl_control_status if scene.nl_control_status else "Ready")

    # Registration of Classes
classes = [
        NL_ControlProperties,
        NL_VisibleObjectItem,
        NL_PT_ControlPanel,
        NL_OT_TakeScreenshot,
        NL_OT_GetVisibleObjects,
        NL_OT_BuildSceneGraph,
        NL_OT_AnalyzeScene,
        NL_OT_SetScreenshotDir,
        NL_OT_SetSceneGraphDir,
        NL_OT_ExtractSurfaces,
        NL_OT_VisualizeTopSurface,
        NL_OT_ExecuteCommand,
        NL_OT_SaveCommandData,
        NL_OT_ExecuteSceneOperation,
    ]

def register():

        for cls in classes:
            bpy.utils.register_class(cls)


        bpy.types.Scene.nl_control_props = PointerProperty(type=NL_ControlProperties)
        bpy.types.Scene.nl_control_status = StringProperty(
            name="Status",
            description="Current status of the NL Control addon",
            default="Ready"
        )
        bpy.types.Scene.nl_control_screenshot = StringProperty(
            name="Screenshot Path",
            description="Path to last taken screenshot",
            default="",
            subtype='FILE_PATH'
        )
        bpy.types.Scene.nl_control_visible_objects = CollectionProperty(type=NL_VisibleObjectItem)
        bpy.types.Scene.nl_control_scene_graph = StringProperty(
            name="Scene Graph",
            description="JSON representation of the scene graph",
            default=""
        )
        bpy.types.Scene.nl_control_scene_graph_path = StringProperty(
            name="Scene Graph Path",
            description="Path to scene graph JSON file",
            default="",
            subtype='FILE_PATH'
        )
        bpy.types.Scene.nl_command = StringProperty(
            name="Command",
            description="Natural language command for scene manipulation",
            default=""
        )

        bpy.types.Scene.nl_control_ai_response = StringProperty(
            name="AI Response",
            description="Response from GPT API",
            default=""
        )

        bpy.types.Scene.nl_control_gpt_analysis = StringProperty(
            name="GPT Analysis",
            description="Detailed analysis output from GPT",
            default=""
        )

def unregister():

        del bpy.types.Scene.nl_control_gpt_analysis
        del bpy.types.Scene.nl_control_ai_response
        del bpy.types.Scene.nl_control_props
        del bpy.types.Scene.nl_control_status
        del bpy.types.Scene.nl_control_screenshot
        del bpy.types.Scene.nl_control_visible_objects
        del bpy.types.Scene.nl_control_scene_graph
        del bpy.types.Scene.nl_control_scene_graph_path
        del bpy.types.Scene.nl_command


        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)

if __name__ == "__main__":
        register()