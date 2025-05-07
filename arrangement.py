bl_info = {
    "name": "Object Arrangement within Floor Boundary",
    "author": "AI Assistant",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "3D View > Sidebar > Object Arrangement",
    "description": "Automatically arrange objects within floor boundary and prevent collisions",
    "category": "3D View"
}

import bpy
import bmesh
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
import math
from mathutils import Vector
from datetime import datetime
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import StringProperty, PointerProperty, EnumProperty, IntProperty, FloatProperty, BoolProperty
from scipy.spatial.distance import pdist, squareform
from matplotlib.path import Path


def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def find_boundary_points_nearest_neighbor(points):
    points = np.array(points)
    if len(points) < 2:
        return points
    dist_matrix = squareform(pdist(points))
    n = len(points)
    path = [0]
    unused = set(range(1, n))
    while unused:
        current = path[-1]
        nearest = min(unused, key=lambda x: dist_matrix[current][x])
        path.append(nearest)
        unused.remove(nearest)
    path.append(path[0])
    return points[path]


def extract_exposed_edges(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.transform(bm, matrix=obj.matrix_world, verts=bm.verts)
    bm.edges.ensure_lookup_table()
    boundary_edges = [e for e in bm.edges if e.is_boundary]
    point_set = set()
    for e in boundary_edges:
        point_set.add((e.verts[0].co.x, e.verts[0].co.y))
        point_set.add((e.verts[1].co.x, e.verts[1].co.y))
    bm.free()
    if not point_set:
        return np.array([])
    return find_boundary_points_nearest_neighbor(list(point_set))


def extract_object_data(obj):
    """Extract object data: center, bounding box and orientation"""
    # Center point
    center = [obj.location.x, obj.location.y]

    # Get rotated bounding box vertices (bottom)
    world_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    zs = [v.z for v in world_corners]
    min_z = min(zs)
    bottom = [v for v in world_corners if abs(v.z - min_z) < 1e-4]
    bbox = [[v.x, v.y] for v in bottom]

    # Ensure correct vertex order (sorted by angle)
    bbox_center = np.mean(bbox, axis=0)
    bbox.sort(key=lambda p: math.atan2(p[1] - bbox_center[1], p[0] - bbox_center[0]))

    # Z-axis rotation angle
    orientation = obj.rotation_euler.z

    return center, bbox, orientation


def extract_obstacle_data(obj):
    """Extract obstacle data: center, bounding box and orientation"""
    # Center point
    center = [obj.location.x, obj.location.y]

    # Get rotated bounding box vertices (bottom)
    world_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    zs = [v.z for v in world_corners]
    min_z = min(zs)
    bottom = [v for v in world_corners if abs(v.z - min_z) < 1e-4]
    bbox = [[v.x, v.y] for v in bottom]

    # Ensure correct vertex order (sorted by angle)
    bbox_center = np.mean(bbox, axis=0)
    bbox.sort(key=lambda p: math.atan2(p[1] - bbox_center[1], p[0] - bbox_center[0]))

    # Z-axis rotation angle
    orientation = obj.rotation_euler.z

    return center, bbox, orientation


def is_rect_inside(bbox, boundary_path):
    if not all(boundary_path.contains_point(corner) for corner in bbox):
        return False

    edges = []
    for i in range(len(bbox)):
        j = (i + 1) % len(bbox)
        mid_point = [(bbox[i][0] + bbox[j][0]) / 2, (bbox[i][1] + bbox[j][1]) / 2]
        edges.append(mid_point)

    return all(boundary_path.contains_point(edge) for edge in edges)


def aabb_distance(bbox1, bbox2):
    """

    """

    bbox1_min = np.min(bbox1, axis=0)
    bbox1_max = np.max(bbox1, axis=0)
    bbox2_min = np.min(bbox2, axis=0)
    bbox2_max = np.max(bbox2, axis=0)

    dx = max(0, max(bbox1_min[0] - bbox2_max[0], bbox2_min[0] - bbox1_max[0]))
    dy = max(0, max(bbox1_min[1] - bbox2_max[1], bbox2_min[1] - bbox1_max[1]))

    return np.sqrt(dx * dx + dy * dy)


def generate_target_positions(centers, bboxes, orientations, boundary_vertices, mode, center_ref=None):
    num_objects = len(centers)
    boundary_path = Path(boundary_vertices)

    if center_ref is None:
        center_ref = np.mean(boundary_vertices, axis=0)

    if mode == "line":

        x_coords = np.linspace(center_ref[0] - 4, center_ref[0] + 4, num_objects)
        y_coords = np.full(num_objects, center_ref[1])
        targets = np.stack((x_coords, y_coords), axis=-1)

    elif mode == "circle":

        theta = np.linspace(0, 2 * np.pi, num_objects, endpoint=False)

        max_dim = max(
            np.max(boundary_vertices[:, 0]) - np.min(boundary_vertices[:, 0]),
            np.max(boundary_vertices[:, 1]) - np.min(boundary_vertices[:, 1])
        )
        radius = max_dim * 0.3

        targets = np.stack((
            center_ref[0] + radius * np.cos(theta),
            center_ref[1] + radius * np.sin(theta)
        ), axis=-1)

    elif mode == "grid":

        grid_size = math.ceil(math.sqrt(num_objects))
        spacing = 2.0

        targets = []
        for i in range(num_objects):
            row = i // grid_size
            col = i % grid_size
            x = center_ref[0] - (grid_size - 1) * spacing / 2 + col * spacing
            y = center_ref[1] - (grid_size - 1) * spacing / 2 + row * spacing
            targets.append([x, y])
        targets = np.array(targets)

    else:  # random

        targets = []
        min_x, min_y = np.min(boundary_vertices, axis=0) + 1
        max_x, max_y = np.max(boundary_vertices, axis=0) - 1

        while len(targets) < num_objects:
            candidate = np.random.uniform(low=[min_x, min_y], high=[max_x, max_y])
            if boundary_path.contains_point(candidate):
                targets.append(candidate)
        targets = np.array(targets)

    valid_targets = []
    for i, target in enumerate(targets):

        predicted_bbox = []
        bbox_center = np.mean(bboxes[i], axis=0)
        for point in bboxes[i]:
            offset = point - centers[i]

            predicted_bbox.append(target + offset)

        if is_rect_inside(predicted_bbox, boundary_path):
            valid_targets.append(target)
        else:

            for _ in range(10):

                direction = center_ref - target
                distance = np.linalg.norm(direction)
                if distance > 0:
                    direction = direction / distance
                target = target + direction * 0.5

                predicted_bbox = []
                for point in bboxes[i]:
                    offset = point - centers[i]
                    predicted_bbox.append(target + offset)

                if is_rect_inside(predicted_bbox, boundary_path):
                    valid_targets.append(target)
                    break
            else:

                valid_targets.append(centers[i])

    return np.array(valid_targets)


def apply_arrangement(objects, floor_obj, mode="circle", iterations=100, use_center_ref=True, obstacle_objects=None,
                      destination_object=None):
    """Apply arrangement algorithm"""
    # Extract floor boundary
    vertices = extract_exposed_edges(floor_obj)
    if vertices.size < 3:
        return False, "Unable to form a closed boundary"

    # Extract all object data
    centers, bboxes, orientations = [], [], []
    destination_index = -1

    dest_in_selection = False
    if destination_object and destination_object.type == 'MESH' and destination_object != floor_obj:
        for i, obj in enumerate(objects):
            if obj == destination_object:
                dest_in_selection = True
                destination_index = i
                break

    if not dest_in_selection:
        destination_object = None
        destination_index = -1

    i = 0
    for obj in objects:
        if obj.type == 'MESH' and obj != floor_obj:
            center, bbox, orientation = extract_object_data(obj)
            centers.append(center)
            bboxes.append(bbox)
            orientations.append(orientation)
            i += 1

    if not centers:
        return False, "No objects selected for arrangement"

    # Extract obstacle data if provided
    obstacle_centers = []
    obstacle_bboxes = []
    if obstacle_objects:
        for obs_obj in obstacle_objects:
            if obs_obj.type == 'MESH' and obs_obj != floor_obj and obs_obj not in objects:
                obs_center, obs_bbox, _ = extract_obstacle_data(obs_obj)
                obstacle_centers.append(obs_center)
                obstacle_bboxes.append(obs_bbox)

    # Convert to NumPy arrays
    centers = np.array(centers)
    original_centers = centers.copy()  # Keep original positions
    original_bboxes = bboxes.copy()  # Keep original bounding boxes

    # Set reference center (if needed)
    center_ref = None
    if use_center_ref:
        center_ref = np.mean(vertices, axis=0)

    # Generate target positions
    target_positions = generate_target_positions(
        centers, bboxes, orientations, vertices, mode, center_ref
    )

    # Apply force iterations to update positions
    boundary_path = Path(vertices)
    max_force = 0.3  # Maximum force per iteration

    # First check if any objects are outside boundary, move them in if necessary
    for i in range(len(centers)):

        if i == destination_index:
            continue

        # Check if current position is valid
        current_bbox = []
        for point in bboxes[i]:
            offset = point - centers[i]
            current_bbox.append(centers[i] + offset)

        if not is_rect_inside(current_bbox, boundary_path):
            # If outside, move toward center
            boundary_center = np.mean(vertices, axis=0)
            dir_to_center = boundary_center - centers[i]
            dist = np.linalg.norm(dir_to_center)

            if dist > 0:
                # Try different step sizes until object is inside
                for scale in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                    new_center = centers[i] + dir_to_center / dist * scale * 5.0

                    test_bbox = []
                    for point in bboxes[i]:
                        offset = point - centers[i]
                        test_bbox.append(new_center + offset)

                    if is_rect_inside(test_bbox, boundary_path):
                        centers[i] = new_center
                        break

    # Apply force iterations
    success = True
    for _ in range(iterations):
        # Calculate forces
        forces = np.zeros_like(centers)

        # Target force
        for i in range(len(centers)):

            if i == destination_index:
                continue

            target_dir = target_positions[i] - centers[i]
            dist = np.linalg.norm(target_dir)
            if dist > 0.05:
                forces[i] += target_dir / dist * min(dist * 0.1, max_force)

        # Repulsion between objects
        for i in range(len(centers)):

            if i == destination_index:
                continue

            for j in range(len(centers)):
                if i != j:
                    # Calculate current bounding boxes
                    bbox_i = []
                    for point in bboxes[i]:
                        offset = point - original_centers[i]
                        bbox_i.append(centers[i] + offset)

                    bbox_j = []
                    for point in bboxes[j]:
                        offset = point - original_centers[j]
                        bbox_j.append(centers[j] + offset)

                    d = aabb_distance(bbox_i, bbox_j)

                    min_dist = 0.3
                    if d < min_dist:
                        dir_vec = centers[i] - centers[j]
                        dist = np.linalg.norm(dir_vec)
                        if dist > 0:

                            if j == destination_index:
                                repel_force = dir_vec / dist * (min_dist - d) * 0.8
                            else:
                                repel_force = dir_vec / dist * (min_dist - d) * 0.5
                            forces[i] += repel_force

        if obstacle_bboxes:
            for i in range(len(centers)):

                if i == destination_index:
                    continue

                bbox_i = []
                for point in bboxes[i]:
                    offset = point - original_centers[i]
                    bbox_i.append(centers[i] + offset)

                for j, obs_bbox in enumerate(obstacle_bboxes):
                    d = aabb_distance(bbox_i, obs_bbox)

                    min_dist = 0.5
                    if d < min_dist:

                        dir_vec = centers[i] - obstacle_centers[j]
                        dist = np.linalg.norm(dir_vec)
                        if dist > 0:
                            repel_force = dir_vec / dist * (min_dist - d) * 0.8
                            forces[i] += repel_force

        # Apply forces and check boundary
        new_centers = centers.copy()
        for i in range(len(centers)):

            if i == destination_index:
                continue

            # Limit force magnitude
            force_mag = np.linalg.norm(forces[i])
            if force_mag > max_force:
                forces[i] = forces[i] / force_mag * max_force

            # Apply force
            new_center = centers[i] + forces[i]

            # Check if new position is valid
            test_bbox = []
            for point in bboxes[i]:
                offset = point - original_centers[i]
                test_bbox.append(new_center + offset)

            if is_rect_inside(test_bbox, boundary_path):
                # Now check for collision with obstacles
                collision_with_obstacle = False
                for obs_bbox in obstacle_bboxes:
                    if aabb_distance(test_bbox, obs_bbox) < 0.1:  # Minimum separation from obstacles
                        collision_with_obstacle = True
                        break

                if not collision_with_obstacle:
                    new_centers[i] = new_center
                else:
                    # Try reduced force if colliding with obstacle
                    for scale in [0.8, 0.6, 0.4, 0.2, 0.1, 0.05]:
                        scaled_center = centers[i] + forces[i] * scale

                        scaled_bbox = []
                        for point in bboxes[i]:
                            offset = point - original_centers[i]
                            scaled_bbox.append(scaled_center + offset)

                        # Check both boundary and obstacles
                        if is_rect_inside(scaled_bbox, boundary_path):
                            collision = False
                            for obs_bbox in obstacle_bboxes:
                                if aabb_distance(scaled_bbox, obs_bbox) < 0.1:
                                    collision = True
                                    break

                            if not collision:
                                new_centers[i] = scaled_center
                                break
            else:
                # Try reduced force if outside boundary
                for scale in [0.8, 0.6, 0.4, 0.2, 0.1, 0.05]:
                    scaled_center = centers[i] + forces[i] * scale

                    scaled_bbox = []
                    for point in bboxes[i]:
                        offset = point - original_centers[i]
                        scaled_bbox.append(scaled_center + offset)

                    if is_rect_inside(scaled_bbox, boundary_path):
                        # Check obstacles too
                        collision = False
                        for obs_bbox in obstacle_bboxes:
                            if aabb_distance(scaled_bbox, obs_bbox) < 0.1:
                                collision = True
                                break

                        if not collision:
                            new_centers[i] = scaled_center
                            break

        # Check if we've converged
        moving_indices = [i for i in range(len(centers)) if i != destination_index]
        if len(moving_indices) > 0:
            converged = np.all(np.linalg.norm(new_centers[moving_indices] - centers[moving_indices], axis=1) < 0.01)
            if converged:
                break

        centers = new_centers

    # Final check to ensure all objects are inside and not colliding with obstacles or other objects
    all_valid = True
    for i in range(len(centers)):

        if i == destination_index:
            continue

        final_bbox = []
        for point in bboxes[i]:
            offset = point - original_centers[i]
            final_bbox.append(centers[i] + offset)

        if not is_rect_inside(final_bbox, boundary_path):
            all_valid = False
            break

        for obs_bbox in obstacle_bboxes:
            if aabb_distance(final_bbox, obs_bbox) < 0.1:
                all_valid = False
                break

        collision_found = False
        for j in range(len(centers)):
            if i != j:
                bbox_j = []
                for point in bboxes[j]:
                    offset = point - original_centers[j]
                    bbox_j.append(centers[j] + offset)

                if aabb_distance(final_bbox, bbox_j) < 0.1:
                    collision_found = False
                    break

        if collision_found:
            all_valid = False
            break

    if not all_valid:
        success = False
        # Revert to original positions
        centers = original_centers

    # Apply new positions to objects
    i = 0
    for obj in objects:
        if obj.type == 'MESH' and obj != floor_obj:

            if obj != destination_object:
                obj.location.x = centers[i][0]
                obj.location.y = centers[i][1]
            i += 1

    # Update bounding boxes based on new positions
    updated_bboxes = []
    for i, (center, original_bbox) in enumerate(zip(centers, original_bboxes)):
        updated_box = []
        for point in original_bbox:
            offset = point - original_centers[i]
            updated_box.append(center + offset)
        updated_bboxes.append(updated_box)

    return success, centers, updated_bboxes, orientations, obstacle_centers, obstacle_bboxes, destination_index


def save_arrangement_result(vertices, centers, bboxes, orientations, floor_name, output_dir, obstacle_centers=None,
                            obstacle_bboxes=None, destination_index=-1):
    """Save arrangement result"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = get_timestamp()
    json_path = os.path.join(output_dir, f"{floor_name}_arrangement_{timestamp}.json")

    # Convert data to serializable format
    data = {
        "name": floor_name,
        "floor_vertices": vertices.tolist() if isinstance(vertices, np.ndarray) else [],
        "object_centers": centers.tolist() if isinstance(centers, np.ndarray) else [],
        "object_bboxes": [[list(point) for point in box] for box in bboxes] if bboxes else [],
        "object_orientations": orientations if orientations else [],
        "destination_index": destination_index
    }

    # Add obstacle data if available
    if obstacle_centers is not None and obstacle_bboxes is not None:
        data["obstacle_centers"] = obstacle_centers.tolist() if isinstance(obstacle_centers, np.ndarray) else [list(c)
                                                                                                               for c in
                                                                                                               obstacle_centers]
        data["obstacle_bboxes"] = [[list(point) for point in box] for box in obstacle_bboxes] if obstacle_bboxes else []

    # Save JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    # Generate visualization
    png_path = None
    if len(vertices) >= 3:
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw floor boundary
        ax.plot(vertices[:, 0], vertices[:, 1], 'b-', linewidth=2, label='Floor Boundary')

        # Draw obstacles if available
        if obstacle_bboxes:
            for i, obs_bbox in enumerate(obstacle_bboxes):
                poly = np.array(obs_bbox)
                patch = Polygon(poly, closed=True, fill=True, alpha=0.3, linewidth=1.5,
                                linestyle='-', edgecolor='red', facecolor='red')
                ax.add_patch(patch)

                # Label obstacle
                obs_center = np.mean(poly, axis=0)
                ax.annotate(f"Obs{i + 1}", (obs_center[0], obs_center[1]), color='darkred',
                            ha='center', va='center', weight='bold')

        # Draw object bounding boxes and orientation
        for i, (center, bbox, orientation) in enumerate(zip(centers, bboxes, orientations)):
            # Draw bounding box
            poly = np.array(bbox)

            if i == destination_index:

                patch = Polygon(poly, closed=True, fill=True, alpha=0.5, linewidth=1.5,
                                linestyle='-', edgecolor='green', facecolor='green')
                print(f"Draw the end point object {i} in green")
            else:

                patch = Polygon(poly, closed=True, fill=True, alpha=0.5, linewidth=1.5,
                                linestyle='-', edgecolor='yellow', facecolor='yellow')

            ax.add_patch(patch)

            # Label object number
            box_center = np.mean(poly, axis=0)
            if i == destination_index:
                ax.annotate(f"Dest{i + 1}", (box_center[0], box_center[1]), color='darkgreen',
                            ha='center', va='center', weight='bold')
            else:
                ax.annotate(f"Obj{i + 1}", (box_center[0], box_center[1]), color='darkorange',
                            ha='center', va='center')

            # Draw center point
            ax.scatter(center[0], center[1], s=30, c='red')

        ax.set_title(f"{floor_name} Floor Boundary and Object Arrangement")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis('equal')
        ax.grid(True)

        png_path = os.path.join(output_dir, f"{floor_name}_arrangement_{timestamp}.png")
        fig.savefig(png_path, dpi=300)
        plt.close(fig)

    return json_path, png_path


class ObjectArrangementProps(PropertyGroup):
    output_dir: StringProperty(
        name="Output Path",
        subtype='DIR_PATH',
        default=""
    )

    floor_object: PointerProperty(
        name="Floor Object",
        type=bpy.types.Object,
        description="Explicitly specify the object to use as floor (Mesh)"
    )

    destination_object: PointerProperty(
        name="Destination Object",
        type=bpy.types.Object,
        description="Specify a destination object that doesn't move (shown in green)"
    )

    obstacle_collection: PointerProperty(
        name="Obstacle Collection",
        type=bpy.types.Collection,
        description="Collection containing obstacles (objects that won't be arranged but will be avoided)"
    )

    arrangement_mode: EnumProperty(
        name="Arrangement Mode",
        items=[
            ('circle', "Circular Arrangement", "Arrange objects in a circle"),
            ('line', "Linear Arrangement", "Arrange objects in a line"),
            ('grid', "Grid Arrangement", "Arrange objects in a grid"),
            ('random', "Random Arrangement", "Randomly arrange objects (within boundary)")
        ],
        default='circle'
    )

    iterations: IntProperty(
        name="Iteration Count",
        description="Number of force simulation iterations",
        default=100,
        min=10,
        max=500
    )

    use_center_ref: BoolProperty(
        name="Use Center Reference",
        description="Use floor boundary center as reference point",
        default=True
    )


class OBJECT_OT_ArrangeInFloor(Operator):
    bl_idname = "object.arrange_in_floor"
    bl_label = "Arrange Objects"

    def execute(self, context):
        props = context.scene.object_arrangement_props
        output_dir = props.output_dir or bpy.path.abspath("//")
        floor_obj = props.floor_object
        obstacle_collection = props.obstacle_collection
        destination_object = props.destination_object

        if not floor_obj or floor_obj.type != 'MESH':
            self.report({'WARNING'}, "Please select a valid mesh object as floor!")
            return {'CANCELLED'}

        # Check destination object
        if destination_object:
            if destination_object.type != 'MESH':
                self.report({'WARNING'}, "Destination object must be a mesh!")
                destination_object = None
            elif destination_object == floor_obj:
                self.report({'WARNING'}, "Destination object cannot be the floor!")
                destination_object = None
            else:
                # Ensure destination object is in selection
                if destination_object not in context.selected_objects:
                    self.report({'INFO'}, "Adding destination object to selection.")
                    destination_object.select_set(True)

        # Get objects to arrange (selected objects, excluding floor and obstacles)
        objects_to_arrange = [obj for obj in context.selected_objects
                              if obj.type == 'MESH' and obj != floor_obj]

        if not objects_to_arrange:
            self.report({'WARNING'}, "Please select objects to arrange!")
            return {'CANCELLED'}

        # Extract floor boundary
        vertices = extract_exposed_edges(floor_obj)
        if vertices.size < 3:
            self.report({'WARNING'}, f"{floor_obj.name} cannot form a closed boundary!")
            return {'CANCELLED'}

        # Get obstacle objects if collection is specified
        obstacle_objects = []
        if obstacle_collection:
            obstacle_objects = [obj for obj in obstacle_collection.objects
                                if obj.type == 'MESH' and obj != floor_obj and obj not in objects_to_arrange]

        # Apply arrangement algorithm
        result = apply_arrangement(
            objects_to_arrange,
            floor_obj,
            mode=props.arrangement_mode,
            iterations=props.iterations,
            use_center_ref=props.use_center_ref,
            obstacle_objects=obstacle_objects,
            destination_object=destination_object
        )

        if isinstance(result, tuple) and len(result) > 0 and not result[0]:
            if isinstance(result[1], str):
                self.report({'WARNING'}, result[1])
            else:
                self.report({'WARNING'},
                            "Failed to arrange objects within boundary. Please try with fewer objects or a larger floor.")
            return {'CANCELLED'}

        # Get result data
        _, centers, bboxes, orientations, obstacle_centers, obstacle_bboxes, destination_index = result

        # Save result
        json_path, png_path = save_arrangement_result(
            vertices, centers, bboxes, orientations, floor_obj.name, output_dir,
            obstacle_centers, obstacle_bboxes, destination_index
        )

        self.report({'INFO'}, f"Object arrangement complete! Saved to {output_dir}")
        return {'FINISHED'}


class OBJECT_PT_ArrangementPanel(Panel):
    bl_label = "Object Auto Arrangement"
    bl_idname = "OBJECT_PT_ArrangementPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Object Arrangement'

    def draw(self, context):
        layout = self.layout
        props = context.scene.object_arrangement_props

        layout.prop(props, "floor_object")
        layout.prop(props, "destination_object")
        layout.prop(props, "obstacle_collection")
        layout.prop(props, "arrangement_mode")
        layout.prop(props, "iterations")
        layout.prop(props, "use_center_ref")
        layout.prop(props, "output_dir")

        layout.separator()
        op = layout.operator("object.arrange_in_floor", icon='MESH_GRID')


def register():
    for cls in (ObjectArrangementProps, OBJECT_OT_ArrangeInFloor, OBJECT_PT_ArrangementPanel):
        bpy.utils.register_class(cls)
    bpy.types.Scene.object_arrangement_props = PointerProperty(type=ObjectArrangementProps)


def unregister():
    for cls in reversed((ObjectArrangementProps, OBJECT_OT_ArrangeInFloor, OBJECT_PT_ArrangementPanel)):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.object_arrangement_props


if __name__ == "__main__":
    register()