from typing import List
from pathlib import Path

import numpy as np
import open3d as o3d

from mixedsignals.utils.bbox_utils import get_boxes_vertices_coord

# ---------------------------------------------------
# LineMesh hotfix  to get thicker 3d bounding box lines
# Source: https://github.com/isl-org/Open3D/pull/738
# ---------------------------------------------------

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    if np.linalg.norm(axis_) > 0:
        axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes
        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.
        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
              
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


class PointPainter(object):
    def __init__(self) -> None:
        self.objs_to_draw = list()


    def add_pointclouds_(self, 
                         pc: np.ndarray, 
                         colors: np.ndarray = None,
                         voxel_size: float = 0.2) -> None:
        assert pc.shape[1] >= 3, f"{pc.shape}, pc.shape[1] not >= 3"
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc[:, :3])

        if colors is None:
            o3d_pc.colors = o3d.utility.Vector3dVector(np.zeros((pc.shape[0], 3)) + 0.44)
        else:
            assert isinstance(colors, np.ndarray), f"{type(colors)} != np.ndarray"
            if colors.shape == (3,):
                o3d_pc.colors = o3d.utility.Vector3dVector(np.zeros((pc.shape[0], 3)) + colors)
            elif colors.shape == (pc.shape[0], 3):
                o3d_pc.colors = o3d.utility.Vector3dVector(colors)
            else:
                raise ValueError(f"{colors.shape} is neither (3,) nor (num_points, 3)")

        o3d_pc = o3d_pc.voxel_down_sample(voxel_size=voxel_size)

        self.objs_to_draw.append(o3d_pc)
        return
    
    def add_boxes_(self, 
                  boxes: np.ndarray, 
                  colors: np.ndarray = None,
                  edge_thickness: float = 0.06) -> None:
        assert boxes.shape[1] >= 7, f"{boxes.shape}, boxes.shape[1] must >= 7"

        if colors is None:
            colors = np.zeros((boxes.shape[0], 3)) + np.array([1, 0, 0])
        else:
            if colors.shape == (3,):
                colors = np.zeros((boxes.shape[0], 3)) + colors
            elif colors.shape == (boxes.shape[0], 3):
                pass  # already good
            else:
                raise ValueError(f"{colors.shape} is neither (3,) nor (num_boxes, 3)")
        
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # front
            [4, 5], [5, 6], [6, 7], [7, 4],  # back
            [0, 4], [1, 5], [2, 6], [3, 7],  # connecting front & back
            [0, 2], [1, 3]  # denote forward face
        ]

        for ii, box in enumerate(boxes):
            vers = get_boxes_vertices_coord(box.reshape(1, -1))[0]

            cube = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(vers),
                lines=o3d.utility.Vector2iVector(lines),
            )
            
            line_mesh = LineMesh(cube.points, cube.lines, colors[ii], radius=edge_thickness) 
            line_mesh_geoms = line_mesh.cylinder_segments

            # add this box to object to draw
            self.objs_to_draw.extend(line_mesh_geoms)

        return

    def show(self, view_points: dict = None, save_to_path: Path = None):
        if save_to_path is not None:
            assert view_points is not None

        if view_points is not None:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.clear_geometries()
            
            for obj in self.objs_to_draw:
                vis.add_geometry(obj)

            vis.set_view_status(view_points)
            vis.get_render_option().point_size = 1.0
            vis.poll_events()
            vis.update_renderer()

            if save_to_path is not None:
                vis.capture_screen_image(save_to_path)
            else:
                vis.run()
        else:
            o3d.visualization.draw_geometries(self.objs_to_draw)

    def show_video(self,):
        pass  # TODO
