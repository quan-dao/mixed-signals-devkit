from typing import Union, List
import numpy as np
from pyquaternion import Quaternion


def rotation_matrix_to_yaw(rot: np.ndarray) -> float:
    return np.arctan2(rot[1, 0], rot[0, 0])


def apply_se3_(se3_tf: np.ndarray, 
               points_: np.ndarray = None, 
               boxes_: np.ndarray = None, boxes_has_velocity: bool = False, 
               vector_: np.ndarray = None) -> None:
    """
    Inplace function

    Args:
        se3_tf: (4, 4) - homogeneous transformation
        points_: (N, 3 + C) - x, y, z, [others]
        boxes_: (N, 7 + 2 + C) - x, y, z, dx, dy, dz, yaw, [vx, vy], [others]
        boxes_has_velocity: make boxes_velocity explicit
        vector_: (N, 2 [+ 1]) - x, y, [z]
    """
    if points_ is not None:
        assert points_.shape[1] >= 3, f"points_.shape: {points_.shape}"
        points_[:, :3] = points_[:, :3] @  se3_tf[:3, :3].T + se3_tf[:3, -1]

    if boxes_ is not None:
        assert boxes_.shape[1] >= 7, f"boxes_.shape: {boxes_.shape}"
        boxes_[:, :3] = boxes_[:, :3] @  se3_tf[:3, :3].T + se3_tf[:3, -1]
        boxes_[:, 6] += rotation_matrix_to_yaw(se3_tf[:3, :3])
        boxes_[:, 6] = np.arctan2(np.sin(boxes_[:, 6]), np.cos(boxes_[:, 6]))
        if boxes_has_velocity:
            boxes_velo = np.pad(boxes_[:, 7: 9], pad_width=[(0, 0), (0, 1)], constant_values=0.0)  # (N, 3) - vx, vy, vz
            boxes_velo = boxes_velo @ se3_tf[:3, :3].T
            boxes_[:, 7: 9] = boxes_velo[:, :2]

    if vector_ is not None:
        if vector_.shape[1] == 2:
            vector = np.pad(vector_, pad_width=[(0, 0), (0, 1)], constant_values=0.)
            vector_[:, :2] = (vector @ se3_tf[:3, :3].T)[:, :2]
        else:
            assert vector_.shape[1] == 3, f"vector_.shape: {vector_.shape}"
            vector_[:, :3] = vector_ @ se3_tf[:3, :3].T

    return


def apply_se3(se3_tf: np.ndarray, 
              points_: np.ndarray = None, 
              boxes_: np.ndarray = None, 
              boxes_has_velocity: bool = False, 
              vector_: np.ndarray = None) -> np.ndarray:
    """
    Non-inplace version of apply_se3_
    """
    points = points_.copy() if points_ is not None else None
    boxes = boxes_.copy() if boxes_ is not None else None
    vectors = vector_.copy() if vector_ is not None else None
    apply_se3_(se3_tf, 
               points_=points, 
               boxes_=boxes, 
               vector_=vectors, 
               boxes_has_velocity=boxes_has_velocity)
    out = [ele for ele in [points, boxes, vectors] if ele is not None]
    if len(out) == 1:
        return out[0]
    else:
        return out


def make_se3(translation: Union[List[float], np.ndarray], yaw: float = None, rotation_matrix: np.ndarray = None, 
             quaternion: Union[List[float], np.ndarray] = None):
    if yaw is not None:
        assert quaternion is None
        assert rotation_matrix is None
        rotation_matrix = make_rotation_around_z(yaw)
    elif quaternion is not None:
        assert yaw is None
        assert rotation_matrix is None
        assert isinstance(quaternion, list) or isinstance(quaternion, np.ndarray), f"{type(quaternion)} is not supported"
        rotation_matrix = Quaternion(quaternion).rotation_matrix
    else:
        assert rotation_matrix is not None
        
    
    out = np.zeros((4, 4))
    out[-1, -1] = 1.0

    out[:3, :3] = rotation_matrix

    if not isinstance(translation, np.ndarray):
        translation = np.array(translation)
    out[:3, -1] = translation.reshape(3)

    return out


def make_rotation_around_z(yaw: float) -> np.ndarray:
    cos, sin = np.cos(yaw), np.sin(yaw)
    out = np.array([
        [cos, -sin, 0.],
        [sin, cos, 0.],
        [0., 0., 1.]
    ])
    return out


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def linear_interp(time_query: float, 
                  val_start: float, 
                  val_end: float, 
                  time_start: float, 
                  time_end: float) -> float:
    assert (time_end - time_start) > 0.0, f"{time_end - time_start} not > 0.0"
    assert (time_query - time_start) >= 0.0, f"{time_query - time_start} not >= 0.0"

    val_query = val_start + \
        (val_end - val_start) * (time_query - time_start) / (time_end - time_start)
    return val_query


def interpolate_yaw(time_query: float,
                    yaw_start: float,
                    yaw_end: float,
                    time_start: float,
                    time_end: float) -> float:
    q0 = Quaternion(axis=[0, 0, 1], angle=yaw_start)
    q1 = Quaternion(axis=[0, 0, 1], angle=yaw_end)
    q = Quaternion.slerp(q0, q1, amount=(time_query - time_start) / (time_end - time_start))
    rot_mat = q.rotation_matrix
    '''
    [c -s 0]
    [s c  0]
    [0 0 1]
    '''
    yaw = np.arctan2(-rot_mat[0, 1], rot_mat[0, 0])
    return yaw
